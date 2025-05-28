import uuid
import json
from typing import Dict, List
from graphviz import Digraph

class TreeNode:
    # 静态字典，存储 (level, name) 到 id 的映射
    _level_name_to_id = {}

    def __init__(self, name: str, is_entity: bool = False, level: int = 0, r_relevance: str = None, e_relevance: str = None):
        self.name = name
        self.is_entity = is_entity
        self.level = level
        self.r_relevance = r_relevance  # Relevance for relation (for edges to this node if relation)
        self.e_relevance = e_relevance  # Relevance for entity (for edges from this node if entity)
        # 为实体分配 ID，同一 level 的同名实体共享 ID
        if is_entity:
            key = (level, name)
            if key in TreeNode._level_name_to_id:
                self.id = TreeNode._level_name_to_id[key]
            else:
                self.id = str(uuid.uuid4())
                TreeNode._level_name_to_id[key] = self.id
        else:
            self.id = name  # 关系节点使用名称作为 ID
        self.children = {}

    def add_child(self, name: str, is_entity: bool = False, r_relevance: str = None, e_relevance: str = None) -> 'TreeNode':
        # 计算子节点的 level
        child_level = self.level + 1
        # 为关系节点使用名称作为 key，为实体节点使用 ID 作为 key
        key = name if not is_entity else str(TreeNode._get_entity_id(child_level, name))
        if key not in self.children:
            self.children[key] = TreeNode(name, is_entity, child_level, r_relevance, e_relevance)
        else:
            # 更新现有节点的 relevance（如果提供）
            if r_relevance and not is_entity:
                self.children[key].r_relevance = r_relevance
            if e_relevance and is_entity:
                self.children[key].e_relevance = e_relevance
        return self.children[key]

    @staticmethod
    def _get_entity_id(level: int, name: str) -> str:
        """获取或生成指定 level 和 name 的实体 ID。"""
        key = (level, name)
        if key not in TreeNode._level_name_to_id:
            TreeNode._level_name_to_id[key] = str(uuid.uuid4())
        return TreeNode._level_name_to_id[key]

    def __repr__(self, level=0):
        ret = "  " * level + f"{self.name} (id={self.id}, level={self.level}"
        if self.r_relevance:
            ret += f", r_relevance={self.r_relevance}"
        if self.e_relevance:
            ret += f", e_relevance={self.e_relevance}"
        ret += ")\n"
        for child in self.children.values():
            ret += child.__repr__(level + 1)
        return ret

def parse_triple(triple: str) -> tuple[str, str, str]:
    """Parse a triple string into (entity1, relation, entity2)."""
    triple = triple.strip('()').split(',')
    return triple[0].strip(), triple[1].strip(), triple[2].strip()

def find_node(node: TreeNode, target_name: str, is_entity: bool, path: List[str] = None) -> TreeNode:
    """Recursively find the deepest node by name, type, and level, avoiding cycles."""
    if path is None:
        path = []
    result = None
    if node.name == target_name and node.is_entity == is_entity and node.id not in path:
        result = node
    path.append(node.id)
    for child in node.children.values():
        child_result = find_node(child, target_name, is_entity, path.copy())
        if child_result:
            result = child_result  # Keep the deepest match
    return result

def add_path_to_tree(tree: TreeNode, entity1: str, relation: str, entity2: str, r_match: Dict, e_match: Dict):
    """Add a path (entity1 -> relation -> entity2) to the tree with relevance."""
    parent = find_node(tree, entity1, is_entity=True)
    if not parent:
        return  # Skip if entity1 not found

    # 获取关系和实体的 relevance
    r_relevance = r_match.get(relation, {}).get('relevance', None)
    e_relevance = e_match.get(entity2, None)

    relation_node = parent.add_child(relation, is_entity=False, r_relevance=r_relevance)
    relation_node.add_child(entity2, is_entity=True, e_relevance=e_relevance)

def build_tree(data: Dict) -> TreeNode:
    """Build a tree from the verbose list with query as root."""
    verbose = data['verbose']
    query = data.get('query', 'Unknown Query')

    # 清空 ID 映射
    TreeNode._level_name_to_id.clear()
    
    # Initialize tree with query as root
    tree = TreeNode(query, is_entity=True, level=0)

    # Process verbose[-1] to get connecting entities
    last_path_data = verbose[-1]
    last_path = last_path_data['path']
    r_match = last_path_data.get('r_match', {})
    e_match = last_path_data.get('e_match', {})
    for triple_str in last_path:
        entity1, relation, entity2 = parse_triple(triple_str)
        entity_node = tree.add_child(entity1, is_entity=True)
        relation_node = entity_node.add_child(
            relation, 
            is_entity=False, 
            r_relevance=r_match.get(relation, {}).get('relevance', None)
        )
        relation_node.add_child(
            entity2, 
            is_entity=True, 
            e_relevance=e_match.get(entity2, None)
        )

    # Process remaining paths in reverse order (excluding verbose[-1])
    for path_data in verbose[:-1][::-1]:
        r_match = path_data.get('r_match', {})
        e_match = path_data.get('e_match', {})
        for triple_str in path_data['path']:
            entity1, relation, entity2 = parse_triple(triple_str)
            add_path_to_tree(tree, entity1, relation, entity2, r_match, e_match)

    return tree

def visualize_tree(root: TreeNode, verbose: List[Dict], output_file: str = 'tree', utility_score=0, ):
    """Visualize the tree using Graphviz with relevance on edges, rationality on entity nodes, and underline for root."""
    dot = Digraph(comment='Tree Visualization', format='png')
    dot.attr(rankdir='TB')
    if utility_score != 0:
        dot.attr(label=f"Overall Uility: {utility_score}")
    def get_rationality(entity_name: str) -> str:
        """Retrieve rationality score for an entity from verbose data."""
        for path_data in verbose:
            e_match = path_data.get('e_match', {})
            if entity_name in e_match and 'rationality' in path_data:
                return path_data['rationality']
        return ""

    def add_nodes_edges(node: TreeNode, parent_id: str = None, is_root: bool = False):
        node_id = node.id
        if is_root:
            # 为根节点设置下划线样式
            dot.node(node_id, 
                     label=f'{node.name}',
                     shape='underline')
        else:
            # 为实体节点添加 rationality 分数
            label = node.name
            if node.is_entity:
                rationality = get_rationality(node.name)
                if rationality:
                    label += f"\n{rationality}"
                    # label = f'{node.name}<BR/><FONT COLOR="#FF0000">{rationality}</FONT>'
            dot.node(node_id, 
                     label=label, 
                     shape='box' if not node.is_entity else 'ellipse')
        
        if parent_id:
            # 根据边类型添加 relevance 标签
            label = ""
            if node.is_entity and node.e_relevance:
                label = node.e_relevance  # 关系到实体的边
            elif not node.is_entity and node.r_relevance:
                label = node.r_relevance  # 实体到关系的边
            dot.edge(parent_id, node_id, label=label)
        
        for child in node.children.values():
            add_nodes_edges(child, node_id, is_root=False)

    # 为根节点传入 is_root=True
    add_nodes_edges(root, is_root=True)
    dot.render(output_file, view=False, cleanup=True)
    print(f"Tree visualization saved as {output_file}.png")
    return dot