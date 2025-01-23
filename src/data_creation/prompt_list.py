
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

few_shot_relation_prompt = FewShotPromptTemplate(
    examples=[{
        "query": "Kronoberg County is part of the country using what currency?",
        "topic": "Kronoberg County",
        "evidence": "location.administrative_division.country;location.country.currency_used;location.country.currency_formerly_used;location.country.first_level_divisions;location.administrative_division.first_level_division_of",
        "preceding_sentences": "",
        "output": """1. {{location.country.currency_used (Score: [Fully Relevant])}}: This relationship is highly relevant as it directly provides information about the currency currently used in the country, which is essential for answering the query regarding the currency.
2. {{location.administrative_division.country (Score: [Partially Relevant])}}: This relationship is relevant as it point to which country Kronoberg County is in, thus can be helpful in finding the currency used.
3. {{location.country.currency_formerly_used (Score: [Unrelevant])}}: This relation is unrelevant to the question as the query asks about current currency, currency formerly used do not help."""
    },
        {
        "query": "Kronoberg County is part of the country using what currency?",
            "topic": "Sweden",
            "evidence": "location.country.currency_used;location.country.official_language;location.country.currency_formerly_used;location.country.form_of_government;finance.currency.countries_used",
            "preceding_sentences": "(Kronoberg County,location.administrative_division.first_level_division_of,Sweden)",
            "output": """1. {{location.country.currency_used (Score: [Fully Relevant])}}: This relationship is highly relevant as it directly provides information about the currency currently used in the country, which is essential for answering the query regarding the currency.
2. {{finance.currency.countries_used (Score: [Fully Relevant])}}: This relationship is highly relevant as it provides the currency the Sweden country used.
3. {{location.country.currency_formerly_used (Score: [Unrelevant])}}: This relation is unrelevant to the question as the query asks about current currency, currency formerly used do not help."""
    }],
    example_prompt=PromptTemplate.from_template("""###
Query: {query}
Topic Entity: {topic}
Preceding sentences: {preceding_sentences}
Evidence: {evidence}
Output: {output}
"""),
    prefix="""You will receive a query, topic entity, evidence and optional preceding sentences containing history information. The evidence contains graph relationships possibly useful to answering the query. Your task is to filters out 3 valid information from the evidence that contribute to answering the query and provide a relevance score for each output, output your explanations for the score.
The score of relevance range from [Fully Relevant], [Partially Relevant] to [Unrelevant]:
- If the relationship directly contains information directly about the query or can answer the query with information in preceding sentences, return [Fully Relevant].
- If the relationship do not directly answer the query, but includes information possibly point to the answer, return [Partially Relevant].
- If the relationship contains irrelevant information about the query, return [Unrelevant].""",
    suffix="""###
Query: {query}
Topic Entity: {topic}
Preceding sentences: {preceding_sentences}
Evidence: {evidence}
Output: """,
    input_variables=["query", "evidence", "preceding_sentences", "topic"],
)

few_shot_entity_prompt = FewShotPromptTemplate(
    examples=[{
        "query": "Name the president of the country whose main spoken language was Brahui in 1980",
        "evidence": "(Unknown-Entity, people.place_lived.location, De Smet)",
        "preceding_sentences": "(The Long Winter, book.written_work.author, Laura Ingalls Wilder);(Laura Ingalls Wilder, people.person.places_lived, Unknown-Entity);",
        "output": """1. {{De Smet (Score: [Fully Relevant])}}. The De Smet is fully relevant to the query as the triplet directly provide the answer to the query."""
    },
        {
        "query": "what is the name of justin bieber brother",
            "evidence": "(Justin Bieber, people.person.parents, Jeremy Bieber);(Justin Bieber, people.person.parents, Pattie Mallette)",
            "preceding_sentences": "",
            "output": """1. {{Jeremy Bieber (Score: [Partially Relevant])}}. Jeremy Bieber is rated as partially relevant because he is a potential candidate as Justin Bieber's brother based on the evidence provided.
2. {{Pattie Mallette (Score: [Unrelevant])}}. Pattie Mallette is rated as unrelevant as she is not the brother of Justin Bieber."""
    },
        {
        "query": "who did draco malloy end up marrying",
            "evidence": "(m.09k254w, fictional_universe.marriage_of_fictional_characters.spouses, Astoria Greengrass);",
            "preceding_sentences": "(Draco Malfoy', 'fictional_universe.fictional_character.married_to', 'm.09k254w)",
            "output": """1. {{Astoria Greengrass (Score: [Fully Relevant])}}. Astoria Greengrass is fully relevant as she is identified as the character who Draco Malfoy ended up marrying based on the evidence provided."""
    }],
    example_prompt=PromptTemplate.from_template("""###Example
Query: {query}
Preceding information: {preceding_sentences}
Evidence: {evidence}
Output: {output}
"""),
    prefix="""You will receive a query ,evidence and optional preceding historical information for the task. The evidence and preceding information include associated retrieved knowledge graph triplets presented as (head entity, relation, tail entity). 
Your task is to assign a relevance score to the query for each tail entity in the evidence. Additionally, you are required to provide explanations for the scores assigned.
The relevance scores should fall into one of the following categories: [Fully Relevant], [Partially Relevant], or [Unrelevant]. Below are some examples:""",
    suffix="""###Task
Query: {query}
Preceding information: {preceding_sentences}
Evidence: {evidence}
Output: """,
    input_variables=["query", "evidence", "preceding_sentences", ],
)
# test_all_relation
few_shot_all_relation_prompt = FewShotPromptTemplate(
    examples=[
        {
            "query": "Kronoberg County is part of the country using what currency?",
            "topic": "Kronoberg County",
            "evidence": "location.administrative_division.country;location.country.currency_used;location.country.currency_formerly_used;location.administrative_division.first_level_division_of",
            "preceding_sentences": "",
            "output": """1. {{location.administrative_division.country (Score: [Partially Relevant])}}: This relationship is relevant as it point to which country Kronoberg County is in, thus can be helpful in finding the currency used.
2. {{location.country.currency_used (Score: [Fully Relevant])}}: This relationship is highly relevant as it directly provides information about the currency currently used in the country, which is essential for answering the query regarding the currency.
3. {{location.country.currency_formerly_used (Score: [Unrelevant])}}: This relation is unrelevant to the question as the query asks about current currency, currency formerly used do not help.
4. {{location.administrative_division.first_level_division_of (Score: [Partially Relevant])}}: This relationship is relevant as it point to which administrative division Kronoberg County is in, which may be helpful in finding the currency used."""
        },
        {
            "query": "Kronoberg County is part of the country using what currency?",
            "topic": "Sweden",
            "evidence": "location.country.currency_used;location.country.official_language;location.country.currency_formerly_used;finance.currency.countries_used",
            "preceding_sentences": "(Kronoberg County,location.administrative_division.first_level_division_of,Sweden)",
            "output": """1. {{location.country.currency_used (Score: [Fully Relevant])}}: This relationship is highly relevant as it directly provides information about the currency currently used in the country, which is essential for answering the query regarding the currency.
2. {{location.country.official_language (Score: [Unrelevant])}}: This relation is unrelevant to the question as the query asks about currency, official language do not help.
3. {{location.country.currency_formerly_used (Score: [Unrelevant])}}: This relation is unrelevant to the question as the query asks about current currency, currency formerly used do not help.
4. {{finance.currency.countries_used (Score: [Fully Relevant])}}: This relationship is highly relevant as it provides the currency the Sweden country used."""
        }],
    example_prompt=PromptTemplate.from_template("""###
Query: {query}
Topic Entity: {topic}
Preceding sentences: {preceding_sentences}
Evidence: {evidence}
Output: {output}
"""),
    prefix="""You will receive a query, topic entity, evidence and optional preceding sentences containing history information. The evidence contains graph relationships possibly useful to answering the query. Your task is evaluate each relationship's contribution to answering the query and provide a relevance score for each relation, output your explanations for the score.
The score of relevance range from [Fully Relevant], [Partially Relevant] to [Unrelevant]:
- If the relationship directly contains information directly about the query or can answer the query with information in preceding sentences, return [Fully Relevant].
- If the relationship do not directly answer the query, but includes information possibly point to the answer, return [Partially Relevant].
- If the relationship contains irrelevant information about the query, return [Unrelevant].""",
    suffix="""###
Query: {query}
Topic Entity: {topic}
Preceding sentences: {preceding_sentences}
Evidence: {evidence}
Output: """,
    input_variables=["query", "evidence", "preceding_sentences", "topic"],
)


few_shot_path_prompt = FewShotPromptTemplate(
    examples=[{
        "query": "What college did Bill Clinton attend that is in Eastern Time Zone?",
        "output": "Georgetown University",
        "preceding_sentences": "(Eastern Time Zone, common.topic.image, Timezoneswest)",
        "score": "[Unreasonable]",
        "explain": 'The reasoning path provided does not directly lead to the answer of "Georgetown University". The path jumps from the topic entity of "Eastern Time Zone" to a random connection of "common.topic.image" with "Timezoneswest", which does not relate to the college attended by Bill Clinton. Therefore, the path is not reasonable in explaining how Georgetown University is the college that Bill Clinton attended in the Eastern Time Zone.'}, {
        "query": "Kronoberg County is part of the country using what currency?",
        "output": "Swedish krona",
        "preceding_sentences": "(Kronoberg County, location.administrative_division.first_level_division_of, Sweden)",
        "score": "[Fully Reasonable]",
        "explain": "The reasoning path correctly identifies that Kronoberg County is located in Sweden. Since Sweden uses the Swedish krona as its currency, it is a logical conclusion that Kronoberg County also uses the Swedish krona. The path from Kronoberg County to Swedish krona is logically sound"},
        {"query": "What college did Bill Clinton attend that is in Eastern Time Zone?",
         "output": "Georgetown University",
         "preceding_sentences": "(Bill Clinton, people.person.education, m.0125bptr)",
         "score": "[Partially Reasonable]",
         "explain": "The reasoning path correctly identifies Bill Clinton's education, but it does not specifically mention that Georgetown University is in the Eastern Time Zone. Further clarification or exploration of the time zone of Georgetown University would be helpful."},],
    example_prompt=PromptTemplate.from_template("""###
Query: {query}
Output: {output}
Reasoning path: {preceding_sentences}
Score: {score}
Explanation: {explain}"""),
    prefix="""You will receive a query, output and a reasoning path. The reasoning path contains the current reasoning process starting from the topic entitiy to the answer. Your task is to rate rationality score for the path and output your explanations for the score.
The score of relevance range from [Fully Reasonable], [Partially Reasonable] to [Unreasonable].""",
    suffix="""###
Query: {query}
Output: {output}
Reasoning path: {preceding_sentences}
Score: """,
    input_variables=["query", "output", "preceding_sentences"]
)
