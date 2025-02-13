
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

# 新的reflection token1
# few_shot_is_useful = FewShotPromptTemplate(
#         examples=[{
#             "query": "What is the name of Justin Bieber's brother?",
#             "output": "Justin Bieber's younger brother is named Jaxon Bieber. He was born in 2009 and is the son of Justin’s father, Jeremy Bieber, from his relationship with Erin Wagner.",
#             "explanation": "The output is complete, highly detailed and informative, fully satisfying the need of the query. So the rating should be [Utility:5].",
#             "rating": "[Utility:5]"},
#                   {
#             "query": "Who was the US President in 2023?",
#             "output": "The President was Biden.",
#             "explanation": "The output answers the query and provides useful information. However, it is too brief and can be slightly improved, such as providing the full name of the President: Joe Biden. So the rating should be [Utility:4].",
#             "rating": "[Utility:4]"},
#                   {
#             "query": "Recommend 5 famous scenic spots in Beijing and provide detailed descriptions for each.",
#             "output": "The National Museum of China, the Palace Museum, and Renmin University of China.",
#             "explanation": "The output responds to the query and provides useful information, but the number of scenic spots recommended is less than 5 and no description is provided. So the rating should be [Utility:3].",
#             "rating": "[Utility:3]"},
#                   {
#             "query": "Who was the Governor of Michigan in July 2017?",
#             "output": "The governor of Michigan is Gretchen Whitmer, who has been serving as governor since 2019.",
#             "explanation": "The output is still talking about the governor of Michigan and providing some correct information. But it does not answer who the governor was in July 2017, not providing useful information for the query. So the rating should be [Utility:2].",
#             "rating": "[Utility:2]"},
#                   {
#             "query": "What are the advantages of Python in data analysis?",
#             "output": "Cloud computing is a type of distributed computing. It is a system with extremely strong computing ability formed through a computer network.",
#             "explanation": "The output is completely irrelevant to the query. So the rating should be [Utility:1].",
#             "rating": "[Utility:1]"}],
#         example_prompt=PromptTemplate.from_template("""
# Query: {query}\n
# Output: {output}
# Explanation: {explanation}
# Rating: {rating}
# """),
#         prefix="""Given a query and an output, rate whether the response appears to be a helpful and informative answer to the query, from 1 (lowest) - 5 (highest). We call this Utility Score.
# Use the following entailment scale to generate a rating:
# [Utility:5]: The response provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs. 
# [Utility:4]: The response mostly fulfills the need in the query and provides helpful answers, while there can be some minor improvements such as discussing more detailed information, having better structure of the response, improving coherence or having less repetition.
# [Utility:3]: The response is acceptable, but some major additions or improvements are needed to satisfy users’ needs. 
# [Utility:2]: The response still addresses the main request, but it is not complete or not relevant to the query.
# [Utility:1]: The response is completely irrelevant to the query or does not give an answer in the end.
# Additionally, make sure to **always** provide the rating in square brackets. (e.g., [Utility:5] instead of Utility:5)
# When you are given the same query-output pairs, ensure that you should generate the **SAME** ratings.""",
#         suffix=
# """
# Query: {query}\n
# Output: {output}\n""",
#         input_variables=["query", "output"],
# )
# 新的reflection token1
few_shot_is_useful = FewShotPromptTemplate(
        examples=[{
            "query": "What is the name of Justin Bieber's brother?",
            "output": "Jaxon Bieber",
            "rating": """{{"individual_scores": {{"Jaxon Bieber": "[Fully Useful]"}}, "overall_scores": "[Utility:5]", "explanation": "The answer Jaxon Bieber provides useful information for the query, so its individual score should be [Fully Useful]. Generally, the query is fully answered, so the overall score should be [Utility:5]."}}"""},
                  {
            "query": "In which year did Donald Trump win the US presidential election?",
            "output": "2024",
            "rating": """{{"individual_scores": {{"2024": "[Fully Useful]"}}, "overall_scores": "[Utility:4]", "explanation": "2024 is a correct answer to the query, so its individual score should be [Fully Useful]. Generally, the query is answered correctly, but there should be two correct answers: 2016 and 2024, while the output only provides one. So the overall score should be [Utility:4]."}}"""},
                  {
            "query": "I want to visit Renmin University of China, but I don't know where it is. Tell me its address.",
            "output": "Beijing, Haidian District",
            "rating":  """{{"individual_scores": {{"Beijing": "[Partially Useful]","Haidian District": "[Partially Useful]"}}, "overall_scores": "[Utility:3]", "explanation": "Both Beijing and Haidian District are correct but too vague to help locate the university campus. So the individual scores should be [Partially Useful], and the overall score should be [Utility:3]."}}"""},
                  {
            "query": "Who was the Governor of Michigan in July 2017?",
            "output": "Gretchen Whitmer",
            "rating": """{{"individual_scores": {{"Gretchen Whitmer": "[Not Useful]"}}, "overall_scores": "[Utility:2]", "explanation": "The output is still talking about the governor of Michigan and providing some correct information. But it does not answer who the governor was in July 2017 (Gretchen Whitmer has been serving as the governor from 2019), not providing useful information for the query. So the individual score of Gretchen Whitmer should be [Not Useful], and the overall score should be [Utility:2]."}}"""},
                  {
            "query": "What are the advantages of Python in data analysis?",
            "output": "Guido van Rossum, Portugal",
            "rating": """{{"individual_scores": {{"Guido van Rossum": "[Not Useful]", "Portugal": "[Not Useful]"}}, "overall_scores": "[Utility:1]", "explanation": "The output is completely irrelevant to the query. So the individual scores should be [Not Useful], and the overall score should be [Utility:1]."}}"""}],
        example_prompt=PromptTemplate.from_template("""
Query: {query}
Answer: {output}
Rating: {rating}
"""),
        prefix="""You will be given a query and the answers, where the answers may consist of one or more individual answers, separated by commas(,). 
Your task is to generate a **rating** to evaluate whether the answer is a useful response to the query. The rating provide an individual score for each individual answer, and then give an overall score for the entire output.
Use the following entailment scale to give individual score(s):
[Fully Useful]: The individual answer provides correct and useful information for the query.
[Partially Useful]: The individual answer may provide some correct information that is helpful in addressing the query, but it has some flaws, such as lacking specificity.
[Not Useful]: The individual answer provides incorrect information or is irrelevant to the query.
Use the following entailment scale to give an overall score:
[Utility:5]: Generally, the output provides a complete, highly detailed, and informative response to the query, fully satisfying the information needs.
[Utility:4]: Generally, the output mostly fulfills the need in the query and provides helpful answers, while there can be some minor improvements, such as discussing more detailed information or providing additional correct answers beyond the current output.
[Utility:3]: Generally, the output is correct and acceptable, but there are obvious problems, such as being too vague or not specific enough, limiting its helpfulness in addressing the query. 
[Utility:2]: Generally, the output still discusses the topic of the query, but it is incorrect or does not actually meet the requirements of the query.
[Utility:1]: Generally, the output is completely irrelevant to the query or does not give an answer in the end.
""",
        suffix=
"""
Query: {query}
Answers: {output}
Rating:
 """,
        input_variables=["query", "output"],
)


# The name of Justin Bieber's brother is Jaxon Bieber. This is based on the reasoning path that connects Justin Bieber to Jaxon Bieber through the relationship of sibling. The other paths that connect Justin Bieber to Jazmyn Bieber or back to Justin Bieber himself are incorrect in this context.
few_shot_path_prompt_meta = FewShotPromptTemplate(
        examples=[{
            "query": "what were the release years of the movies that share screenwriters with the movie Ivanhoe",
            "output": "1949;1968;1969;1979;1941;1970;1937;1953;1952;2010;1939;1958",
            "preceding_sentences": "(Ivanhoe, starred_actors, James Mason)",
            "score": "[Partially Reasonable]",
            "explain": 'The preceding reasoning path pointed out the actor, and the final answer may be obtained through other movies of this actor, which is partially reasonable.'},{
            "query": "the movies that share directors with the movie Broadway Serenade are written by who",
            "output": "John Meehan;John P. Marquand;Jane Hall;Helen Jerome;Zelda Sears;",
            "preceding_sentences": "(Broadway Serenade, directed_by, Robert Z. Leonard)",
            "score": "[Fully Reasonable]",
            "explain": "The preceding reasoning path is fully reasonable as it correctly identifies the director of the movie Broadway Serenade."},
            {"query": "the movies that share directors with the movie Broadway Serenade are written by who",
            "output": "John Meehan;John P. Marquand;Jane Hall;Helen Jerome;Zelda Sears;",
            "preceding_sentences": "(Broadway Serenade, directed_by, Robert Z. Leonard),(Robert Z. Leonard, directed_by, Maytime)",
            "score": "[Fully Reasonable]",
            "explain": "The preceding reasoning path is fully reasonable as it correctly identifies the movie with the same director of the movie Broadway Serenade."},
            {"query": "who are the directors of movies whose writers also wrote Hot Fuzz",
            "output": "Steven Spielberg;Greg Mottola;Edgar Wright",
            "preceding_sentences": "(Hot Fuzz, has_tags, cornetto trilogy)",
            "score": "[Unreasonable]",
            "explain": "The question asks about the director or screenwriter. It is not reasonable to search for tags."},],
        example_prompt=PromptTemplate.from_template("""###
Query: {query}
Output: {output}
Reasoning path: {preceding_sentences}
Score: {score}
Explanation: {explain}"""),
        prefix=
        """You will receive a query, output and a reasoning path. The reasoning path contains the current reasoning process starting from the topic entitiy to the answer. Your task is to rate rationality score for the path and output your explanations for the score.
The score of relevance range from [Fully Reasonable], [Partially Reasonable] to [Unreasonable].""",
        suffix=
        """######
Query: {query}
Output: {output}
Reasoning path: {preceding_sentences}
Score: """,
        input_variables= ["query", "output", "preceding_sentences"]
)


