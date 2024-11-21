import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

load_dotenv()

logging.basicConfig(
    filename="compare.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    filemode="w",
    force=True,
)


llm = ChatOpenAI(model="gpt-4o-mini")

json_schema = {
    "title": "Comparison",
    "description": "Comparing two answers to a question",
    "type": "object",
    "properties": {
        "Conclusion": {"type": "string", "enum": ["Answer1更好", "Answer2更好", "Tie"]},
        "Reason": {"type": "string"},
    },
    "required": ["Conclusion", "Reason"],
}

structured_llm = llm.with_structured_output(json_schema)

csv_path = "qa_pairs.csv"
df = pd.read_csv(csv_path)

comparisons = []
count = 0
comparison_json_path = "comparison.json"

for row in df.itertuples():
    question = row.Question
    answer1 = row.Answer
    answer2 = row.Response_gpt
    id = row.id
    try:
        comparison_prompt = f"""
        针对以下问题有两个答案：
        
        Question: {question}
        
        Answer 1：
        {answer1}

        Answer 2：
        {answer2}

        请以生态环境领域专家的角度，根据以下评估准则，从专业性、清晰度、可行性三个方面综合考虑、评估哪个答案更好，并简略描述评估理由。
        好答案的评估准则：
        1. 专业性高: 答案包含具体细节，内容准确详细，具备较强的专业性，而非一般的科普知识。当涉及公式、数字与计算时，Answer1一定是更为准确的。
        2. 简洁清晰：答案的语言简洁明了，论述有条理、有逻辑性，没有冗余信息。
        3. 可行性高：当答案涉及提出方案时，提出的解决方案是否具有可行性和实际操作性。

        注意：在评估时，专业性是最重要的，其次是清晰度，最后是可行性。

        """
        comparison_response = structured_llm.invoke(comparison_prompt)

        comparisons.append(
            {
                "id": id,
                "Conclusion": comparison_response["Conclusion"],
                "Reason": comparison_response["Reason"],
            }
        )
    except Exception as e:
        logging.error(f"Error comparing question '{id}': {e}")
        comparisons.append({"id": id, "Conclusion": "Error", "Reason": str(e)})

    count += 1
    if count >= 20:
        comparison_df = pd.DataFrame(comparisons)
        with open(comparison_json_path, "a", encoding="utf-8") as f:
            comparison_df.to_json(f, orient="records", lines=True, force_ascii=False)
        logging.info(f"Comparison finished for {id}")
        count = 0
        comparisons = []

if count % 20 != 0:
    comparison_df = pd.DataFrame(comparisons)
    with open(comparison_json_path, "a", encoding="utf-8") as f:
        comparison_df.to_json(f, orient="records", lines=True, force_ascii=False)
    logging.info(f"Comparison finished for all questions")
