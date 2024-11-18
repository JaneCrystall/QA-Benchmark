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

answer_csv_path = "/root/projects/QA-Benchmark/QA-Benchmark/files/q_and_a.csv"
response_csv_path = "/root/projects/QA-Benchmark/QA-Benchmark/files/responses.csv"
df_q_and_a = pd.read_csv(answer_csv_path)
responses_df = pd.read_csv(response_csv_path)

questions = responses_df["Question"].head(2)
answers = df_q_and_a["Answer"].head(2)
responses = responses_df["Response"].head(2)

comparisons = []


for question, answer, response in zip(questions, answers, responses):
    try:
        comparison_prompt = f"根据以下问题：\n\n{question}\n\n比较以下两个回答哪个更好，并得出结论：\n\nAnswer: {answer}\n\nResponse: {response}\n\n结论分为三个选项：Answer更好，Response更好，Answer和Response侧重点不同。"
        comparison_response = llm.invoke(comparison_prompt)
        comparison_content = comparison_response.content  # 使用属性访问content
        comparisons.append({"Question": question, "Answer": answer, "Response": response, "Comparison": comparison_content})
    except Exception as e:
        logging.error(f"Error comparing question '{question}': {e}")
        comparisons.append({"Question": question, "Answer": answer, "Response": response, "Comparison": "Error generating comparison"})

# 将比较结果写入新的CSV文件
comparison_csv_path = "/root/projects/QA-Benchmark/QA-Benchmark/files/comparisons.csv"
comparison_df = pd.DataFrame(comparisons)
comparison_df.to_csv(comparison_csv_path, index=False, encoding='utf-8')

print(f"Comparisons have been written to {comparison_csv_path}")