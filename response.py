import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

load_dotenv()

logging.basicConfig(
    filename="response2.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    filemode="w",
    force=True,
)


llm = ChatOpenAI(model="gpt-4o-mini")

input_csv_path = "/root/projects/QA-Benchmark/QA-Benchmark/files/q_and_a.csv"
df = pd.read_csv(input_csv_path)
questions = df["Question"].iloc[2000:5088]

responses = []

for question in questions:
    try:
        response = llm.invoke(question)
        response_content = response.content
        responses.append({"Question": question, "Response": response_content})
    except Exception as e:
        logging.error(f"Error processing question '{question}': {e}")
        responses.append({"Question": question, "Response": "Error generating response"})


output_csv_path = "/root/projects/QA-Benchmark/QA-Benchmark/files/responses2.csv"
response_df = pd.DataFrame(responses)
response_df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"Responses have been written to {output_csv_path}")