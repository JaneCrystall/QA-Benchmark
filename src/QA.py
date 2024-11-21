import csv
import logging
import os
import pickle
from io import StringIO
import pandas as pd
import random

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import boto3
from opensearchpy import AWSV4SignerAuth
import tiktoken
from bs4 import BeautifulSoup

load_dotenv()

logging.basicConfig(
    filename="QA.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    filemode="w",
    force=True,
)

host = os.environ.get("AWS_OPENSEARCH_URL")
region = "us-east-1"

service = "aoss"
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

s3_client = boto3.client("s3")


llm = ChatOpenAI(model="gpt-4o-mini")
bucket_name = "tiangong"
prefix = "processed_docs/edu_textbooks_pickle/"
suffix = ".pkl"

json_schema = {
    "title": "Questions_and_Answers",
    "description": "Generating questions and answers for evaluating LLM performance",
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Question": {
                        "type": "string",
                        "description": "Question generated for evaluating LLM performance",
                    },
                    "Answer": {
                        "type": "string",
                        "description": "Answer to the question",
                    },
                    "Level": {
                        "type": "string",
                        "description": "Level of the question from perspective of environmenal domain",
                        "enum": ["Easy", "Medium", "Hard"],
                    },
                    "Type": {
                        "type": "string",
                        "description": "Type of the question",
                        "enum": ["专业基础知识", "计算", "逻辑推理"],
                    },
                    "Domain": {
                        "type": "string",
                        "description": "Domain of the question",
                        "enum": [
                            "环境工程学",
                            "大气环境学",
                            "水环境学",
                            "环境生态学",
                            "土壤环境学",
                            "生物环境学",
                            "环境控制学",
                            "环境监测学",
                            "环境化学",
                            "环境地学",
                            "环境毒理学",
                            "环境法学",
                            "环境经济学",
                            "环境管理学",
                            "环境伦理学",
                        ],
                    },
                },
                "required": ["Question", "Answer", "Level", "Type", "Domain"],
            },
            "minItems": 1,
            "maxItems": 1,
        }
    },
    "required": ["qa_pairs"],
}
structured_llm = llm.with_structured_output(json_schema)


def load_pickle_from_s3(bucket_name, s3_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    body = response["Body"].read()
    data = pickle.loads(body)
    return data


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string))


def fix_utf8(original_list):
    cleaned_list = []
    for original_str in original_list:
        cleaned_str = original_str[0].replace("\ufffd", " ")
        cleaned_list.append([cleaned_str, original_str[1]])
    return cleaned_list

def split_dataframe_table(html_table, chunk_size=8100):
    dfs = pd.read_html(StringIO(html_table))
    if not dfs:
        return []

    df = dfs[0]
    tables = []
    sub_df = pd.DataFrame()
    token_count = 0

    for _, row in df.iterrows():
        row_html = row.to_frame().T.to_html(index=False, border=0, classes=None)
        row_token_count = num_tokens_from_string(row_html)

        if token_count + row_token_count > chunk_size and not sub_df.empty:
            sub_html = sub_df.to_html(index=False, border=0, classes=None)
            tables.append(sub_html)
            sub_df = pd.DataFrame()
            token_count = 0

        sub_df = pd.concat([sub_df, row.to_frame().T])
        token_count += row_token_count

    if not sub_df.empty:
        sub_html = sub_df.to_html(index=False, border=0, classes=None)
        tables.append(sub_html)

    return tables

def merge_pickle_list(data):
    temp = ""
    result = []
    for d in data:
        if num_tokens_from_string(d[0]) > 8100:
            soup = BeautifulSoup(d[0], "html.parser")
            tables = soup.find_all("table")
            for table in tables:
                table_content = str(table)
                if num_tokens_from_string(table_content) < 8100:
                    if table_content:  # check if table_content is not empty
                        result.append([table_content, d[1]])
                else:
                    try:
                        sub_tables = split_dataframe_table(table_content)
                        for sub_table in sub_tables:
                            if sub_table:
                                soup = BeautifulSoup(sub_table, "html.parser")
                                result.append([str(soup), d[1]])
                    except Exception as e:
                        logging.error(f"Error splitting dataframe table: {e}")
        elif num_tokens_from_string(d[0]) < 15:
            temp += d[0] + " "
        else:
            result.append([(temp + d[0]), d[1]])
            temp = ""
    if temp:
        result.append([temp, d[1]])

    return result

def trim_data(data):
    """
    去掉列表前后5%的元素
    """
    data = [d[0] for d in data]
    n = len(data)
    remove_count = int(n * 0.05)
    return data[remove_count:n-remove_count]

def get_random_chunks(data, k):
    """
    从列表中随机取出5个长度为k的不重叠子列表
    """
    data = trim_data(data)
    n = len(data)
    if k * 5 > n:
        return []

    random_chunks = []
    available_indices = list(range(n - k + 1))

    for _ in range(5):
        start_index = random.choice(available_indices)
        random_chunks.append(data[start_index:start_index + k])
        # 移除已选定子列表范围内的所有索引
        available_indices = [i for i in available_indices if i < start_index or i >= start_index + k]

    return random_chunks

def get_chunks_from_pickle(pickle_filename,chunk_k=40):
    data = load_pickle_from_s3(bucket_name, prefix + pickle_filename)
    data = merge_pickle_list(data)
    data = fix_utf8(data)
    chunks = get_random_chunks(data, k=chunk_k)
    return chunks

def generating_qa_pairs(chunks, pickle_filename):
    qa_list = []
    for chunk in chunks:
        # 将chunk转换为字符串
        chunk_text = " ".join(chunk)
        # 执行structured_llm，生成问题和答案
        try:
            response = structured_llm.invoke(
                f"""基于以下内容，编写1个问答对，用来测试大语言模型及其相关应用（比如rag）在生态环境专业领域中的能力和表现: 

                {chunk_text}

                要尽可能生成在生态环境专业领域中具有挑战性的问题，并提供相应的详细答案（需要详细的阐述）。
                在确定问题难度时要考虑问题的复杂性、推理深度和背景知识需求。"""
            )

            # print(response)
            qa_pair = response.get("qa_pairs", [])[0]

            # 添加到qa_list
            qa_list.append(
                {
                    "Question": qa_pair["Question"],
                    "Answer": qa_pair["Answer"],
                    "Level": qa_pair["Level"],
                    "Type": qa_pair["Type"],
                    "Domain": qa_pair["Domain"],
                    "PickleFile": pickle_filename,
                }
            )
        except Exception as e:
            logging.error(f"Error in processing chunk: {e}")

    return qa_list


output_folder = "qa_output"
with open("filtered_books.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    pickle_names = [row[0] for row in reader]

for item in pickle_names:
    if not os.path.isfile(os.path.join(output_folder, f"{item}.json")):
        pickle_filename = item + ".pkl"
        try:
            random_chunks = get_chunks_from_pickle(pickle_filename, chunk_k=60)
        except Exception as e:
            logging.error(f"Error in processing {item}: {e}")
            continue
        if len(random_chunks) == 0:
            logging.error(f"Error in processing {item}")
            continue
        else:
            qa_list = generating_qa_pairs(random_chunks, item)
            df = pd.DataFrame(qa_list)
            output_filepath = os.path.join(output_folder, f"{item}.json")
            df.to_json(output_filepath, orient="records", lines=True, force_ascii=False)
            logging.info(f"Processed {item}.")
