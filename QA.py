import logging
import os
import pickle

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from tools.data import merge_pickle_list, fix_utf8, trim_data

load_dotenv()

logging.basicConfig(
    filename="QA.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    filemode="w",
    force=True,
)


llm = ChatOpenAI(model="gpt-4o-mini")

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
                        "description": "Question generated for evaluating LLM performance"
                    },
                    "Answer": {
                        "type": "string",
                        "description": "Answer to the question"
                    },
                    "Level": {
                        "type": "string",
                        "description": "Level of the question from perspective of environmenal domain",
                        "enum": ["Easy", "Medium", "Hard"]
                    },
                    "Type": {
                        "type": "string",
                        "description": "Type of the question",
                        "enum": ["专业基础知识", "计算", "逻辑推理"]
                    },
                    "Domain": {
                        "type": "string",
                        "description": "Domain of the question",
                        "enum": ["环境工程学", "大气环境学", "水环境学"]
                    }
                },
                "required": ["Question", "Answer", "Category"]
            },
            "minItems": 1,
            "maxItems": 1
        }
    },
    "required": ["qa_pairs"]
}
structured_llm = llm.with_structured_output(json_schema)

# 遍历文件夹中的pickle文件
def process_pickle_file(pickle_filename):
    with open(pickle_filename, "rb") as f:
        text_list = pickle.load(f)

    data = merge_pickle_list(text_list)
    data = fix_utf8(data)
    # data = trim_data(data)

    qa_list = []

    for i in range(0, len(data), 20):
        chunk = data[i : i + 20]
        if len(chunk) < 10:
            break

        # 将chunk转换为字符串
        chunk_text = " ".join([text[0] for text in chunk])

        # 执行structured_llm，生成问题和答案
        response = structured_llm.invoke(
            f"""基于以下内容，编写1个问答对，用来测试大语言模型及其相关应用（比如rag）在生态环境专业领域中的能力和表现: 

            {chunk_text}

            问题要尽量难一点，并提供相应的详细答案（需要详细的阐述）。"""
        )

        print(response)
        qa_pair = response.get('qa_pairs', [])

        # 添加到qa_list
        qa_list.append({
            'Question': qa_pair['Question'],
            'Answer': qa_pair['Answer'],
            'Level': qa_pair['Level'],
            'Type': qa_pair['Type'],
            'Domain': qa_pair['Domain'],
            'PickleFile': pickle_filename
        })

    return qa_list

folder_path = "pickles"
pickle_names = os.listdir(folder_path)

for pickle in pickle_names:
    pickle_filepath = os.path.join(folder_path, pickle)
    qa_list = process_pickle_file(pickle_filepath)
    df = pd.DataFrame(qa_list)
    if not os.path.isfile("output.csv"):
        df.to_csv("output.csv", index=False, encoding='utf-8')
    else:
        df.to_csv("output.csv", mode='a', header=False, index=False, encoding='utf-8')

    

