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


def process_pickle_file(pickle_filename):
    with open(pickle_filename, "rb") as f:
        text_list = pickle.load(f)

    data = merge_pickle_list(text_list)
    data = fix_utf8(data)
    # data = trim_data(data)

    qa_list = []

    for i in range(0, len(data), 40):
        chunk = data[i : i + 40]
        if len(chunk) < 40:
            break

        # 将chunk转换为字符串
        chunk_text = " ".join([text[0] for text in chunk])

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
                    "PickleFile": os.path.splitext(os.path.basename(pickle_filename))[0],
                }
            )
        except Exception as e:
            logging.error(f"Error in processing chunk: {e}")


    return qa_list


folder_path = "pickles"
output_folder = "output"
pickle_names = os.listdir(folder_path)

for item in pickle_names:
    item_id = os.path.splitext(item)[0]
    if not os.path.isfile(os.path.join(output_folder, f"{item_id}.csv")):
        pickle_filepath = os.path.join(folder_path, item)
        qa_list = process_pickle_file(pickle_filepath)

        df = pd.DataFrame(qa_list)
        output_filepath = os.path.join(output_folder, f"{item_id}.csv")
        df.to_csv(output_filepath, index=False, encoding="utf-8")
        logging.info(f"Processed {item_id}.")

