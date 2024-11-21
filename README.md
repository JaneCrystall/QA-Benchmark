# QA-Benchmark

## Env Preparing
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
python.exe -m pip install --upgrade pip

pip install --upgrade pip

pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt --upgrade
```
```bash
nohup .venv/bin/python3.12 QA.py > QA_log.txt 2>&1 &

find output/ -type f | wc -l
find pickles/ -type f | wc -l


nohup .venv/bin/python3.12 src/edu_textbooks/QA.py > QA_log.txt 2>&1 &
find qa_output/ -type f | wc -l

nohup .venv/bin/python3.12 src/edu_textbooks/comparison.py > compare_log.txt 2>&1 &
```
