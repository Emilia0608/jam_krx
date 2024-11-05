from datasets import load_dataset
import pandas as pd
import numpy as np
import json
import time
import random

import os
# os.environ["OPENAI_API_KEY"]=""

import sys
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

with open('data/new_stock.json', 'r', encoding='utf-8') as file:
    stock_data = json.load(file)

class CodeResult(BaseModel):
    question: str
    options: list[str]
    answer: str

text="""
user가 입력하는 data를 사용해서 아래 샘플과 같이 ### question 과 ### options, ### answer을 생성해줘.
question은 코드를 활용하여 금융 데이터를 분석하고 복잡한 금융 계산을 수행하는 능력을 평가하기 위한 문제들로 구성해줘.
question은 컬럼을 명시하는것이 아닌 자연어로 작성해줘.
options에 컬럼명 형식을 다르게 적은 오답도 만들어줘.

샘플
### df.head()
|    | Symbol    | Series   | Date        |   Prev Close |   Open Price |   High Price |   Low Price |   Last Price |   Close Price |   Average Price |   Total Traded Quantity |    Turnover |   No. of Trades |   Deliverable Qty |   % Dly Qt to Traded Qty |
|---:|:----------|:---------|:------------|-------------:|-------------:|-------------:|------------:|-------------:|--------------:|----------------:|------------------------:|------------:|----------------:|------------------:|-------------------------:|
|  0 | GODREJIND | EQ       | 15-May-2017 |       564.6  |       581    |       584    |      568.5  |       578.9  |        578.55 |          578.09 |                  797171 | 4.60836e+08 |           21649 |            360927 |                    45.28 |
|  1 | GODREJIND | EQ       | 16-May-2017 |       578.55 |       581.45 |       589    |      572.25 |       583.8  |        584.8  |          583.6  |                  500223 | 2.9193e+08  |           17204 |            210364 |                    42.05 |
|  2 | GODREJIND | EQ       | 17-May-2017 |       584.8  |       583    |       594    |      576.85 |       584.9  |        588.6  |          588.74 |                  504155 | 2.96815e+08 |            8567 |            261667 |                    51.9  |
|  3 | GODREJIND | EQ       | 18-May-2017 |       588.6  |       582    |       588.85 |      571.2  |       572.25 |        574.6  |          580.9  |                  223583 | 1.29879e+08 |            7144 |             99785 |                    44.63 |
|  4 | GODREJIND | EQ       | 19-May-2017 |       574.6  |       581    |       585.8  |      567.55 |       579.85 |        578    |          577.31 |                  245436 | 1.41692e+08 |            4969 |             68041 |                    27.72 |

### question:
"종가" 열의 평균 값을 계산합니다.

### options:
python
df['Close Price'].mean()

python
df['Close_Price'].mean()


python
df['Total Traded Quantity'].median()

python
sum(df['Close Price']) / len(df['Close_Price'])


### answer:
python
df['Close Price'].mean()
"""

def make_column_name(num, dataframe):
    if num==0:
        dataframe=dataframe.replace("Open", "OpenPrice").replace("High", "HighPrice").replace("Low", "LowPrice").replace("Close", "ClosePrice")
    elif num==1:
        dataframe=dataframe.replace("Open", "open_price").replace("High", "high_price").replace("Low", "low_price").replace("Close", "close_price")
    elif num==2:
        dataframe=dataframe.replace("Open", "open-price").replace("High", "high-price").replace("Low", "low-price").replace("Close", "close-price")
    elif num==3:
        dataframe=dataframe.replace("Open", "openprice").replace("High", "highprice").replace("Low", "lowprice").replace("Close", "closeprice")
    elif num==4:
        dataframe=dataframe.replace("Open", "OPEN_PRICE").replace("High", "HIGH_PRICE").replace("Low", "LOW_PRICE").replace("Close", "CLOSE_PRICE")
    return dataframe

with open("data/code_generate_v2.jsonl", "a") as file:
    pt=0
    ct=0
    
    for i in tqdm(range(len(stock_data))):
        num=random.choice([0,1,2,3,4])
        dataframe=stock_data[i]["prompt"].split("### 분석:")[-1]
        dataframe=make_column_name(num, dataframe)
        
        completion = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                        {"role": "system", "content": text},
                        {"role": "user", "content": dataframe},
                        ],
                        response_format=CodeResult,
                        )
        try:
            question=completion.choices[0].message.parsed.question
            options=completion.choices[0].message.parsed.options
            answer=completion.choices[0].message.parsed.answer
            pt+=completion.usage.prompt_tokens
            ct+=completion.usage.completion_tokens

            data = {
                        "iteration": i,
                        "dataframe": dataframe,
                        "question": question,
                        "options": options,
                        "answer": answer
                    }

            print("pt: ", pt, "/ ct: ", ct)
            file.write(json.dumps(data) + "\n")
            file.flush()
        except:
            pass

