# +
from datasets import load_dataset
import pandas as pd
import numpy as np
import json
import time

import os
# os.environ["OPENAI_API_KEY"]=""

import sys
from tqdm import tqdm
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()
# -

with open("data/financial_saeha_text.json", 'r') as file:
    samples = json.load(file)

firstsamples=[]
with open("data/financial_saeha_text_qa.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line.strip())
        firstsamples.append(data)


class CodeResult(BaseModel):
    question: str
    options: list[str]
    answer: str
    reason: str

text="""
user가 입력하는 내용을 사용해서 아래 샘플처럼 ### question, ### options, ### answer, ### reason을 생성해줘.
샘플과 다른 내용으로 생성해줘.

샘플
### question: {}

### options: {}
### answer: {}

### reason: {}
"""

# +
# text="""
# user가 입력하는 내용을 사용해서 아래 샘플처럼 ### question, ### options, ### answer, ### reason을 생성해줘

# 샘플
# ### question: 다음 중 우리나라 주식시장 매매 제도에 대한 기술로 맞는 것은?
# ### options: 
# A. 주식을 매수하려면 반드시 물리적 주식 증서를 거래소에 제출해야 한다.
# B. 모든 주식 거래는 비밀리에 처리되며, 거래 내용은 공개되지 않는다.
# C. 주식시장은 오전 9시부터 오후 3시 30분까지 거래가 이루어진다.
# D. 한국 주식시장에서는 모든 주식 거래가 경매 방식으로만 이루어진다.

# ### answer: C.
# ### reason: 한국거래소(KRX)의 규정에 따르면, 한국의 주식시장은 주식 매매가 오전 9시부터 오후 3시 30분까지 이루어집니다. 이는 일반적인 거래 시간으로, 이 시간 동안 투자자들은 주식을 사고팔 수 있습니다.
# """
# -

with open("data/financial_saeha_text_qa.jsonl", "a") as file:
    pt=0
    ct=0
    for i in tqdm(range(len(samples))):
        context=samples[i]["formatted_text"]
        
        question=firstsamples[i]["question"].replace("질문: ", "")
        options=""
        for tmp in firstsamples[i]["options"]:
            options+=tmp+"\n"
        answer=firstsamples[i]["answer"]
        reason=firstsamples[i]["reason"]
        prompt=text.format(question, options, answer, reason)
        
        completion = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context},
                        ],
                        response_format=CodeResult,
                        )
        try:
            question=completion.choices[0].message.parsed.question
            options=completion.choices[0].message.parsed.options
            answer=completion.choices[0].message.parsed.answer
            reason=completion.choices[0].message.parsed.reason
            pt+=completion.usage.prompt_tokens
            ct+=completion.usage.completion_tokens

            data = {
                            "iteration": i,
                            "question": question,
                            "options": options,
                            "answer": answer,
                            "reason": reason
                        }
            print("pt: ", pt, "/ ct: ", ct)
            file.write(json.dumps(data) + "\n")
            file.flush()
        except:
            pass
