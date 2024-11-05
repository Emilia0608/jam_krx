import pandas as pd
import sys
import json

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import pandas as pd
import time

from bs4 import BeautifulSoup
from tqdm import tqdm

import re

# +
# Chrome 옵션 설정
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# driver = webdriver.Chrome(executable_path="/usr/local/bin/chromedriver", options=chrome_options)
service = Service(executable_path="/usr/local/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=chrome_options)


# -

def normalize_whitespace(text):
    # \s+ 는 하나 이상의 공백 문자 (띄어쓰기, 탭, 줄바꿈 등)를 의미
    # 이를 ' ' (한 개의 띄어쓰기)로 대체
    return re.sub(r'\s+', ' ', text).strip()


samples=[]
with open("data/krx_doc_crawling.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line.strip())  # 각 줄을 JSON 형식으로 파싱
        samples.append(data)
df = pd.DataFrame(samples)
df['report_date'] = pd.to_datetime(df['report_date'])  # report_date를 datetime 형식으로 변환
latest_docs = df.loc[df.groupby('company_name')['report_date'].idxmax(), ['company_name', 'company_doc_num']]
latest_docs=latest_docs[latest_docs["company_name"]!=""]
latest_docs.reset_index(inplace=True, drop=True)

latest_docs=latest_docs[2500:]

with open("data/krx_doc_report_crawling_1104_v2.jsonl", "a") as file:
    for i in tqdm(range(len(latest_docs))):
        company_name=latest_docs.iloc[i]["company_name"]
        company_doc_num=latest_docs.iloc[i]["company_doc_num"]

        url=f"https://kind.krx.co.kr/common/disclsviewer.do?method=search&acptno={company_doc_num}"
        driver.get(url)
        time.sleep(1)

        driver.switch_to.frame("docViewFrm")
        time.sleep(0.5)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        start_printing = False
        saup=""
        try:
            for tag in soup.find_all(['h2', 'h3', 'p']):
                context=tag.get_text(strip=True)

                if tag.name == 'h2' and "사업의 내용" in context:    
                    start_printing = True
                    continue
                if start_printing:
                    if tag.name=="p":
                        if len(context.split(" "))<5:
                                pass
                        else:
                            saup+=context+"\n "
                    elif tag.name=='h3' and "개요" not in context and "제품" not in context:
                        break

            data = {"company_name": company_name,
                    "company_report_context": saup}
            file.write(json.dumps(data) + "\n")
            file.flush()
        except:
            pass
