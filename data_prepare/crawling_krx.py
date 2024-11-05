# +
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


# -

def get_report_num(company_doc_num):
    return re.search(r"'(\d+)'", company_doc_num).group(1)


# Chrome 옵션 설정
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# driver = webdriver.Chrome(executable_path="/usr/local/bin/chromedriver", options=chrome_options)
service = Service(executable_path="/usr/local/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=chrome_options)


# +
url="https://kind.krx.co.kr/disclosure/details.do?method=searchDetailsMain"
driver.get(url)
time.sleep(1)

from_date_input = driver.find_element(By.ID, 'fromDate')
from_date_input.clear()
from_date_input.send_keys("2021-01-01")
time.sleep(0.5)

to_date_input = driver.find_element(By.ID, 'toDate')
to_date_input.clear()
to_date_input.send_keys("2023-12-31")
time.sleep(0.5)

date_shudown=driver.find_element(By.CLASS_NAME, "ui-state-default.ui-state-active")
date_shudown.click()
time.sleep(0.5)

checkbox = driver.find_element(By.ID, 'lastReport')
    
# 체크가 안 되어 있으면 클릭해서 체크
if not checkbox.is_selected():
    checkbox.click()
time.sleep(0.5)

jungki = driver.find_element(By.ID, 'dsclsType05')
jungki.click()
time.sleep(0.5)

sahup_checkbox1 = driver.find_element(By.ID, 'dsclsLayer05_3')
sahup_checkbox2 = driver.find_element(By.ID, 'dsclsLayer05_4')
sahup_checkbox3 = driver.find_element(By.ID, 'dsclsLayer05_5')
    
# 체크가 안 되어 있으면 클릭해서 체크
if not sahup_checkbox1.is_selected():
    sahup_checkbox1.click()
time.sleep(0.5)
if not sahup_checkbox2.is_selected():
    sahup_checkbox2.click()
time.sleep(0.5)
if not sahup_checkbox3.is_selected():
    sahup_checkbox3.click()
time.sleep(0.5)

search_button = driver.find_element(By.CLASS_NAME, 'search-btn')
search_button.click()
time.sleep(0.5)
# -

with open("data/krx_doc_crawling.jsonl", "a") as file:
    for i in tqdm(range(2083)):
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        companies=soup.select("#main-contents tbody tr")
        print(len(companies))
        for company in companies:
            try:
                company_name=company.select("td")[4].text.strip()
                company_doc_num=company.select("td a")[1]["onclick"]
                company_doc_num=get_report_num(company_doc_num)
                report_date=company.select("td.txc")[1].text.strip().split(" ")[0]
                search_doc_num=company.select("td.txc")[0].text.strip()

                data = {"search_doc_num": search_doc_num, 
                        "company_name": company_name,
                        "company_doc_num": company_doc_num,
                        "report_date":report_date}
                
                file.write(json.dumps(data) + "\n")
                file.flush()
            except:
                print("error")
        time.sleep(1)
        next_page_button = driver.find_element(By.CLASS_NAME, 'next')
        next_page_button.click()
        time.sleep(5)


