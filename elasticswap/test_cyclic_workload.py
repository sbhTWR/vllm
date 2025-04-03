import wget
import json
from time import sleep
import time
from typing import List
from openai import OpenAI
import openai
import requests
from bs4 import BeautifulSoup
import threading

def get_text_from_url(url: str):
    html = requests.get(url).text
    return html

class SimpleMultiTurnWorkload:

    def __init__(self, port=8000, user_id='default_user_id'):
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:%s/v1" % port

        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        models = self.client.models.list()
        print(models)

        self.user_id = user_id

        self.model = models.data[0].id

        # self.contexts: List[str] = []
        self.interrupts: List[int] = []

        self.messages: List[dict] = []


    def add_context(self, context_str):

        message = []
        if not self.messages:
            message.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
        
        message.append({
            "role": "user",
            "content": context_str
        })

        self.messages.append(message)
    
    def add_interrupt(self, elapsed_time):
        self.interrupts.append(elapsed_time)
    

    def send_fin(self):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": "placeholder"
                }],
                model=self.model,
                user=json.dumps({"id": self.user_id, "type": "fin"}),
                timeout=1
            )
        except openai.APITimeoutError as e:
            print("3. ------ END --------")

    def execute_workload(self):

        while self.messages:
            message = self.messages.pop(0)
            chat_completion = self.client.chat.completions.create(
                messages=message,
                model=self.model,
                temperature=0,
                user=json.dumps({"id": self.user_id, "type": "append"})
            )

            print(chat_completion)

            if self.interrupts:
                elapse_t = self.interrupts.pop(0)
                print('sleeping for %d seconds' % elapse_t)
                time.sleep(elapse_t) 
        
        self.send_fin()

def main():
    url = "https://en.wikipedia.org/wiki/List_of_chiropterans"

    threads = []

    text = get_text_from_url(url)
    print(len(text))

    workload = SimpleMultiTurnWorkload(port=8000, user_id="test1")
    workload.add_context("Please summarize the following text: " + text[:60000])
    workload.add_interrupt(5)
    workload.add_context("What species are mentioned in the article?")
    workload.add_interrupt(5)
    workload.add_context("What are chiropetrans in short?")
    th = threading.Thread(target=workload.execute_workload, args=())
    threads.append(th)

    workload1 = SimpleMultiTurnWorkload(port=8000, user_id="test2")
    workload1.add_context("Please summarize the following text: " + text[60000:120000])
    workload1.add_interrupt(5)
    workload1.add_context("What species are mentioned in the article?")
    workload1.add_interrupt(5)
    workload1.add_context("What are chiropetrans in short?")
    th = threading.Thread(target=workload1.execute_workload, args=())
    threads.append(th)

    workload2 = SimpleMultiTurnWorkload(port=8000, user_id="test3")
    workload2.add_context("Please summarize the following text: " + text[120000:180000])
    workload2.add_interrupt(5)
    workload2.add_context("What species are mentioned in the article?")
    workload2.add_interrupt(5)
    workload2.add_context("What are chiropetrans in short?")
    th = threading.Thread(target=workload2.execute_workload, args=())
    threads.append(th)

    workload3 = SimpleMultiTurnWorkload(port=8000, user_id="test4")
    workload3.add_context("Please summarize the following text: " + text[180000:240000])
    workload3.add_interrupt(5)
    workload3.add_context("What species are mentioned in the article?")
    workload3.add_interrupt(5)
    workload3.add_context("What are chiropetrans in short?")
    th = threading.Thread(target=workload3.execute_workload, args=())
    threads.append(th)

    workload4 = SimpleMultiTurnWorkload(port=8000, user_id="test5")
    workload4.add_context("Please summarize the following text: " + text[240000:300000])
    workload4.add_interrupt(5)
    workload4.add_context("What species are mentioned in the article?")
    workload4.add_interrupt(5)
    workload4.add_context("What are chiropetrans in short?")
    th = threading.Thread(target=workload4.execute_workload, args=())
    threads.append(th)

    workload5 = SimpleMultiTurnWorkload(port=8000, user_id="test6")
    workload5.add_context("Please summarize the following text: " + text[300000:360000])
    workload5.add_interrupt(5)
    workload5.add_context("What species are mentioned in the article?")
    workload5.add_interrupt(5)
    workload5.add_context("What are chiropetrans in short?")
    th = threading.Thread(target=workload5.execute_workload, args=())
    threads.append(th)

    workload6 = SimpleMultiTurnWorkload(port=8000, user_id="test7")
    workload6.add_context("Please summarize the following text: " + text[360000:420000])
    workload6.add_interrupt(5)
    workload6.add_context("What species are mentioned in the article?")
    workload6.add_interrupt(5)
    workload6.add_context("What are chiropetrans in short?")
    th = threading.Thread(target=workload6.execute_workload, args=())
    threads.append(th)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    

if __name__ == "__main__":
    main()
            
        

