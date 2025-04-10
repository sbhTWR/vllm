import wget
import json
import numpy as np
from time import sleep
import time
from typing import List
from openai import OpenAI
import openai
import requests
from bs4 import BeautifulSoup
import threading
from lorem_text import lorem

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
            print("[%s] sending FIN" % self.user_id)
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
            print("[%s] ------ END --------" % self.user_id)

    def execute_workload(self):
        
        message_num = 1
        while self.messages:
            message = self.messages.pop(0)
            print("[%s] sending message_id=%d" % (self.user_id, message_num))
            chat_completion = self.client.chat.completions.create(
                messages=message,
                model=self.model,
                temperature=0,
                user=json.dumps({"id": self.user_id, "type": "append"})
            )

            print("[%s] processed message_id=%d" % (self.user_id, message_num))
            print(chat_completion)

            if self.interrupts:
                elapse_t = self.interrupts.pop(0)
                print('[%s] sleeping for %d seconds' % (self.user_id, elapse_t))
                time.sleep(elapse_t) 
            
            message_num += 1
        
        self.send_fin()

def generate_workload(request_size, interrupt_len, num_interrpts, user_id, port=8000):
    context = lorem.words(request_size)
    workload = SimpleMultiTurnWorkload(port=port, user_id=user_id)
    workload.add_context("Please summarize the following text: " + context)

    for _ in range(num_interrpts):
        workload.add_interrupt(interrupt_len)
        workload.add_context("What is mentioned in the article?")
    
    return workload

def main():

    threads = []
    request_size = 30000
    interrupt_len = 5
    num_interrupts = 5
    rate = 0.5
    t = 120
    num_events = int(rate * t)
    exp_times = np.random.exponential(scale=1/rate, size=num_events)
    print(np.cumsum(exp_times))
    arrival_times = list(exp_times)

    print(arrival_times)
    input()

    print('--- test1 ---')
    req1 = lorem.words(100)
    print(req1)
    
    print('--- test2 ---')
    req2 = lorem.words(100)
    print(req2)

    input()

    for i in range(num_events):
        workload = generate_workload(request_size, interrupt_len, num_interrupts, "test_%d" % i)
        th = threading.Thread(target=workload.execute_workload, args=())
        threads.append(th)

    for thread in threads:
        wait_time = arrival_times.pop(0)
        time.sleep(wait_time)
        thread.start()

    for thread in threads:
        thread.join()

    

if __name__ == "__main__":
    main()
            
        

