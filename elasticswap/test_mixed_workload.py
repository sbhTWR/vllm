import json
from time import sleep
import time
from typing import List
from openai import OpenAI
import openai

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
            
        

