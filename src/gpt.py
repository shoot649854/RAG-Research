import tiktoken
import os
import glob
import json
import dotenv

from openai import OpenAI
dotenv.load_dotenv()
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class GPT:
    def __init__(self, model='gpt-3.5-turbo') -> None:
        self.encoding = tiktoken.encoding_for_model(model) # gpt-3.5, gpt-3.5-turbo
        self.API_KEY = os.environ.get('OPEN_AI_API_KEY')
        pass

    def translate_code(self, code_segment: str):
        prompt_message_system = "Translate the following code into English"
        prompt_message_user = f"{code_segment}"
        client = OpenAI(api_key=self.API_KEY)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": prompt_message_system},
                {"role": "user", "content": prompt_message_user}
            ]
        )
        return completion.choices[0].message.content

    def generate_text(function):
        prompt_message_system = "Translate the following code into English"
        prompt_message_user = f"{function}"
        client = OpenAI(api_key='-')
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": prompt_message_system},
                {"role": "user", "content": prompt_message_user}
            ]
        )
        return completion.choices[0].message.content

    def tokenize_translated_text(self, code):
        english = GPT.generate_text(code)
        tokens = self.encoding.encode(english)
        return tokens