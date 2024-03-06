import tiktoken
import os
import glob
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class Tokenize:
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        pass

    def tokenizer(self, file_content: str):
        result = self.encoding.encode(file_content, disallowed_special=())
        return result
    
    def save_json(self, file_path: str, result: dict):
        try:
            with open(file_path, 'w') as json_file:
                json.dump(result, json_file)
            logger.info(f"JSON file saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Error saving JSON file: {e}")
