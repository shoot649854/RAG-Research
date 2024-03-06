import os
import ast
from openai import OpenAI
import numpy as np
import dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine

dotenv.load_dotenv()
client = OpenAI(api_key=os.environ.get('OPEN_AI_API_KEY'))

class CodeRetriever:
    def __init__(self) -> None:
        pass

    def parse_functions(self, code_str: str) -> list:
        tree = ast.parse(code_str)

        function_list = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                function_list.append(ast.unparse(node))
        return function_list

    def translate_to_english(self, function_list: list) -> list:
        """
        Translates the given list of functions into English summaries using OpenAI's GPT-4 model.

        Parameters:
            function_list (list): A list of strings representing code functions to be translated.

        Returns:
            list: A list of strings containing English summaries of the input functions.
        """
        translated_functions = []
        
        prompt_system = "Summarize the functions into English within 20 tokens."
        for func in function_list:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": func},
                ],
                temperature=0,
                max_tokens=25
            )
            translation = response.choices[0].message.content
        
            translated_functions.append(translation)
        return translated_functions


    # Normalize function embeddings
    def normalize_l2(self, np_arr: np.ndarray) -> np.ndarray:
        np_arr = np.array(np_arr)
        if np_arr.ndim == 1:
            normalized = np.linalg.norm(np_arr)
            if normalized == 0:
                return np_arr
            return np_arr / normalized
        else:
            normalized = np.linalg.norm(np_arr, 2, anp_arris=1, keepdims=True)
            return np.where(normalized == 0, np_arr, np_arr / normalized)


    # Generate embeddings for text
    def generate_embeddings(self, text: str, model_type: str) -> np.ndarray:
        """
        Generates embeddings for the given text using the specified embedding model.

        Parameters:
            text (str): The input text for which embeddings are to be generated.
            model_type (str): The type of the embedding model to be used.
                Available options: 'openai', 'sentence-transformers'.

        Returns:
            np.ndarray: An array containing the embeddings generated for the input text.
        """
        if model_type == 'openai':
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small",
            )
            embedding = np.array(response.data[0].embedding)
            print(embedding)
        else:
            model = SentenceTransformer(model_type)
            embedding = model.encode(text)[0]

        return embedding

    # Rank functions based on query
    def rank_functions(self, query: str, translated_functions: list, model_type: str) -> dict:
        """
        Ranks the translated functions based on similarity to the given query using embeddings.

        Parameters:
            query (str): The query string to compare against the translated functions.
            translated_functions (list): A list of strings containing translated function summaries.
            model_type (str): The type of the embedding model to be used.
                Available options: 'openai', 'sentence-transformers'.

        Returns:
            dict: A dictionary containing ranks of translated functions based on similarity to the query.
                Keys represent the index of the function in the input list, and values represent similarity scores.
        """
        query_embedding = self.generate_embeddings(query, model_type)
        function_embeddings = [self.generate_embeddings(func, model_type) for func in translated_functions]
        ranks = {}
        
        for i, func_embedding in enumerate(function_embeddings):
            similarity = cosine(query_embedding, func_embedding)
            ranks[i] = similarity
    
        sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item: item[1])}
        
        return sorted_ranks

    # Process file to retrieve and rank functions
    def process_file(self, file_path: str, query: str, model_type: str) -> None:
        with open(file_path) as file:
            code_content = file.read()

        functions = self.parse_functions(code_content)
        english_functions = self.translate_to_english(functions)
        ranked_functions = self.rank_functions(query, english_functions, model_type)

        print("Ranked functions:")
        for i, (index, similarity) in enumerate(ranked_functions.items(), 1):
            function_name = functions[index]
            print(f"Rank {i}: Similarity: {similarity}, Function: {function_name}")
