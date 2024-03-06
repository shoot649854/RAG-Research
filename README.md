# RAG-Research

[![OpenAI](https://img.shields.io/badge/OpenAI-API-blue)](https://openai.com/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21.5-blue)](https://numpy.org/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-2.2.0-blue)](https://www.sbert.net/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-blue)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7.3-blue)](https://www.scipy.org/)

## Overview

RAG-Research is a Python package for conducting research in the field of natural language processing (NLP) using techniques like Retrieval-Augmented Generation (RAG). It provides functionality to retrieve, translate, and rank functions or code snippets based on a given query.

## Installation

You can install RAG-Research via pip:

```bash
pip install rag-research
```

## Usage
To use RAG-Research in your project, first import the necessary modules:

```python
from rag_research import CodeRetriever
```

Then, create an instance of the CodeRetriever class and use its methods to parse, translate, and rank functions:

```python
retriever = CodeRetriever()
functions = retriever.parse_functions(code_str)
translated_functions = retriever.translate_to_english(functions)
ranked_functions = retriever.rank_functions(query, translated_functions, model_type)
```

Ensure that you have the required API keys set up for OpenAI's API. You may need to set environment variables accordingly.

For more detailed usage examples and API documentation, please refer to the documentation.

## License
This project is licensed under the MIT License - see the LICENSE file for details.