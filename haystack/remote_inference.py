import requests
from typing import List, Union, Dict, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def get_embeddings(host: str, api_key: str, text_list: List[str], batch_size=64) -> Union[np.ndarray, None]:
    total_embeddings = []

    for i in range(0, len(text_list), batch_size):
        batch_texts = text_list[i : i + batch_size]

        # Construct the API URL
        api_url = f"{host}/ml/embedding/"

        # Prepare headers
        headers = {"API-Key": api_key, "Content-Type": "application/json"}

        # Prepare payload
        payload = {"texts": batch_texts}

        # Make the API request
        response = requests.post(api_url, json=payload, headers=headers)

        # Raise an exception for unsuccessful status codes
        response.raise_for_status()

        # Accumulate results
        total_embeddings.extend(response.json())

    return np.array(total_embeddings)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def rerank_documents(
    host: str, api_key: str, query: str, documents: Dict[str, Any], top_k: int
) -> Dict[str, Dict[str, Any]]:
    url = f"{host}/ml/rerank/"
    headers = {"API-Key": api_key, "Content-Type": "application/json"}
    payload = {"query": query, "documents": documents, "top_k": top_k}

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

    reranked_documents = response.json()
    return reranked_documents
