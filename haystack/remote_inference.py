import requests
from typing import List, Union, Dict, Any
from tenacity import retry, wait_random_exponential, stop_after_attempt


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def get_embeddings(host: str, api_key: str, text_list: List[str]) -> Union[List[List[float]], None]:
    # Construct the API URL
    api_url = f"{host}/ml/embedding/"

    # Prepare headers
    headers = {"API-Key": api_key, "Content-Type": "application/json"}

    # Prepare payload
    payload = {"texts": text_list}

    # Make the API request
    response = requests.post(api_url, json=payload, headers=headers)

    # Raise an exception for unsuccessful status codes
    response.raise_for_status()

    # Check if request was successful
    return response.json()


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
