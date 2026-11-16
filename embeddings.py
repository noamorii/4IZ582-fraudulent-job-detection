import os
from dotenv import load_dotenv
from litellm import embedding
import numpy as np

MODEL="openai/qwen3-embedding:4b" # the only embeddings model available

load_dotenv() # reads .env and puts the values into the environment

BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ.get("API_KEY")

print("base url :", BASE_URL)
print("model    :", MODEL)
print("key      :", "set" if API_KEY else "MISSING")


def get_embeddings(texts, cache_path, batch_size=64, use_cache=True):
    if use_cache and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached) == len(texts):
            return cached

    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = embedding(
            model=MODEL,
            input=batch,
            api_base=BASE_URL,
            api_key=API_KEY,
        )
        vectors.extend(item["embedding"] for item in response.data)

    arr = np.array(vectors)
    np.save(cache_path, arr)
    return arr

