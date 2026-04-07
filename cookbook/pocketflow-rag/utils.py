import os
import numpy as np
from openai import OpenAI

def call_llm(prompt):    
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
    # r = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    client = OpenAI(
        api_key="sk-2d473aff192243ea9f44bf468d16258b", base_url="https://api.deepseek.com"
    )
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False,
    )
    return r.choices[0].message.content

def get_embedding(text):
    # client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"))
    
    # response = client.embeddings.create(
    #     model="text-embedding-ada-002",
    #     input=text
    # )

    # # Extract the embedding vector from the response
    # embedding = response.data[0].embedding
    
    # # Convert to numpy array for consistency with other embedding functions
    # return np.array(embedding, dtype=np.float32)

    # from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    # from llama_index.core import Settings, VectorStoreIndex, Document

    # # pip install llama-index
    # # pip install llama-index-embeddings-huggingface

    # import os
    # os.environ["OPENAI_API_KEY"] = "not_needed_for_hf"

    # embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    # Settings.embed_model = embed_model # we specify the embedding model to be used
    # docs = [
    #     Document(text="Hello world! This is a test document."),
    #     Document(text="Another example document for embeddings.")
    # ]
    # # https://llamaindexxx.readthedocs.io/en/latest/module_guides/indexing/vector_store_guide.html
    # index = VectorStoreIndex.from_documents(docs)
    # response = index.as_query_engine().query(text)  # It does not work!!!!!

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    embs = model.encode([text], normalize_embeddings=True)
    response = embs[0]
    return response

def fixed_size_chunk(text, chunk_size=2000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i : i + chunk_size])
    return chunks

if __name__ == "__main__":
    print("=== Testing call_llm ===")
    prompt = "In a few words, what is the meaning of life?"
    print(f"Prompt: {prompt}")
    response = call_llm(prompt)
    print(f"Response: {response}")

    print("=== Testing embedding function ===")
    
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "Python is a popular programming language for data science."
    
    oai_emb1 = get_embedding(text1)
    oai_emb2 = get_embedding(text2)
    print(f"OpenAI Embedding 1 shape: {oai_emb1.shape}")
    oai_similarity = np.dot(oai_emb1, oai_emb2)
    print(f"OpenAI similarity between texts: {oai_similarity:.4f}")