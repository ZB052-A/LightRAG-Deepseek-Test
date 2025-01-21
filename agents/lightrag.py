import asyncio
import inspect
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, siliconcloud_embedding
from lightrag.lightrag import always_get_an_event_loop
from lightrag.utils import EmbeddingFunc
import numpy as np
import os
import json
import textract
from typing import AsyncGenerator

# 文件存储目录
FILE_STORAGE_DIR = "./file_storage"
if not os.path.exists(FILE_STORAGE_DIR):
    os.mkdir(FILE_STORAGE_DIR)
WORKING_DIR = "./working"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

class LightRAGAgent:
    def __init__(self):
        self.llm_api_key = os.getenv(DEEPSEEK_API_KEY) or ""
        self.llm_model = os.getenv(DEEPSEEK_MODEL) or "deepseek-reasoner"
        self.llm_base_url = "https://api.deepseek.com/v1"
        self.embedding_model = os.getenv(SILICONCLOUD_EMBEDDING_MODEL) or "BAAI/bge-m3"
        self.embedding_api_key = os.getenv(SILICONCLOUD_API_KEY) or ""
        self.agent = None
    
    async def llm_model_func(
        self, prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            self.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            **kwargs,
        )

    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        return await siliconcloud_embedding(
            texts,
            model=self.embedding_model,
            api_key=self.embedding_api_key,
            max_token_size=8192,
        )
    
    async def get_embedding_dim(self):
        test_text = ["This is a test sentence."]
        embedding = await self.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        return embedding_dim

    # function test
    async def test_funcs(self):
        results = []
        try:
            result = await self.llm_model_func("你好。你的版本是？")
            assert len(result) > 0
            # print("llm_model_func: ", result)
            results.append("LLM OK.")
            result = await self.embedding_func(["你好。你是谁？"])
            assert len(result) > 0
            # print("embedding_func: ", result)
            results.append("Embedding OK.")
            return results
        except Exception as e:
            print(f"lightrag test funcs error occurred: {e}")
            results.append(f"lightrag test funcs error occurred: {e}")
            return results

    async def init_rag(self):
        try:
            embedding_dimension = await self.get_embedding_dim()
            print(f"Detected embedding dimension: {embedding_dimension}")
            self.agent = LightRAG(
                working_dir=WORKING_DIR,
                llm_model_func=self.llm_model_func,
                llm_model_max_token_size=32768,
                llm_model_max_async=64,
                addon_params={"language": "Simplified Chinese"},
                # text chunking
                chunk_token_size=1024,
                chunk_overlap_token_size=100,
                embedding_func=EmbeddingFunc(
                    embedding_dim=embedding_dimension,
                    max_token_size=8192,
                    func=self.embedding_func,
                ),
            )
        except Exception as e:
            print(f"Init lightrag error occurred: {e}")

    def get_doc_id(self):
        try:
            with open(f"{WORKING_DIR}/kv_store_doc_status.json", 'r', encoding='utf-8') as file:
                data = json.load(file)
            return list(data.keys())
        except Exception as e:
            print(f"lightrag get doc id error occurred: {e}")
            return None

    async def insert_file(self, file_paths):
        contents = []
        for file_path in file_paths:
            text_content = textract.process(file_path).decode("utf-8")
            contents.append(text_content)
        try:
            await self.agent.ainsert(contents)
            return True
        except Exception as e:
            print(f"lightrag insert error occurred: {e}")
            return None

    async def get_summarize(self, input_text) -> str:
        response = await self.agent.aquery(
                input_text, param=QueryParam(mode="hybrid", top_k=60)
            )
        # print(f"lightrag response: {response}")
        return response

    async def delete_file(self, file_name):
        try:
            await self.agent.adelete_by_doc_id(file_name)
            return True
        except Exception as e:
            print(f"lightrag delete error occurred: {e}")
            return False