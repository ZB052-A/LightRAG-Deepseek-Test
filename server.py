from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import RedirectResponse, StreamingResponse
from typing import AsyncGenerator
from pydantic import BaseModel
import uvicorn
import os
import json
import asyncio
from openai import OpenAI

from agents.lightrag import LightRAGAgent

FILE_STORAGE_DIR = "./file_storage"
app = FastAPI()

# 根路径重定向到 /docs
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

rag = LightRAGAgent()

# 列出文件列表
@app.get("/file/list/")
async def list_files():
    await rag.init_rag()
    files = os.listdir(FILE_STORAGE_DIR)
    if len(files) == 0:
        return {"files": []}
    else:
        files_id = rag.get_doc_id()
        
        return {"files": 
            [{"id": file_id, "name": file_name} for file_id, file_name in zip(files_id, files)]
        }

async def process_file(files: list[UploadFile], text: str):
    results = []
    file_paths = []

    for file in files:
        content = await file.read()
        file_path = os.path.join(FILE_STORAGE_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        file_paths.append(file_path)
    
    level = await rag.insert_file(file_paths)
    if level == True:
        summary = await rag.get_summarize(text)
        file_names = [file.filename for file in files]

        results.append({"filenames": file_names, "summary": summary})
    else:
        results.append({"summary": "文件上传失败，请检查文件格式和大小。"})

    return results

# 添加文件并总结
@app.post("/file/add_and_summarize/")
async def add_file_and_summarize(files: list[UploadFile] = File(...), text: str = Form(...)):
    await rag.init_rag()
    results = []

    # 将文件分为多个批次，每个批次最多包含10个文件
    for i in range(0, len(files), 10):
        batch = files[i:i+10]
        tasks = [process_file(batch, text)]
        results.extend(await asyncio.gather(*tasks))
    
    return results
    
# 删除文件
@app.delete("/file/delete")
async def delete_file(filename: str):
    await rag.init_rag()
    files = os.listdir(FILE_STORAGE_DIR)
    if len(files) == 0:
        return {"error": "No files found in the storage directory."}

    files_id = rag.get_doc_id()
    for file_id, file_name in zip(files_id, files):
        if file_name == filename:
            tag = await rag.delete_file(file_id)
            break
    if tag == True:
        file_path = os.path.join(FILE_STORAGE_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"message": f"File '{filename}' deleted successfully."}
        else:
            return {"error": f"File '{filename}' does not exist."}
    else:
        return {"summary": "文件删除失败，请检查服务器实现方式。"}

# 文件模块自检
@app.get("/file/health")
async def file_health_check():
    await rag.init_rag()

    response = await rag.test_funcs()
    files = os.listdir(FILE_STORAGE_DIR)
    if len(files) == 0:
        if len(response) == 2:
            return {"status": "OK", "files": [], "response": response}
        else:
            return {"status": "LLM Error", "files": [], "response": response}

    files_id = rag.get_doc_id()

    if len(files) != len(files_id):
        return {
            "status": "File Error", 
            "files": [], 
            "response": response}

    if len(response) == 2:
        return {
            "status": "OK",
            "files": 
                [{"id": file_id, "name": file_name} for file_id, file_name in zip(files_id, files)], 
            "response": response
        }
    else:
        return {
            "status": "LLM Error",
            "files": 
                [{"id": file_id, "name": file_name} for file_id, file_name in zip(files_id, files)], 
            "response": response
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=20000)