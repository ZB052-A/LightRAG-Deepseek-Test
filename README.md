## 项目简介

本项目包含一个基于 LightRAG 的RAG测试系统（接入Deepseek大模型API）。

## 文件结构

```
/
├── agents
│   └── lightrag.py
├── lightrag
│   ├── api
│   │   ├── azure_openai_lightrag_server.py
│   │   ├── lollms_lightrag_server.py
│   │   ├── ollama_lightrag_server.py
│   │   ├── openai_lightrag_server.py
│   │   └── requirements.txt
│   ├── base.py
│   ├── kg
│   │   ├── age_impl.py
│   │   ├── chroma_impl.py
│   │   ├── gremlin_impl.py
│   │   ├── milvus_impl.py
│   │   ├── mongo_impl.py
│   │   ├── neo4j_impl.py
│   │   ├── oracle_impl.py
│   │   ├── postgres_impl.py
│   │   ├── postgres_impl_test.py
│   │   ├── tidb_impl.py
│   │   └── __init__.py
│   ├── lightrag.py
│   ├── llm.py
│   ├── operate.py
│   ├── prompt.py
│   ├── storage.py
│   ├── utils.py
│   └── __init__.py
├── README.md
├── requirements.txt
├── server.py
└── test
    ├── book.txt
    └── book_zh.txt
```

### 文件说明

- `agents/lightrag.py`: 包含 LightRAG + Deepseek API 的测试代码。
- `book.txt book_zh.txt`: 测试文本（Gutenberg eBook和 朱元璋 第一章）。
- `server.py`: main文件，用于初始化和启动测试server。

## 使用方法

1. 确保您的环境中已安装所有必要的依赖项。
2. 在`lightrag.py`设置 `FILE_STORAGE_DIR` 和 `WORKING_DIR` 的路径。
3. 设置环境变量 `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `SILICONCLOUD_EMBEDDING_MODEL`, `SILICONCLOUD_API_KEY`。
4. 运行 `server.py` 脚本以启动代理服务。
5. 通过 API 接口与代理进行交互，或者http://your-ip:20000/查看FastAPI文档。

## 安装依赖

在开始之前，请确保安装了所有必要的依赖项。您可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 运行服务

启动服务的命令如下：

```bash
python server.py
```

欢迎对本项目进行贡献。有问题提 Issue。
