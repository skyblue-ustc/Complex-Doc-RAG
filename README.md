# Complex-Doc-RAG 📄🤖

一个专为**复杂文档（Complex Documents）**设计的工业级 RAG 系统。
针对**科研论文、财报**等包含**密集表格**、**多栏排版**的文档进行了深度优化。

## 🌟 核心特性 (Key Features)

* **📊 表格序列化 (Table Serialization)**: 并非简单提取文本，而是将 PDF 表格重构为 Markdown/HTML 结构，让 LLM 能精准理解行列数据。
* **🧩 深度布局分析**: 解决双栏/三栏排版导致的文本乱序问题。
* **🏷️ 自动化元数据索引 (Metadata Indexing)**: 内置 `ingest_papers.py` 脚本，自动计算文件哈希并生成语义索引，支持文件名与内容的解耦检索。
* **🧠 DeepSeek 推理集成**: 针对 DeepSeek-V3/R1 模型优化了 API 调用逻辑，绕过 SDK 限制，支持长上下文推理。
* **🔍 双路检索 (Hybrid Retrieval)**: 结合 Vector Search (语义) + BM25 (关键词) + Parent Document Retrieval (父文档召回) 策略。

## 📂 目录结构

```text
Complex-Doc-RAG/
├── main.py                # 命令行入口
├── ingest_papers.py       # 自动化数据摄入脚本 (生成 metadata)
├── src/                   # 核心源码 (Pipeline, Retrieval, Parser)
├── configs/               # 配置文件
├── data/                  # 数据存放区 (Raw PDFs & Indices)
└── outputs/               # 运行结果 (Answers & Logs)