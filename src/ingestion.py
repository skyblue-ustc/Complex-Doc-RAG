import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt
# === 修改1：引入本地模型库 ===
from sentence_transformers import SentenceTransformer

class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices."""
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # Load the report
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Extract text chunks and create BM25 index
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

class VectorDBIngestor:
    def __init__(self):
        self.llm = self._set_up_llm()
        # === 修改2：初始化本地模型 ===
        print("Loading local embedding model (BAAI/bge-small-en-v1.5)...")
        # 这会自动下载模型（如果本地没有），大约 100MB
        self.local_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    def _set_up_llm(self):
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com"), # 增加兼容性读取 .env
            timeout=None,
            max_retries=2
        )
        return llm

    # === 修改3：重写获取向量的逻辑，不再调用 API ===
    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-large") -> List[List[float]]:
        """
        Modified to use local HuggingFace model instead of OpenAI API to avoid 404 errors.
        """
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        
        # 确保输入是列表
        if isinstance(text, str):
            text_chunks = [text]
        else:
            text_chunks = text

        try:
            # 使用本地模型计算向量 (normalize_embeddings=True 对应余弦相似度)
            embeddings = self.local_model.encode(text_chunks, normalize_embeddings=True)
            
            # encode 返回的是 numpy array，转回 list 保持格式兼容
            return embeddings.tolist()
            
        except Exception as e:
            print(f"Error generating embeddings locally: {e}")
            # 万一出错，返回空列表防止崩溃（虽然一般不会）
            return []

    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")