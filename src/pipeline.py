from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd

# 引入项目内的模块
from src.pdf_parsing import PDFParser
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.tables_serialization import TableSerializer

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path = None, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "raw_pdfs", serialized: bool = False, config_suffix: str = ""):
        # === 1. 自动定位项目根目录 ===
        # 如果没有传入 root_path，则自动获取当前项目的根目录 (包含 .git 或 requirements.txt 的目录)
        if root_path is None:
            self.root_path = here()
        else:
            self.root_path = root_path
            
        # === 2. 定义标准目录结构 (Complex-Doc-RAG 标准) ===
        self.data_dir = self.root_path / "data"
        self.output_dir = self.root_path / "outputs"
        
        # 自动创建关键目录 (防止报错)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "answers").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "debug").mkdir(parents=True, exist_ok=True)

        # === 3. 数据源路径 ===
        # Metadata: data/metadata/subset.csv
        self.subset_path = self.data_dir / "metadata" / subset_name
        self.questions_file_path = self.data_dir / "metadata" / questions_file_name
        
        # Raw PDFs: data/raw_pdfs/
        self.pdf_reports_dir = self.data_dir / pdf_reports_dir_name
        
        # Indices (Vector DBs): data/indices/
        self.databases_path = self.data_dir / "indices"
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        # === 4. 输出路径 ===
        # Answers: outputs/answers/answers_xxx.json
        self.answers_file_path = self.output_dir / "answers" / f"answers{config_suffix}.json"       
        self.debug_data_path = self.output_dir / "debug"

        # Debug intermediate files
        suffix = "_ser_tab" if serialized else ""
        self.parsed_reports_dirname = "01_parsed_reports"
        self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname
        
        # 打印调试信息
        print(f"✅ Pipeline Configured at Root: {self.root_path}")
        print(f"📂 Metadata: {self.subset_path}")
        print(f"💾 Indices: {self.databases_path}")
        print(f"📤 Outputs: {self.answers_file_path}")

@dataclass
class RunConfig:
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 10
    team_email: str = "79250515615@yandex.com"
    submission_name: str = "Complex-Doc-RAG v1.0"
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "openai"
    answering_model: str = "gpt-4o-mini-2024-07-18" 
    config_suffix: str = ""

class Pipeline:
    def __init__(self, root_path: Path = None, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "raw_pdfs", run_config: RunConfig = RunConfig()):
        self.run_config = run_config
        # Allow passing None to let PipelineConfig auto-detect
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        # Update path to look in metadata folder
        json_path = self.paths.data_dir / "metadata" / "subset.json"
        csv_path = self.paths.data_dir / "metadata" / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                df.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    # ... [Keep existing parsing methods: download_docling_models, parse_pdf_reports_sequential, etc.] ...
    # 为了节省篇幅，这里的方法逻辑通常不需要改，因为它们都引用 self.paths
    # 请确保保留原来类中的这些方法，或者直接把原代码块粘贴回来，
    # 只要确保 self.paths 是从上面的 PipelineConfig 来的即可。
    
    @staticmethod
    def download_docling_models(): 
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_sequential(self):
        logging.basicConfig(level=logging.DEBUG)
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path
        pdf_parser.parse_and_export(doc_dir=self.paths.pdf_reports_dir)
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        logging.basicConfig(level=logging.DEBUG)
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path
        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def serialize_tables(self, max_workers: int = 10):
        serializer = TableSerializer()
        serializer.process_directory_parallel(
            self.paths.parsed_reports_path,
            max_workers=max_workers
        )

    def merge_reports(self):
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        _ = ptp.process_reports(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.merged_reports_path
        )
        print(f"Reports saved to {self.paths.merged_reports_path}")

    def export_reports_to_markdown(self):
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        ptp.export_to_markdown(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.reports_markdown_path
        )
        print(f"Reports saved to {self.paths.reports_markdown_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        text_splitter = TextSplitter()
        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.paths.parsed_reports_path
        text_splitter.split_all_reports(
            self.paths.merged_reports_path,
            self.paths.documents_dir,
            serialized_tables_dir
        )
        print(f"Chunked reports saved to {self.paths.documents_dir}")

    def create_vector_dbs(self):
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path
        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)
        else:
            self.parse_pdf_reports_sequential()
    
    def process_parsed_reports(self):
        print("Starting reports processing pipeline...")
        print("Step 1: Merging reports...")
        self.merge_reports()
        print("Step 2: Exporting reports to markdown...")
        self.export_reports_to_markdown()
        print("Step 3: Chunking reports...")
        self.chunk_reports()
        print("Step 4: Creating vector databases...")
        self.create_vector_dbs()
        print("Reports processing pipeline completed successfully!")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        if not base_path.exists():
            return base_path
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            team_email=self.run_config.team_email,
            submission_name=self.run_config.submission_name,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

# === Configs Definitions ===

max_nst_o3m_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    # === CRITICAL: Ensure reranking is FALSE for DeepSeek stability ===
    llm_reranking=False,
    parallel_requests=25,
    submission_name="Complex-Doc-RAG v1.0",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = deepseek-chat",
    answering_model="deepseek-chat",
    config_suffix="_max_nst_o3m"
)

# === 补回缺失的 preprocess_configs ===
preprocess_configs = {
    "ser_tab": RunConfig(use_serialized_tables=True),
    "no_ser_tab": RunConfig(use_serialized_tables=False)
}

# === 主配置字典 ===
configs = {
    "max_nst_o3m": max_nst_o3m_config, 
}

if __name__ == "__main__":
    # Auto-detect root path using here()
    pipeline = Pipeline(run_config=max_nst_o3m_config)