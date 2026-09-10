import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config import constants
from config.settings import settings
from utils.logging import logger

class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use pre-downloaded docling models if available, so no Hugging Face
        # download is attempted at runtime.
        artifacts_path = Path(
            settings.DOCLING_ARTIFACTS_PATH
            or Path(__file__).resolve().parent.parent / "docling_models"
        )
        self.docling_artifacts_path = str(artifacts_path) if artifacts_path.is_dir() else None
        if self.docling_artifacts_path:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            logger.info(f"Using local docling models from {self.docling_artifacts_path}")
        else:
            logger.warning(
                "No local docling models found; docling will try to download "
                "from Hugging Face on first use."
            )
        
    def validate_files(self, files: List) -> None:
        """Validate the total size of the uploaded files."""
        total_size = sum(os.path.getsize(f.name) for f in files)
        if total_size > constants.MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {constants.MAX_TOTAL_SIZE//1024//1024}MB limit")

    def process(self, files: List) -> List:
        """Process files with caching for subsequent queries"""
        self.validate_files(files)
        all_chunks = []
        seen_hashes = set()
        
        for file in files:
            try:
                # Generate content-based hash for caching
                with open(file.name, "rb") as f:
                    file_hash = self._generate_hash(f.read())
                
                cache_path = self.cache_dir / f"{file_hash}.pkl"
                
                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {file.name}")
                    chunks = self._load_from_cache(cache_path)
                else:
                    logger.info(f"Processing and caching: {file.name}")
                    chunks = self._process_file(file)
                    self._save_to_cache(chunks, cache_path)
                
                # Deduplicate chunks across files
                for chunk in chunks:
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)
                        
            except Exception as e:
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue
                
        logger.info(f"Total unique chunks: {len(all_chunks)}")
        return all_chunks

    def _process_file(self, file) -> List:
        """Original processing logic with Docling"""
        if not file.name.endswith(('.pdf', '.docx', '.txt', '.md')):
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []

        if self.docling_artifacts_path:
            pipeline_options = PdfPipelineOptions(artifacts_path=self.docling_artifacts_path)
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        else:
            converter = DocumentConverter()
        markdown = converter.convert(file.name).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)

    def _generate_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _save_to_cache(self, chunks: List, cache_path: Path):
        with open(cache_path, "wb") as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": chunks
            }, f)

    def _load_from_cache(self, cache_path: Path) -> List:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["chunks"]

    def _is_cache_valid(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
            
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)