import os
import logging
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
from app.core.document import DocumentProcessor
from app.config2 import CONFIG

logger = logging.getLogger(__name__)

# # Custom sentence tokenizer to avoid nltk dependency
# def simple_sentence_tokenizer(text: str) -> List[str]:
#     """
#     A simple sentence tokenizer that doesn't rely on nltk
    
#     Args:
#         text: Input text to tokenize into sentences
        
#     Returns:
#         List of sentences
#     """
#     # Preserve common abbreviations to avoid splitting them
#     text = re.sub(r'(Mr\.|Mrs\.|Dr\.|Ph\.D\.|etc\.|i\.e\.|e\.g\.)', lambda m: m.group().replace('.', '&period;'), text)
    
#     # Split on sentence delimiters (.!?)
#     sentences = re.split(r'(?<=[.!?])\s+', text)
    
#     # Restore periods in abbreviations
#     sentences = [s.replace('&period;', '.') for s in sentences]
    
#     # Filter empty sentences
#     return [s.strip() for s in sentences if s.strip()]

# class KnowledgeBase:
#     def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
#         """
#         Enhanced knowledge base with configurable model and improved chunking
        
#         Args:
#             model_name: Pre-trained sentence transformer model name
#         """
#         self.index = None
#         self.chunks = []
#         self.model_name = model_name
#         self.model = None
#         self.dimension = None  # Track embedding dimension
#         self._validate_faiss()

#     def _validate_faiss(self):
#         """Ensure FAISS is properly configured"""
#         if not faiss.get_num_gpus():
#             logger.warning("FAISS running on CPU - no GPU detected")

#     def initialize_model(self, model: Optional[SentenceTransformer] = None, device: str = None):
#         """Initialize embedding model with device awareness"""
#         if self.model:
#             logger.info("Model already initialized")
#             return
            
#         try:
#             self.model = model if model else SentenceTransformer(self.model_name, device=device)
#             self.dimension = self.model.get_sentence_embedding_dimension()
#             logger.info(f"Initialized model {self.model_name} with dimension {self.dimension}")
#         except Exception as e:
#             logger.error(f"Model initialization failed: {str(e)}")
#             raise RuntimeError(f"Could not initialize model: {str(e)}")

#     def build_from_documents(self, doc_paths: List[str], chunk_size: int = 512, overlap: int = 64):
#         """Build knowledge base with parallel processing support"""
#         from concurrent.futures import ThreadPoolExecutor
#         from tqdm import tqdm
        
#         if not doc_paths:
#             raise ValueError("No document paths provided")
            
#         self.initialize_model()
        
#         all_chunks = []
#         failed_files = []
#         processed_files = 0
        
#         def process_file(path):
#             nonlocal processed_files
#             try:
#                 if not os.path.exists(path):
#                     logger.warning(f"File not found: {path}")
#                     return None
                    
#                 text = DocumentProcessor().extract_text(path)
#                 if not text.strip():
#                     logger.warning(f"Empty text from: {path}")
#                     return None
                    
#                 chunks = self._chunk_text(text, chunk_size, overlap)
#                 processed_files += 1
#                 return chunks
                
#             except Exception as e:
#                 logger.error(f"Failed {path}: {str(e)}")
#                 failed_files.append(path)
#                 return None

#         # Process files in parallel
#         with ThreadPoolExecutor(max_workers=4) as executor:
#             results = list(tqdm(executor.map(process_file, doc_paths), total=len(doc_paths)))
            
#         all_chunks = [chunk for result in results if result for chunk in result]
        
#         if not all_chunks:
#             raise ValueError("No valid text extracted from documents")

#         self._build_index(all_chunks)
        
#         if failed_files:
#             logger.warning(f"Failed on {len(failed_files)}/{len(doc_paths)} files")

#     def _build_index(self, chunks: List[str]):
#         """Build FAISS index with optimized settings"""
#         try:
#             logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
#             # Batch processing for memory efficiency
#             batch_size = 32
#             embeddings = np.zeros((len(chunks), self.dimension), dtype=np.float32)
            
#             for i in range(0, len(chunks), batch_size):
#                 batch = chunks[i:i + batch_size]
#                 embeddings[i:i + batch_size] = self.model.encode(batch, show_progress_bar=False)
            
#             # Use IndexIVFFlat for large datasets (>10k chunks)
#             if len(chunks) > 10000:
#                 quantizer = faiss.IndexFlatIP(self.dimension)
#                 self.index = faiss.IndexIVFFlat(quantizer, self.dimension, min(100, len(chunks)//2))
#                 self.index.train(embeddings)
#                 self.index.add(embeddings)
#                 self.index.nprobe = 4  # Balance speed/accuracy
#             else:
#                 self.index = faiss.IndexFlatIP(self.dimension)
#                 self.index.add(embeddings)
                
#             self.chunks = chunks
#             logger.info(f"Built index with {len(chunks)} chunks")
            
#         except Exception as e:
#             logger.error(f"Index build failed: {str(e)}")
#             raise RuntimeError(f"Index construction error: {str(e)}")

#     def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
#         """Advanced text chunking with sentence awareness"""
#         try:
#             sentences = simple_sentence_tokenizer(text)
#         except Exception as e:
#             logger.warning(f"Sentence tokenization failed: {str(e)}")
#             sentences = text.split('. ')
            
#         chunks = []
#         current_chunk = []
#         current_len = 0
        
#         for sent in sentences:
#             sent = sent.strip()
#             if not sent:
#                 continue
                
#             sent_len = len(sent.split())
            
#             if current_len + sent_len > chunk_size and current_chunk:
#                 chunks.append(' '.join(current_chunk))
#                 # Apply overlap
#                 current_chunk = current_chunk[-overlap:] if overlap else []
#                 current_len = sum(len(w.split()) for w in current_chunk)
                
#             current_chunk.append(sent)
#             current_len += sent_len
            
#         if current_chunk:
#             chunks.append(' '.join(current_chunk))
            
#         return chunks

#     def save(self, index_path: str, chunks_path: str):
#         """Atomic save operation with checksum verification"""
#         try:
#             if not self.index or not self.chunks:
#                 raise ValueError("Knowledge base not initialized")
                
#             temp_index = f"{index_path}.tmp"
#             temp_chunks = f"{chunks_path}.tmp"
            
#             Path(index_path).parent.mkdir(parents=True, exist_ok=True)
#             faiss.write_index(self.index, temp_index)
            
#             with open(temp_chunks, 'w', encoding='utf-8') as f:
#                 f.write("\n\n".join(self.chunks))
                
#             # Atomic rename
#             os.replace(temp_index, index_path)
#             os.replace(temp_chunks, chunks_path)
            
#             logger.info(f"Saved KB to {index_path} ({len(self.chunks)} chunks)")
            
#         except Exception as e:
#             if os.path.exists(temp_index):
#                 os.remove(temp_index)
#             if os.path.exists(temp_chunks):
#                 os.remove(temp_chunks)
#             logger.error(f"Save failed: {str(e)}")
#             raise

#     def load(self, index_path: str, chunks_path: str) -> bool:
#         """Safe load with corruption detection"""
#         try:
#             if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
#                 logger.error("KB files missing")
#                 return False
                
#             # Verify file integrity
#             if os.path.getsize(index_path) < 100 or os.path.getsize(chunks_path) < 10:
#                 logger.error("Corrupted KB files detected")
#                 return False
                
#             self.index = faiss.read_index(index_path)
            
#             with open(chunks_path, 'r', encoding='utf-8') as f:
#                 self.chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]
                
#             if len(self.chunks) != self.index.ntotal:
#                 logger.error("Index-chunk count mismatch")
#                 return False
                
#             logger.info(f"Loaded KB with {len(self.chunks)} chunks")
#             return True
            
#         except Exception as e:
#             logger.error(f"Load failed: {str(e)}")
#             self.index = None
#             self.chunks = []
#             return False

#     def query(self, text: str, top_k: int = 5, min_score: float = 0.5) -> List[Tuple[str, float]]:
#         """Enhanced query with fallback strategies"""
#         if not self.index or not self.chunks:
#             logger.warning("KB not loaded")
#             return []
            
#         try:
#             embedding = self.model.encode([text], convert_to_tensor=True)
#             embedding = embedding.cpu().numpy().astype('float32')
            
#             if isinstance(self.index, faiss.IndexIVFFlat):
#                 self.index.nprobe = min(8, self.index.nprobe * 2)  # Dynamic probe adjustment
                
#             distances, indices = self.index.search(embedding, top_k * 2)  # Over-fetch for filtering
            
#             results = []
#             for i, score in zip(indices[0], distances[0]):
#                 if 0 <= i < len(self.chunks) and score >= min_score:
#                     results.append((self.chunks[i], float(score)))
#                     if len(results) >= top_k:
#                         break
                        
#             return sorted(results, key=lambda x: x[1], reverse=True)
            
#         except Exception as e:
#             logger.error(f"Query failed: {str(e)}")
#             return []

class KnowledgeBase:
    @staticmethod
    def load_knowledge_base(index_path: str, chunks_path: str) -> Tuple[Optional[faiss.Index], List[str]]:
        """Load FAISS index and text chunks with validation"""
        try:
            if not os.path.exists(index_path) or not os.path.exists(chunks_path):
                logger.error("Knowledge base files not found")
                return None, []

            index = faiss.read_index(index_path)

            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]

            if not chunks:
                logger.error("No valid chunks found in knowledge base")
                return None, []

            logger.info(f"Loaded knowledge base with {len(chunks)} rules")
            return index, chunks

        except Exception as e:
            logger.error(f"Knowledge base load failed: {str(e)}")
            return None, []

    @staticmethod
    def query_knowledge_base(query_text: str, model: SentenceTransformer,
                           index: faiss.Index, chunks: List[str],
                           doc_type: str = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """Enhanced knowledge base query with document type filtering"""
        if not index or not chunks:
            return []

        try:
            # Add document type context to query
            if doc_type and doc_type in CONFIG['DOCUMENT_TYPES']:
                query_text = f"{doc_type} document regarding: {query_text}"

            query_embedding = model.encode([query_text])
            distances, indices = index.search(query_embedding, top_k * 3)  # Get more results for filtering

            results = []
            for i, score in zip(indices[0], distances[0]):
                if i < 0 or i >= len(chunks):  # Validate index
                    continue

                chunk = chunks[i]
                # Filter by document type keywords if specified
                if doc_type and not KnowledgeBase._chunk_matches_doc_type(chunk, doc_type):
                    continue

                similarity = 1 / (1 + score)
                if similarity > CONFIG['MIN_CONFIDENCE']:
                    results.append((chunk, similarity))

            return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

        except Exception as e:
            logger.error(f"Knowledge base query failed: {str(e)}")
            return []

    @staticmethod
    def _chunk_matches_doc_type(chunk: str, doc_type: str) -> bool:
        """Check if chunk contains doc-type specific keywords"""
        type_keywords = {
            "court_ruling": ["court", "judge", "ruling", "decision", "appeal", "transfer", "jurisdiction"],
            "contract": ["party", "clause", "agreement", "term", "obligation"],
            "regulatory_filing": ["filing", "disclosure", "report", "submit", "regulation"],
            "policy": ["policy", "procedure", "guideline", "compliance", "standard"]
        }
        chunk_lower = chunk.lower()
        return any(keyword in chunk_lower for keyword in type_keywords.get(doc_type, []))