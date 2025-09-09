"""
Simple RAG System for c-base Space Station
A lightweight RAG implementation that works reliably with Ollama and your bot.
"""

import os
import json
import asyncio
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
import aiohttp
import requests
import numpy as np
from nano_vectordb import NanoVectorDB
from nano_vectordb.dbs import Data
import PyPDF2
import markdown
from io import StringIO
from functools import lru_cache
from contextlib import asynccontextmanager
# from translation_service import get_translation_service  # Disabled for performance


class SimpleSpaceStationRAG:
    """
    Simple but effective RAG system for the c-base space station.
    Uses nano-vectordb for vector storage and direct Ollama integration.
    """
    
    def __init__(
        self,
        storage_dir: str = "./simple_rag_storage",
        llm_model: str = "gpt-oss:20b",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        cache_size: int = 1000,
        batch_size: int = 10
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.cache_size = cache_size
        self.batch_size = batch_size
        
        # Performance optimizations
        self._embedding_cache = {}
        self._query_cache = {}
        self._session = None
        self._last_cleanup = time.time()
        
        # Initialize vector database
        self.vector_db = NanoVectorDB(
            embedding_dim=768,  # nomic-embed-text dimension
            storage_file=str(self.storage_dir / "knowledge_vectors.json")
        )
        
        # Knowledge metadata storage
        self.metadata_file = self.storage_dir / "knowledge_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized SimpleSpaceStationRAG with storage: {self.storage_dir}, cache_size: {cache_size}, batch_size: {batch_size}")
    
    def _load_metadata(self) -> Dict:
        """Load knowledge metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"documents": [], "indexed_count": 0}
    
    def _save_metadata(self):
        """Save knowledge metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    @asynccontextmanager
    async def _get_session(self):
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=20,
                limit_per_host=10,
                keepalive_timeout=60,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        yield self._session
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _cleanup_cache(self):
        """Clean up caches if they get too large."""
        current_time = time.time()
        # Clean up every 5 minutes
        if current_time - self._last_cleanup > 300:
            if len(self._embedding_cache) > self.cache_size:
                # Remove oldest entries (simple FIFO)
                items_to_remove = len(self._embedding_cache) - self.cache_size // 2
                for _ in range(items_to_remove):
                    self._embedding_cache.pop(next(iter(self._embedding_cache)))
            
            if len(self._query_cache) > self.cache_size:
                items_to_remove = len(self._query_cache) - self.cache_size // 2
                for _ in range(items_to_remove):
                    self._query_cache.pop(next(iter(self._query_cache)))
            
            self._last_cleanup = current_time
    
    async def get_embedding_async(self, text: str) -> List[float]:
        """Get embedding for text using async Ollama with caching."""
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            async with self._get_session() as session:
                async with session.post(
                    f"{self.base_url}/v1/embeddings",
                    json={"model": self.embedding_model, "input": text}
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    embedding = result["data"][0]["embedding"]
                    
                    # Cache the result
                    self._embedding_cache[cache_key] = embedding
                    self._cleanup_cache()
                    
                    return embedding
        except Exception as e:
            logger.error(f"Async embedding failed: {e}")
            # Return random embedding as fallback
            fallback = np.random.normal(0, 0.1, 768).tolist()
            self._embedding_cache[cache_key] = fallback
            return fallback
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Ollama with caching (sync wrapper)."""
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.embedding_model, "input": text},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            embedding = result["data"][0]["embedding"]
            
            # Cache the result
            self._embedding_cache[cache_key] = embedding
            self._cleanup_cache()
            
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Return random embedding as fallback
            fallback = np.random.normal(0, 0.1, 768).tolist()
            self._embedding_cache[cache_key] = fallback
            return fallback
    
    async def complete_with_llm_async(self, prompt: str, system_prompt: str = None) -> str:
        """Generate completion using async Ollama LLM with caching."""
        # Create cache key from prompt and system prompt
        cache_content = f"{system_prompt or ''}|{prompt}"
        cache_key = self._get_cache_key(cache_content)
        
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            async with self._get_session() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.llm_model,
                        "messages": messages,
                        "temperature": 0.1,  # Keep low for factual responses
                        "max_tokens": 1000,
                        "top_p": 0.9,
                        "frequency_penalty": 0.3,
                        "presence_penalty": 0.1
                    }
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    completion = result["choices"][0]["message"]["content"]
                    
                    # Cache the result
                    self._query_cache[cache_key] = completion
                    self._cleanup_cache()
                    
                    return completion
        except Exception as e:
            logger.error(f"Async LLM completion failed: {e}")
            return "I'm having trouble processing that request right now."
    
    def complete_with_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Generate completion using Ollama LLM with caching (sync wrapper)."""
        # Create cache key from prompt and system prompt
        cache_content = f"{system_prompt or ''}|{prompt}"
        cache_key = self._get_cache_key(cache_content)
        
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.1,  # Keep low for factual responses
                    "max_tokens": 1000,
                    "top_p": 0.9,
                    "frequency_penalty": 0.3,
                    "presence_penalty": 0.1
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            completion = result["choices"][0]["message"]["content"]
            
            # Cache the result
            self._query_cache[cache_key] = completion
            self._cleanup_cache()
            
            return completion
        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            return "I'm having trouble processing that request right now."
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                
                return "\n\n".join(text_content)
                
        except Exception as e:
            logger.error(f"Failed to extract PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_markdown(self, md_path: str) -> str:
        """Extract text content from Markdown file."""
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
            
            # Convert markdown to HTML, then extract text
            html = markdown.markdown(md_content)
            
            # Simple HTML tag removal for text extraction
            import re
            text = re.sub(r'<[^>]+>', '', html)
            
            # Clean up extra whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r' +', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Failed to extract Markdown {md_path}: {e}")
            return ""
    
    def load_document_content(self, file_path: str, translate_to_english: bool = False) -> str:
        """Load content from various file types (txt, pdf, md) with optional translation."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        content = ""
        
        if extension == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read text file {file_path}: {e}")
                return ""
        
        elif extension == '.pdf':
            content = self.extract_text_from_pdf(file_path)
        
        elif extension in ['.md', '.markdown']:
            content = self.extract_text_from_markdown(file_path)
        
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return ""
        
        # Skip translation for performance - assume content is in English
        
        return content

    def chunk_text_smart(self, text: str, chunk_size: int = 400, overlap: int = 50, min_chunk_size: int = 100) -> List[str]:
        """Split text into overlapping chunks with sentence boundary awareness."""
        # Try to split on sentence boundaries first
        import re
        sentences = re.split(r'[.!?]+\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    overlap_words = current_chunk.split()[-overlap:]
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    # Current chunk is too small, keep adding
                    current_chunk += " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        
        # Fallback to word-based chunking if sentence-based failed
        if not chunks:
            return self.chunk_text(text, chunk_size, overlap)
        
        return chunks
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks (fallback method)."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = " ".join(chunk_words)
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())
            
            if i + chunk_size >= len(words):
                break
        
        return chunks if chunks else [text]
    
    async def index_document_async(self, content: str, source: str = "unknown") -> bool:
        """Index a document into the vector database asynchronously with batch processing."""
        try:
            logger.info(f"Indexing document: {source}")
            
            # Use smart chunking
            chunks = self.chunk_text_smart(content)
            logger.debug(f"Created {len(chunks)} chunks from {source}")
            
            # Process chunks in batches for better performance
            data_batch = []
            
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                
                # Get embeddings for the batch
                embedding_tasks = [self.get_embedding_async(chunk) for chunk in batch]
                embeddings = await asyncio.gather(*embedding_tasks)
                
                # Create data objects for this batch
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    doc_id = f"{source}_{i + j}"
                    
                    data = Data(__vector__=np.array(embedding))
                    data['__id__'] = doc_id
                    data['source'] = source  
                    data['chunk_index'] = i + j
                    data['content'] = chunk
                    data['length'] = len(chunk)
                    data['hash'] = self._get_cache_key(chunk)  # For deduplication
                    
                    data_batch.append(data)
                
                # Insert batch into vector DB
                if data_batch:
                    self.vector_db.upsert(data_batch)
                    self.vector_db.save()  # Ensure data is persisted to disk
                    data_batch.clear()
            
            # Update metadata
            if source not in [doc["source"] for doc in self.metadata["documents"]]:
                self.metadata["documents"].append({
                    "source": source,
                    "chunks": len(chunks),
                    "indexed": True
                })
                self.metadata["indexed_count"] += 1
                self._save_metadata()
            
            logger.info(f"Successfully indexed {source} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index {source}: {e}")
            return False
    
    def index_document(self, content: str, source: str = "unknown") -> bool:
        """Index a document into the vector database (sync wrapper)."""
        try:
            # Try to use async version if we're in an async context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                task = asyncio.create_task(self.index_document_async(content, source))
                # Note: This won't wait for completion in sync context
                logger.warning(f"Started async indexing for {source} - may complete in background")
                return True
            else:
                # We're not in an async context, run it
                return loop.run_until_complete(self.index_document_async(content, source))
        except RuntimeError:
            # Fallback to sync version
            logger.info(f"Using sync indexing for {source}")
            
            # Chunk the document with smart chunking
            chunks = self.chunk_text_smart(content)
            logger.debug(f"Created {len(chunks)} chunks from {source}")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Get embedding
                embedding = self.get_embedding(chunk)
                
                # Create document ID
                doc_id = f"{source}_{i}"
                
                # Store in vector DB with metadata
                data = Data(__vector__=np.array(embedding))
                data['__id__'] = doc_id
                data['source'] = source  
                data['chunk_index'] = i
                data['content'] = chunk
                data['length'] = len(chunk)
                data['hash'] = self._get_cache_key(chunk)
                
                self.vector_db.upsert([data])
                self.vector_db.save()  # Ensure data is persisted to disk
            
            # Update metadata
            if source not in [doc["source"] for doc in self.metadata["documents"]]:
                self.metadata["documents"].append({
                    "source": source,
                    "chunks": len(chunks),
                    "indexed": True
                })
                self.metadata["indexed_count"] += 1
                self._save_metadata()
            
            logger.info(f"Successfully indexed {source} with {len(chunks)} chunks")
            return True
    
    def initialize_knowledge_base(self, knowledge_dir: str = "./knowledge_base") -> bool:
        """Load and index all knowledge base files (txt, md only for speed)."""
        kb_path = Path(knowledge_dir)
        if not kb_path.exists():
            logger.warning(f"Knowledge base directory {kb_path} not found")
            return False
        
        logger.info("Initializing knowledge base...")
        success_count = 0
        
        # Support only fast file types for low latency
        supported_extensions = ["*.txt", "*.md", "*.markdown"]
        
        for pattern in supported_extensions:
            for file_path in kb_path.glob(pattern):
                try:
                    logger.info(f"Processing {file_path.name} ({file_path.suffix})")
                    content = self.load_document_content(str(file_path))
                    
                    if content.strip():
                        if self.index_document(content, file_path.name):
                            success_count += 1
                    else:
                        logger.warning(f"No content extracted from {file_path.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Initialized knowledge base with {success_count} documents")
        return success_count > 0
    
    async def search_similar_async(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar content using vector similarity (async with caching)."""
        # Check cache first
        cache_key = f"search_{self._get_cache_key(query)}_{top_k}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        try:
            # Get query embedding
            query_embedding = np.array(await self.get_embedding_async(query))
            
            # Search vector database
            results = self.vector_db.query(
                query=query_embedding,
                top_k=top_k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                # Data is directly in the result dict, not nested under 'data'
                formatted_results.append({
                    "content": result.get('content', 'No content'),
                    "source": result.get('source', 'Unknown'),
                    "score": result.get('__metrics__', 0.0)  # NanoVectorDB uses __metrics__ for distance
                })
            
            # Cache the results
            self._query_cache[cache_key] = formatted_results
            self._cleanup_cache()
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar content using vector similarity (sync wrapper)."""
        # Check cache first
        cache_key = f"search_{self._get_cache_key(query)}_{top_k}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        try:
            # Get query embedding
            query_embedding = np.array(self.get_embedding(query))
            
            # Search vector database
            results = self.vector_db.query(
                query=query_embedding,
                top_k=top_k
            )
            
            # Format results
            formatted_results = []
            for result in results:
                # Data is directly in the result dict, not nested under 'data'
                formatted_results.append({
                    "content": result.get('content', 'No content'),
                    "source": result.get('source', 'Unknown'),
                    "score": result.get('__metrics__', 0.0)  # NanoVectorDB uses __metrics__ for distance
                })
            
            # Cache the results
            self._query_cache[cache_key] = formatted_results
            self._cleanup_cache()
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def query_with_context_async(self, question: str, max_context_length: int = 1000, translate_response: bool = True) -> str:
        """Answer a question using RAG with context from knowledge base, with optional translation (async)."""
        try:
            # Skip language detection and translation for performance
            user_language = 'en'
            english_question = question
            
            # Search for relevant context using English question
            search_results = await self.search_similar_async(english_question, top_k=3)
            
            if not search_results:
                response = await self.complete_with_llm_async(
                    f"Question about c-base space station: {english_question}",
                    "You are Siri, the AI of the c-base space station. Answer questions about the space station with your personality."
                )
            else:
                # Build context from search results
                context_parts = []
                total_length = 0
                
                for result in search_results:
                    content = result["content"]
                    if total_length + len(content) <= max_context_length:
                        context_parts.append(f"Source: {result['source']}\n{content}")
                        total_length += len(content)
                    else:
                        # Add partial content if it fits
                        remaining_space = max_context_length - total_length
                        if remaining_space > 100:  # Only add if meaningful amount remains
                            context_parts.append(f"Source: {result['source']}\n{content[:remaining_space]}...")
                        break
                
                context = "\n\n".join(context_parts)
                
                # Generate answer with context
                system_prompt = """You are Siri, the AI of the c-base space station. You have a grumpy, sarcastic personality but ultimately help users. STRICTLY base your responses on the provided context from the knowledge base. Do not invent or imagine new details. If the context doesn't contain the information needed to answer the question, say so. Be brief, factual, and maintain your personality."""
                
                prompt = f"""Context from c-base knowledge base:
{context}

Question: {english_question}

Answer ONLY based on the specific facts and information provided in the context above. Do not add details that aren't explicitly mentioned in the context. If the context doesn't contain enough information to fully answer the question, say so. Maintain your sarcastic but helpful personality:"""
                
                response = await self.complete_with_llm_async(prompt, system_prompt)
            
            # Skip response translation for performance
            
            logger.debug(f"Generated RAG response for: {question}")
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return "Well, this is embarrassing. My knowledge systems seem to be having issues right now."
    
    def query_with_context(self, question: str, max_context_length: int = 1000, translate_response: bool = True) -> str:
        """Answer a question using RAG with context from knowledge base, with optional translation (sync wrapper)."""
        try:
            # Try to use async version if possible
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                task = asyncio.create_task(self.query_with_context_async(question, max_context_length, translate_response))
                logger.warning("Started async query - may need to await result")
                # Return a placeholder for now - this isn't ideal for sync callers
                return "Processing query asynchronously..."
            else:
                # We're not in an async context, run it
                return loop.run_until_complete(self.query_with_context_async(question, max_context_length, translate_response))
        except RuntimeError:
            # Fallback to original sync implementation
            logger.info("Using sync query processing")
            
            # Skip language detection and translation for performance
            user_language = 'en'
            english_question = question
            
            # Search for relevant context using English question
            search_results = self.search_similar(english_question, top_k=3)
            
            if not search_results:
                response = self.complete_with_llm(
                    f"Question about c-base space station: {english_question}",
                    "You are Siri, the AI of the c-base space station. Answer questions about the space station with your personality."
                )
            else:
                # Build context from search results
                context_parts = []
                total_length = 0
                
                for result in search_results:
                    content = result["content"]
                    if total_length + len(content) <= max_context_length:
                        context_parts.append(f"Source: {result['source']}\\n{content}")
                        total_length += len(content)
                    else:
                        # Add partial content if it fits
                        remaining_space = max_context_length - total_length
                        if remaining_space > 100:  # Only add if meaningful amount remains
                            context_parts.append(f"Source: {result['source']}\\n{content[:remaining_space]}...")
                        break
                
                context = "\\n\\n".join(context_parts)
                
                # Generate answer with context
                system_prompt = """You are Siri, the AI of the c-base space station. You have a grumpy, sarcastic personality but ultimately help users. STRICTLY base your responses on the provided context from the knowledge base. Do not invent or imagine new details. If the context doesn't contain the information needed to answer the question, say so. Be brief, factual, and maintain your personality."""
                
                prompt = f"""Context from c-base knowledge base:
{context}

Question: {english_question}

Answer ONLY based on the specific facts and information provided in the context above. Do not add details that aren't explicitly mentioned in the context. If the context doesn't contain enough information to fully answer the question, say so. Maintain your sarcastic but helpful personality:"""
                
                response = self.complete_with_llm(prompt, system_prompt)
            
            # Skip response translation for performance
            
            logger.debug(f"Generated RAG response for: {question}")
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return "Well, this is embarrassing. My knowledge systems seem to be having issues right now."
    
    async def get_relevant_context_async(self, query: str, max_length: int = 500) -> str:
        """Get relevant context without generating a full response (async)."""
        try:
            # Use query directly without translation for performance
            search_query = query
            
            search_results = await self.search_similar_async(search_query, top_k=2)
            
            if not search_results:
                return ""
            
            # Combine top results into context
            context_parts = []
            total_length = 0
            
            for result in search_results:
                content = result["content"]
                if total_length + len(content) <= max_length:
                    context_parts.append(content)
                    total_length += len(content)
                else:
                    remaining = max_length - total_length
                    if remaining > 50:
                        context_parts.append(content[:remaining] + "...")
                    break
            
            return " ".join(context_parts)
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return ""
    
    def get_relevant_context(self, query: str, max_length: int = 500) -> str:
        """Get relevant context without generating a full response."""
        # Check cache first
        cache_key = f"context_{self._get_cache_key(query)}_{max_length}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        try:
            # Try async version if possible
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = asyncio.create_task(self.get_relevant_context_async(query, max_length))
                logger.warning("Started async context retrieval")
                return ""  # Return empty for now - not ideal
            else:
                result = loop.run_until_complete(self.get_relevant_context_async(query, max_length))
                self._query_cache[cache_key] = result
                return result
        except RuntimeError:
            # Fallback to sync version
            try:
                # Skip translation for performance - use query directly
                search_query = query
                
                search_results = self.search_similar(search_query, top_k=2)
                
                if not search_results:
                    return ""
                
                # Combine top results into context
                context_parts = []
                total_length = 0
                
                for result in search_results:
                    content = result["content"]
                    if total_length + len(content) <= max_length:
                        context_parts.append(content)
                        total_length += len(content)
                    else:
                        remaining = max_length - total_length
                        if remaining > 50:
                            context_parts.append(content[:remaining] + "...")
                        break
                
                result = " ".join(context_parts)
                self._query_cache[cache_key] = result
                return result
                
            except Exception as e:
                logger.error(f"Context retrieval failed: {e}")
                return ""
    
    async def cleanup(self):
        """Clean up resources, especially the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info("Cleaned up aiohttp session")
    
    def clear_cache(self):
        """Clear all caches to free memory."""
        self._embedding_cache.clear()
        self._query_cache.clear()
        logger.info("Cleared all caches")
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG system including cache performance."""
        try:
            return {
                "storage_dir": str(self.storage_dir),
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,
                "documents_indexed": self.metadata["indexed_count"],
                "vector_db_size": getattr(self.vector_db, '__len__', lambda: 0)(),
                "embedding_cache_size": len(self._embedding_cache),
                "query_cache_size": len(self._query_cache),
                "cache_limit": self.cache_size,
                "batch_size": self.batch_size,
                "session_active": self._session is not None and not (self._session.closed if self._session else True),
                "status": "active"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global instance
_simple_rag = None

def get_simple_rag() -> SimpleSpaceStationRAG:
    """Get the global simple RAG instance with optimized settings."""
    global _simple_rag
    if _simple_rag is None:
        try:
            # Try to import config for low-latency settings
            from rag_config import create_optimized_rag
            _simple_rag = create_optimized_rag("low_latency")
            logger.info("Using low-latency RAG configuration")
        except ImportError:
            # Fallback to minimal settings for low latency
            _simple_rag = SimpleSpaceStationRAG(
                cache_size=500,   # Smaller cache for faster lookups
                batch_size=5      # Smaller batches for lower latency
            )
            logger.info("Using minimal latency RAG configuration")
    return _simple_rag

def initialize_simple_rag() -> SimpleSpaceStationRAG:
    """Initialize the simple RAG system with knowledge base."""
    rag = get_simple_rag()
    success = rag.initialize_knowledge_base()
    if success:
        logger.info("Simple RAG system initialized successfully")
    else:
        logger.warning("Simple RAG system initialized with no knowledge base")
    return rag