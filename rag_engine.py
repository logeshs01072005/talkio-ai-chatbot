"""
rag_engine.py
Hybrid RAG: ChromaDB (Vector) + NetworkX (Graph) + SQLite (Cache)
+ Reranking + Web Search support
"""

import sqlite3
import hashlib
import time
import re
import networkx as nx
import chromadb
import numpy as np


# ============================================================
# 1. VECTOR STORE — ChromaDB (persistent)
# ============================================================

class VectorStore:

    def __init__(self, persist_dir="./talkio_vectordb"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="talkio_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list, doc_id: str, doc_name: str) -> int:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(chunks, show_progress_bar=False).tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            return 0

        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        try:
            existing = self.collection.get(ids=ids)["ids"]
        except:
            existing = []

        new_ids, new_embeddings, new_chunks, new_meta = [], [], [], []
        for i, cid in enumerate(ids):
            if cid not in existing:
                new_ids.append(cid)
                new_embeddings.append(embeddings[i])
                new_chunks.append(chunks[i])
                new_meta.append({
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_idx": i
                })

        if new_ids:
            self.collection.add(
                ids=new_ids,
                embeddings=new_embeddings,
                documents=new_chunks,
                metadatas=new_meta
            )
        return len(new_ids)

    def search(self, query: str, k: int = 10) -> list:
        """Search top-k chunks — return more for reranking"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            query_embedding = model.encode([query]).tolist()
        except:
            return []

        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"]
        )

        chunks_with_source = []
        if results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                chunks_with_source.append({
                    "content": doc,
                    "doc_name": meta.get("doc_name", "Unknown PDF"),
                    "doc_id": meta.get("doc_id", ""),
                    "score": 1 - dist   # cosine similarity score
                })

        return chunks_with_source

    def get_all_docs(self) -> list:
        try:
            all_items = self.collection.get(include=["metadatas"])
            seen = {}
            for meta in all_items["metadatas"]:
                doc_id = meta.get("doc_id", "")
                if doc_id and doc_id not in seen:
                    seen[doc_id] = meta.get("doc_name", "Unknown")
            return [{"doc_id": k, "doc_name": v} for k, v in seen.items()]
        except:
            return []

    def remove_doc(self, doc_id: str):
        try:
            all_items = self.collection.get(include=["metadatas"])
            ids_to_delete = [
                all_items["ids"][i]
                for i, meta in enumerate(all_items["metadatas"])
                if meta.get("doc_id") == doc_id
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception as e:
            print(f"Remove doc error: {e}")
            return 0

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.client.delete_collection("talkio_chunks")
        self.collection = self.client.get_or_create_collection(
            name="talkio_chunks",
            metadata={"hnsw:space": "cosine"}
        )


# ============================================================
# 2. RERANKER — Cross-encoder style reranking
# ============================================================

class Reranker:
    """
    Reranks retrieved chunks by relevance to query.
    Uses keyword overlap + semantic score for fast reranking.
    """

    def rerank(self, query: str, chunks: list, top_k: int = 5) -> list:
        """
        Score each chunk against the query and return top_k.
        Scoring = semantic score (from vector search) + keyword overlap bonus
        """
        if not chunks:
            return []

        query_words = set(query.lower().split())
        scored = []

        for chunk in chunks:
            content    = chunk["content"].lower()
            base_score = chunk.get("score", 0.5)

            # Keyword overlap bonus
            chunk_words   = set(content.split())
            overlap       = len(query_words & chunk_words)
            keyword_bonus = overlap * 0.02   # small bonus per matching word

            # Exact phrase bonus
            phrase_bonus = 0.1 if query.lower() in content else 0

            # Length penalty (very short chunks are less useful)
            length_penalty = 0 if len(content) > 100 else -0.05

            final_score = base_score + keyword_bonus + phrase_bonus + length_penalty

            scored.append({
                **chunk,
                "final_score": final_score,
                "keyword_matches": overlap
            })

        # Sort by final score descending
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored[:top_k]


# ============================================================
# 3. WEB SEARCH — DuckDuckGo (no API key needed)
# ============================================================

class WebSearcher:
    """
    Free web search using DuckDuckGo.
    Used when PDF context is insufficient.
    """

    def search(self, query: str, max_results: int = 3) -> list:
        """Search the web and return results"""
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "body":  r.get("body", ""),
                        "url":   r.get("href", "")
                    })
            return results
        except ImportError:
            return [{"title": "Web search unavailable",
                     "body": "Install duckduckgo-search: pip install duckduckgo-search",
                     "url": ""}]
        except Exception as e:
            print(f"Web search error: {e}")
            return []

    def format_results(self, results: list) -> str:
        """Format web results into context string"""
        if not results:
            return ""
        lines = ["🌐 Web Search Results:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**[{i}] {r['title']}**")
            lines.append(r["body"])
            if r["url"]:
                lines.append(f"Source: {r['url']}")
            lines.append("---")
        return "\n".join(lines)


# ============================================================
# 4. GRAPH STORE — NetworkX
# ============================================================

class GraphStore:

    def __init__(self):
        self.graph = nx.DiGraph()

    def extract_entities(self, text: str) -> list:
        pattern   = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'
        entities  = re.findall(pattern, text)
        stopwords = {"The", "A", "An", "In", "On", "At", "To", "For", "Of", "And", "Or"}
        entities  = [e for e in entities if e not in stopwords and len(e) > 2]
        return list(set(entities))[:20]

    def build_from_chunks(self, chunks: list, doc_id: str = ""):
        for chunk in chunks:
            entities = self.extract_entities(chunk)
            for entity in entities:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, mentions=1)
                else:
                    self.graph.nodes[entity]["mentions"] += 1
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    src, tgt = entities[i], entities[j]
                    if self.graph.has_edge(src, tgt):
                        self.graph[src][tgt]["weight"] += 1
                    else:
                        self.graph.add_edge(src, tgt, weight=1)

    def get_graph_context(self, query: str) -> str:
        query_words = set(query.lower().split())
        matched = [
            node for node in self.graph.nodes
            if any(word in node.lower() for word in query_words)
        ]
        related = set(matched)
        for node in matched:
            neighbors = sorted(
                self.graph.neighbors(node),
                key=lambda n: self.graph[node][n].get("weight", 0),
                reverse=True
            )
            related.update(neighbors[:3])

        if not related:
            return ""

        lines = ["🔗 Knowledge Graph:"]
        for entity in list(related)[:5]:
            neighbors = list(self.graph.neighbors(entity))[:3]
            if neighbors:
                lines.append(f"  • {entity} → {', '.join(neighbors)}")
            else:
                lines.append(f"  • {entity}")
        return "\n".join(lines)

    def node_count(self):
        return self.graph.number_of_nodes()


# ============================================================
# 5. CACHE STORE — SQLite
# ============================================================

class CacheStore:

    def __init__(self, db_path="./talkio_cache.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash TEXT PRIMARY KEY,
                query      TEXT,
                answer     TEXT,
                created_at REAL
            )
        """)
        self.conn.commit()

    def _hash(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def get(self, query: str):
        h   = self._hash(query)
        row = self.conn.execute(
            "SELECT answer, created_at FROM query_cache WHERE query_hash=?", (h,)
        ).fetchone()
        if row:
            answer, created_at = row
            if time.time() - created_at < 3600:
                return answer
        return None

    def set(self, query: str, answer: str):
        if self.count() >= 100:
            self.conn.execute("""
                DELETE FROM query_cache WHERE query_hash IN (
                    SELECT query_hash FROM query_cache
                    ORDER BY created_at ASC LIMIT 20
                )
            """)
        h = self._hash(query)
        self.conn.execute("""
            INSERT OR REPLACE INTO query_cache
            (query_hash, query, answer, created_at)
            VALUES (?, ?, ?, ?)
        """, (h, query, answer, time.time()))
        self.conn.commit()

    def clear(self):
        self.conn.execute("DELETE FROM query_cache")
        self.conn.commit()

    def count(self):
        return self.conn.execute(
            "SELECT COUNT(*) FROM query_cache"
        ).fetchone()[0]


# ============================================================
# 6. HYBRID RAG ENGINE
# ============================================================

class HybridRAGEngine:

    def __init__(self):
        self.vector_store = VectorStore()
        self.graph_store  = GraphStore()
        self.cache_store  = CacheStore()
        self.reranker     = Reranker()
        self.web_searcher = WebSearcher()
        self.loaded_docs  = {}
        self.is_loaded    = False

    def load_document(self, chunks: list, doc_id: str, doc_name: str = "document") -> dict:
        added = self.vector_store.add_chunks(chunks, doc_id, doc_name)
        self.graph_store.build_from_chunks(chunks, doc_id)
        self.loaded_docs[doc_id] = doc_name
        self.is_loaded = True
        return {
            "chunks_added": added,
            "total_vectors": self.vector_store.count(),
            "graph_nodes": self.graph_store.node_count(),
            "doc_name": doc_name
        }

    def retrieve(self, query: str, k: int = 5, use_web: bool = False) -> dict:
        """
        Full hybrid retrieval pipeline:
        1. Cache check
        2. Vector search (top 10)
        3. Reranking (top 5)
        4. Graph context
        5. Web search (optional)
        """

        # 1. Cache check
        cached = self.cache_store.get(query)
        if cached:
            return {
                "context": cached,
                "source": "cache",
                "chunks": [],
                "graph_context": "",
                "web_results": []
            }

        # 2. Vector search (fetch more for reranking)
        raw_chunks = self.vector_store.search(query, k=10)

        # 3. Reranking — pick best chunks
        reranked_chunks = self.reranker.rerank(query, raw_chunks, top_k=k)

        # 4. Graph context
        graph_context = self.graph_store.get_graph_context(query)

        # 5. Web search (if enabled or PDF has no good results)
        web_results  = []
        web_context  = ""
        top_score    = reranked_chunks[0]["final_score"] if reranked_chunks else 0

        if use_web or (self.is_loaded and top_score < 0.3):
            web_results = self.web_searcher.search(query, max_results=3)
            web_context = self.web_searcher.format_results(web_results)

        # Build combined context
        combined = ""
        if reranked_chunks:
            combined += "📄 Relevant Document Sections (reranked):\n\n"
            for chunk in reranked_chunks:
                score = chunk.get("final_score", 0)
                combined += f"**[{chunk['doc_name']}]** _(relevance: {score:.2f})_\n"
                combined += chunk["content"] + "\n\n---\n"

        if graph_context:
            combined += f"\n{graph_context}\n"

        if web_context:
            combined += f"\n{web_context}"

        source = "cache" if cached else "hybrid+web" if web_results else "hybrid"

        return {
            "context": combined,
            "source": source,
            "chunks": reranked_chunks,
            "graph_context": graph_context,
            "web_results": web_results,
            "top_score": top_score
        }

    def web_only_search(self, query: str) -> dict:
        """Search web only (no PDF needed)"""
        web_results = self.web_searcher.search(query, max_results=3)
        web_context = self.web_searcher.format_results(web_results)
        return {
            "context": web_context,
            "source": "web",
            "web_results": web_results
        }

    def remove_document(self, doc_id: str) -> int:
        removed = self.vector_store.remove_doc(doc_id)
        if doc_id in self.loaded_docs:
            del self.loaded_docs[doc_id]
        self.is_loaded = len(self.loaded_docs) > 0
        return removed

    def get_loaded_docs(self) -> list:
        return [{"doc_id": k, "doc_name": v} for k, v in self.loaded_docs.items()]

    def cache_answer(self, query: str, answer: str):
        self.cache_store.set(query, answer)

    def clear_all(self):
        self.vector_store.clear()
        self.cache_store.clear()
        self.graph_store = GraphStore()
        self.loaded_docs = {}
        self.is_loaded   = False

    def stats(self) -> dict:
        return {
            "vectors": self.vector_store.count(),
            "graph_nodes": self.graph_store.node_count(),
            "cached_queries": self.cache_store.count(),
            "loaded_docs": len(self.loaded_docs),
            "loaded": self.is_loaded
        }