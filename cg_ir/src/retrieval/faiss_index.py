"""
retrieval/faiss_index.py
========================
Dense retrieval index menggunakan sentence-transformers + FAISS.

Setiap dokumen di-encode menjadi dense vector, lalu disimpan di FAISS
IndexFlatIP. Karena vectors di-normalize terlebih dahulu, inner product
= cosine similarity.

Model default: paraphrase-multilingual-MiniLM-L12-v2
  - Mendukung Bahasa Indonesia
  - ~120MB, tidak butuh GPU
  - 384 dimensi
"""

import time
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer


class FAISSIndex:
    """
    Dense retrieval index: encode → normalize → FAISS IndexFlatIP.

    Args:
        model_name : sentence-transformers model name
    """

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        print(f"⏳ Loading embedding model '{self.model_name}'...")
        self.model       = SentenceTransformer(self.model_name)
        self.index       = None
        self.documents:  List[Dict]  = []
        self._embeddings: np.ndarray = None
        self.dim:        int         = None

    # ── Document text preparation ──────────────────────────────────────────

    def _doc_to_text(self, doc: Dict) -> str:
        """
        Combine relevant fields untuk encoding.
        Urutan: question > concept > topic (dari spesifik ke umum).
        Context tidak dimasukkan — terlalu panjang, bikin noise di embedding.
        """
        parts = [
            doc.get("question", ""),
            doc.get("concept", ""),
            doc.get("topic", ""),
        ]
        return " | ".join(p for p in parts if p)

    # ── Build index ────────────────────────────────────────────────────────

    def build(self, documents: List[Dict], batch_size: int = 64):
        """
        Encode semua dokumen dan build FAISS index.

        Args:
            documents  : output dari data_loader.load_corpus()
            batch_size : jumlah dokumen per batch saat encoding
        """
        self.documents = documents
        texts = [self._doc_to_text(d) for d in documents]

        print(f"⏳ Encoding {len(texts)} documents (batch_size={batch_size})...")
        t0 = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Normalize → cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        self.dim         = embeddings.shape[1]
        self._embeddings = embeddings

        # IndexFlatIP: exact search, inner product
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        elapsed = time.time() - t0
        print(f"✅ FAISS index built: {self.index.ntotal} vectors, "
              f"dim={self.dim}, took {elapsed:.1f}s")

    # ── Encode query ───────────────────────────────────────────────────────

    def encode_query(self, query: str) -> np.ndarray:
        """Encode dan normalize satu query string → (1, dim) float32."""
        vec = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        bloom_filter: Optional[List[int]] = None,
        candidate_indices: Optional[List[int]] = None,
    ) -> List[Tuple[int, float, Dict]]:
        """
        FAISS semantic search.

        Args:
            query             : query string (sudah di-enrich oleh graph)
            top_k             : jumlah hasil
            bloom_filter      : list bloom levels yang diizinkan [1..6]
            candidate_indices : jika ada, search hanya di dalam subset ini
                                (berasal dari graph has_question traversal)

        Returns:
            List of (doc_index, similarity_score, document)
        """
        self._check_built()
        query_vec = self.encode_query(query)

        if candidate_indices is not None and len(candidate_indices) > 0:
            results = self._search_subset(query_vec, candidate_indices, top_k * 3)
        else:
            scores, indices = self.index.search(query_vec, top_k * 3)
            results = [
                (int(i), float(s))
                for i, s in zip(indices[0], scores[0])
                if i >= 0
            ]

        # Bloom filter (post-retrieval)
        if bloom_filter:
            bloom_set = set(bloom_filter)
            results = [
                (i, s) for i, s in results
                if self.documents[i].get("bloom_level") in bloom_set
            ]

        return [(i, s, self.documents[i]) for i, s in results[:top_k]]

    def _search_subset(
        self,
        query_vec: np.ndarray,
        candidate_indices: List[int],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Dot product search dalam subset dokumen.
        O(n_candidates) — sangat cepat untuk dataset kecil.
        """
        cand_embs = self._embeddings[candidate_indices]   # (n, dim)
        sims      = (query_vec @ cand_embs.T)[0]          # (n,)
        ranked    = np.argsort(sims)[::-1][:top_k]
        return [(candidate_indices[int(i)], float(sims[i])) for i in ranked]

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self, dir_path: str):
        """Simpan FAISS index + embeddings + metadata ke disk."""
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{dir_path}/faiss.index")
        with open(f"{dir_path}/metadata.pkl", "wb") as f:
            pickle.dump({
                "documents":   self.documents,
                "embeddings":  self._embeddings,
                "dim":         self.dim,
                "model_name":  self.model_name,
            }, f)
        print(f"💾 FAISS index saved → {dir_path}/")

    def load(self, dir_path: str):
        """Load FAISS index dari disk (skip encoding ulang)."""
        self.index = faiss.read_index(f"{dir_path}/faiss.index")
        with open(f"{dir_path}/metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        self.documents    = meta["documents"]
        self._embeddings  = meta["embeddings"]
        self.dim          = meta["dim"]
        self.model_name   = meta.get("model_name", self.DEFAULT_MODEL)
        print(f"✅ FAISS index loaded ← {dir_path}/ ({len(self.documents)} docs, dim={self.dim})")

    def _check_built(self):
        if self.index is None:
            raise RuntimeError("Index belum dibangun. Panggil build() atau load() terlebih dahulu.")
