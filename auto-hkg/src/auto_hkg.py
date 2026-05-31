"""
Auto-HKG: Automated Hierarchical Knowledge Graph Constructor

Adapted from: "Beyond Static Question Banks: Dynamic Knowledge Expansion via
LLM-Automated Graph Construction and Adaptive Generation" (Wang et al., PVLDB 2026).

This module implements the Auto-HKG pipeline, which automatically constructs a
hierarchical educational knowledge graph from a structured question bank dataset.
Each question is processed through a large language model using a schema-constrained
prompt. The extracted information — including topic hierarchy, knowledge concepts,
prerequisite and successor relationships, problem-solving methods, and Bloom's
taxonomy level — is assembled into a directed graph (NetworkX DiGraph) that captures
both the conceptual structure of the domain and the relational dependencies between
knowledge components.

Dataset format (semicolon-separated CSV):
    Konteks   : The reading passage or context text for the question.
    Pertanyaan: The question text.
    Level Kognitif: Bloom's taxonomy level label (e.g., C1-Mengingat, C3-Mengaplikasikan).

Pipeline stages:
    1. Load and validate the CSV dataset.
    2. For each row, format a structured extraction prompt and call the LLM.
    3. Parse and validate the JSON response against the Auto-HKG schema.
    4. Pass validated extraction to GraphBuilder for node and edge assembly.
    5. Periodically save checkpoint to allow resuming interrupted runs.
    6. Export the final graph as JSON, CSV (nodes + edges), and summary statistics.
"""

import os
import json
import time
import logging
import hashlib
import pandas as pd
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from llm_client import LLMClient
from schema_validator import validate_extraction, SchemaError
from graph_builder import GraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("output/logs/auto_hkg.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HKGConfig:
    """
    Configuration dataclass for the Auto-HKG pipeline.

    Attributes:
        provider            : LLM provider to use. One of: openai, groq, anthropic,
                              gemini, ollama, unsloth.
        model               : Model name or alias. If None, uses the provider default
                              defined in DEFAULT_MODELS inside llm_client.py.
        max_new_tokens      : Maximum number of tokens to generate per LLM call.
                              Default 256 is sufficient for JSON output on short datasets.
                              Increase if you see JSON parse errors due to truncation.
        context_max_chars   : Maximum characters of context passed to the LLM prompt.
                              Default 500 covers most short passages. Increase for
                              longer documents.
        data_path           : Path to the semicolon-separated CSV dataset.
        output_dir          : Directory where graph outputs (JSON, CSV, stats) are saved.
        checkpoint_path     : Path to the checkpoint JSON file used for resuming
                              interrupted pipeline runs.
        batch_size          : Number of rows processed per batch before saving checkpoint.
        max_retries         : Number of retry attempts per row on LLM error or JSON parse failure.
        retry_delay         : Base delay in seconds between retry attempts (multiplied by attempt number).
        sleep_between_batches: Seconds to sleep between batches. Set to 0 for local inference.
        similarity_threshold: Cosine similarity threshold for merging duplicate concept nodes.
                              Not yet implemented; reserved for future fuzzy deduplication.
    """
    provider: str = "groq"
    model: str = None
    max_new_tokens: int = 256
    context_max_chars: int = 500

    data_path: str = "data/Knowledge_Base_Update.csv"
    output_dir: str = "output/graph"
    checkpoint_path: str = "output/logs/checkpoint.json"

    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 2.0
    sleep_between_batches: float = 1.0

    similarity_threshold: float = 0.75


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a knowledge graph extraction system for educational content.
Analyze the exam question below and extract structured information for a hierarchical knowledge graph.
Return ONLY a valid JSON object — no markdown, no code fences, no extra text.

Context: {context}
Question: {question}
Bloom's Taxonomy Level: {bloom_level}

Required JSON schema:
{{
  "topic_coarse": "<Broad subject category, 2-4 words. Example: Geographic Location>",
  "topic_fine": "<Specific sub-topic, 3-6 words. Example: Absolute and Relative Location>",
  "concepts": ["<Primary concept tested>", "<Additional concept if any — max 4 total>"],
  "methods": ["<Cognitive method or strategy required — empty [] if recall only>"],
  "bloom_level": "{bloom_level}",
  "prerequisites": ["<Concept needed BEFORE learning this — empty [] if none>"],
  "successors": ["<Concept that naturally follows this — empty [] if none>"],
  "difficulty": <1-5, map as: C1=1, C2=2, C3=3, C4=4, C5=5, C6=5>
}}

Rules:
- All string values in Bahasa Indonesia.
- concepts: minimum 1, maximum 4 items.
- prerequisites and successors represent knowledge dependencies, not just vocabulary.
- methods: use [] if the question only requires recall.
- Return JSON only, no explanation outside the object.
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AutoHKG:
    """
    Main pipeline class for automated hierarchical knowledge graph construction.
    """

    def __init__(self, config: HKGConfig):
        self.cfg = config
        self.client = LLMClient(
            provider=config.provider,
            model=config.model,
            max_new_tokens=config.max_new_tokens,
        )
        self.builder = GraphBuilder()
        self.checkpoint = self._load_checkpoint()
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path("output/logs").mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self) -> dict:
        p = Path(self.cfg.checkpoint_path)
        if p.exists():
            with open(p) as f:
                ck = json.load(f)
            log.info(f"Checkpoint loaded. Resuming from row {ck['processed']}.")
            return ck
        return {"processed": 0, "failed_ids": []}

    def _save_checkpoint(self):
        with open(self.cfg.checkpoint_path, "w") as f:
            json.dump(self.checkpoint, f, indent=2)

    @staticmethod
    def _row_id(row: pd.Series) -> str:
        raw = f"{row['Pertanyaan']}|{row['Level Kognitif']}"
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    def _extract_one(self, row: pd.Series) -> Optional[dict]:
        prompt = EXTRACTION_PROMPT.format(
            context=row["Konteks"][:self.cfg.context_max_chars],
            question=row["Pertanyaan"],
            bloom_level=row["Level Kognitif"]
        )

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                raw = self.client.complete(prompt)

                if raw is None:
                    raise ValueError("LLM returned None instead of a string.")

                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()

                if not cleaned.startswith("{"):
                    start = cleaned.find("{")
                    end   = cleaned.rfind("}") + 1
                    if start != -1 and end > start:
                        cleaned = cleaned[start:end]

                data = json.loads(cleaned)
                validate_extraction(data)
                data["_question"] = row["Pertanyaan"]
                data["_row_id"] = self._row_id(row)
                return data

            except json.JSONDecodeError as e:
                log.warning(f"JSON parse error on attempt {attempt}/{self.cfg.max_retries}: {e}")
                log.debug(f"Raw output was: {raw[:300]}")
            except SchemaError as e:
                log.warning(f"Schema validation failed on attempt {attempt}/{self.cfg.max_retries}: {e}")
            except Exception as e:
                log.warning(f"Unexpected error on attempt {attempt}/{self.cfg.max_retries}: {e}")

            time.sleep(self.cfg.retry_delay * attempt)

        return None

    def run(self):
        df = pd.read_csv(self.cfg.data_path, sep=";")
        df.columns = [c.strip() for c in df.columns]

        expected = {"Konteks", "Pertanyaan", "Level Kognitif"}
        if not expected.issubset(set(df.columns)):
            raise ValueError(
                f"Dataset is missing required columns. "
                f"Expected: {expected}. Found: {set(df.columns)}"
            )

        log.info(f"Dataset loaded: {len(df)} rows from '{self.cfg.data_path}'")

        start = self.checkpoint["processed"]
        df = df.iloc[start:].reset_index(drop=True)
        log.info(f"Starting from row {start}. Rows remaining: {len(df)}")

        failed = self.checkpoint.get("failed_ids", [])
        batches = range(0, len(df), self.cfg.batch_size)

        for batch_start in tqdm(batches, desc="Processing batches"):
            batch = df.iloc[batch_start: batch_start + self.cfg.batch_size]

            for _, row in batch.iterrows():
                result = self._extract_one(row)
                if result:
                    self.builder.add_extraction(result)
                else:
                    rid = self._row_id(row)
                    failed.append(rid)
                    log.error(
                        f"Extraction failed after {self.cfg.max_retries} attempts. "
                        f"Row ID: {rid} | Question: {row['Pertanyaan'][:80]}"
                    )

            self.checkpoint["processed"] += len(batch)
            self.checkpoint["failed_ids"] = failed
            self._save_checkpoint()

            time.sleep(self.cfg.sleep_between_batches)

        total = self.checkpoint["processed"]
        log.info(f"Pipeline complete. Total processed: {total} | Failed: {len(failed)}")
        self._save_outputs()

    def _save_outputs(self):
        out = Path(self.cfg.output_dir)
        G = self.builder.get_graph()

        graph_data = nx.node_link_data(G)
        with open(out / "knowledge_graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        nodes_df = pd.DataFrame([{"id": n, **G.nodes[n]} for n in G.nodes])
        edges_df = pd.DataFrame([
            {"source": u, "target": v, **d} for u, v, d in G.edges(data=True)
        ])
        nodes_df.to_csv(out / "nodes.csv", index=False)
        edges_df.to_csv(out / "edges.csv", index=False)

        def count_type(t):
            return len([n for n, d in G.nodes(data=True) if d.get("type") == t])

        stats = {
            "total_nodes"   : G.number_of_nodes(),
            "total_edges"   : G.number_of_edges(),
            "topics_coarse" : count_type("topic_coarse"),
            "topics_fine"   : count_type("topic_fine"),
            "concepts"      : count_type("concept"),
            "questions"     : count_type("question"),
            "methods"       : count_type("method"),
        }
        with open(out / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        log.info(f"Graph exported to '{out}/'")
        log.info(f"Stats: {stats}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-HKG: Automated Hierarchical Knowledge Graph Constructor"
    )
    parser.add_argument(
        "--provider", default="groq",
        choices=["openai", "groq", "anthropic", "gemini", "ollama", "unsloth"],
        help="LLM provider to use for knowledge extraction."
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name or alias. Uses provider default if not specified."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum tokens to generate per LLM call (default: 256)."
    )
    parser.add_argument(
        "--context-max-chars", type=int, default=500,
        help="Maximum characters of context passed to prompt (default: 500)."
    )
    parser.add_argument(
        "--data", default="data/Knowledge_Base_Update.csv",
        help="Path to the semicolon-separated CSV dataset."
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Number of rows per batch (default: 10)."
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds to sleep between batches (default: 1.0). Set to 0 for local inference."
    )
    args = parser.parse_args()

    cfg = HKGConfig(
        provider=args.provider,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        context_max_chars=args.context_max_chars,
        data_path=args.data,
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep,
    )

    pipeline = AutoHKG(cfg)
    pipeline.run()
