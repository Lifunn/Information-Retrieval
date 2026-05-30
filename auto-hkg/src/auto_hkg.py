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
        data_path           : Path to the semicolon-separated CSV dataset.
        output_dir          : Directory where graph outputs (JSON, CSV, stats) are saved.
        checkpoint_path     : Path to the checkpoint JSON file used for resuming
                              interrupted pipeline runs.
        batch_size          : Number of rows processed per batch before saving checkpoint.
                              Recommended: 5 for local Unsloth inference, 10 for API providers.
        max_retries         : Number of retry attempts per row on LLM error or JSON parse failure.
        retry_delay         : Base delay in seconds between retry attempts (multiplied by attempt number).
        sleep_between_batches: Seconds to sleep between batches. Set to 0 for local inference.
                               For API providers, use 0.5-1.0 to avoid rate limits.
        similarity_threshold: Cosine similarity threshold for merging duplicate concept nodes.
                              Not yet implemented; reserved for future fuzzy deduplication.
    """
    provider: str = "groq"
    model: str = None

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
You are a knowledge graph extraction system specialized in educational content analysis.

Your task is to analyze a single exam question along with its reading passage and extract
structured information that will be used to construct a hierarchical educational knowledge graph.
The output will be used downstream for adaptive learning and personalized exercise recommendation.

Read the following input carefully before extracting:

---
Context (reading passage):
{context}

Question:
{question}

Bloom's Taxonomy Level:
{bloom_level}
---

Extract the following fields and return ONLY a valid JSON object with no additional text,
no markdown formatting, no code fences, and no explanation outside the JSON structure.

Required JSON schema:
{{
  "topic_coarse": "<Broad subject category. Use 2-4 words. Example: Geographic Location>",
  "topic_fine":   "<Specific sub-topic within the broad category. Use 3-6 words. Example: Absolute and Relative Location>",
  "concepts": [
    "<Primary knowledge concept tested by this question>",
    "<Additional concept if applicable — include 1 to 4 concepts total>"
  ],
  "methods": [
    "<Problem-solving method or cognitive strategy required to answer this question>",
    "<Add more methods if multiple strategies are needed — leave as empty list [] if none apply>"
  ],
  "bloom_level": "{bloom_level}",
  "prerequisites": [
    "<Knowledge concept that must be understood BEFORE this concept can be learned>",
    "<Add all logically necessary prerequisite concepts — leave as empty list [] if none>"
  ],
  "successors": [
    "<Knowledge concept that logically follows or builds upon this concept>",
    "<Add all natural successor concepts — leave as empty list [] if none>"
  ],
  "difficulty": <Integer from 1 to 5. Map Bloom level as follows: C1=1, C2=2, C3=3, C4=4, C5=5, C6=5>
}}

Extraction rules:
1. All string values must be written in Bahasa Indonesia, matching the language of the input.
2. The 'concepts' list must contain at least one concept and no more than four.
3. 'topic_coarse' must represent the broadest subject category (e.g., a school subject chapter heading).
4. 'topic_fine' must represent the most specific sub-topic directly tested by the question.
5. 'prerequisites' should list concepts that a learner must already know to understand this question.
   Think of dependencies in the knowledge domain, not just vocabulary. If none exist, use [].
6. 'successors' should list concepts that are natural next steps after mastering this question's concept.
   If none exist, use [].
7. 'methods' should describe the cognitive operation required (e.g., 'membaca tabel', 'menghitung selisih',
   'membandingkan dua konsep'). If the question only requires recall, you may use [].
8. Do not include any text, comments, or whitespace outside the JSON object.
9. Do not wrap the JSON in markdown code fences (no ```json).
10. Ensure all lists are valid JSON arrays, even if empty ([]).
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AutoHKG:
    """
    Main pipeline class for automated hierarchical knowledge graph construction.

    Instantiate with an HKGConfig object, then call .run() to process the dataset
    and export the resulting graph.

    The pipeline is designed to be resumable: progress is saved to a checkpoint file
    after each batch. If the run is interrupted, re-running with the same config will
    skip already-processed rows and continue from where it left off.

    Example:
        cfg = HKGConfig(provider="groq", data_path="data/questions.csv", batch_size=10)
        pipeline = AutoHKG(cfg)
        pipeline.run()
    """

    def __init__(self, config: HKGConfig):
        self.cfg = config
        self.client = LLMClient(
            provider=config.provider,
            model=config.model
        )
        self.builder = GraphBuilder()
        self.checkpoint = self._load_checkpoint()
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path("output/logs").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Checkpoint management
    # -------------------------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        """
        Load the checkpoint file if it exists.
        Returns a dict with 'processed' (int) and 'failed_ids' (list).
        If no checkpoint exists, returns a fresh state starting at row 0.
        """
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

    # -------------------------------------------------------------------------
    # Row identity
    # -------------------------------------------------------------------------

    @staticmethod
    def _row_id(row: pd.Series) -> str:
        """
        Generate a short deterministic ID for a dataset row based on its question
        text and Bloom level. Used to track failed extractions in the checkpoint.
        """
        raw = f"{row['Pertanyaan']}|{row['Level Kognitif']}"
        return hashlib.md5(raw.encode()).hexdigest()[:10]

    # -------------------------------------------------------------------------
    # Single-row extraction
    # -------------------------------------------------------------------------

    def _extract_one(self, row: pd.Series) -> Optional[dict]:
        """
        Run LLM extraction for a single dataset row with retry logic.

        The context is truncated to 1500 characters to prevent exceeding the
        model's effective prompt length while retaining the most relevant content.
        On each attempt, the raw LLM output is parsed as JSON and validated against
        the Auto-HKG schema. If all retries fail, returns None and logs the failure.
        """
        prompt = EXTRACTION_PROMPT.format(
            context=row["Konteks"][:1500],
            question=row["Pertanyaan"],
            bloom_level=row["Level Kognitif"]
        )

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                raw = self.client.complete(prompt)

                # FIX: Guard None — model kadang mengembalikan None
                # (bukan string kosong), menyebabkan 'NoneType' subscriptable
                # saat .strip() dipanggil. Raise agar masuk blok retry.
                if raw is None:
                    raise ValueError("LLM returned None instead of a string.")

                # Strip markdown code fences if the model includes them despite instructions
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:]
                    cleaned = cleaned.strip()

                # FIX: Beberapa model (Gemma, Qwen) menambahkan teks preamble
                # di luar JSON (e.g. "Here is the result:\n{...}").
                # Cari kurung kurawal pertama/terakhir untuk ekstrak JSON murni.
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

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    def run(self):
        """
        Execute the full pipeline over the dataset.

        Processing order:
            1. Load CSV and validate column names.
            2. Skip already-processed rows using the checkpoint offset.
            3. For each row, call _extract_one and pass results to GraphBuilder.
            4. Save checkpoint after each batch.
            5. After all rows are processed, export graph outputs.
        """
        df = pd.read_csv(self.cfg.data_path, sep=";")

        # Normalize column names regardless of leading/trailing whitespace in CSV
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

    # -------------------------------------------------------------------------
    # Output export
    # -------------------------------------------------------------------------

    def _save_outputs(self):
        """
        Export the constructed knowledge graph in three formats:

        1. knowledge_graph.json : Full graph in NetworkX node-link format.
                                  Suitable for programmatic loading and downstream
                                  CG-IR retrieval and reasoning tasks.

        2. nodes.csv / edges.csv: Flat tabular exports for manual inspection,
                                  visualization tools (Gephi, Cytoscape), or
                                  spreadsheet review.

        3. stats.json           : Summary statistics including node counts by type
                                  and total edge count.
        """
        out = Path(self.cfg.output_dir)
        G = self.builder.get_graph()

        # 1. JSON (node-link format)
        graph_data = nx.node_link_data(G)
        with open(out / "knowledge_graph.json", "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        # 2. Tabular exports
        nodes_df = pd.DataFrame([{"id": n, **G.nodes[n]} for n in G.nodes])
        edges_df = pd.DataFrame([
            {"source": u, "target": v, **d} for u, v, d in G.edges(data=True)
        ])
        nodes_df.to_csv(out / "nodes.csv", index=False)
        edges_df.to_csv(out / "edges.csv", index=False)

        # 3. Summary stats
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
        data_path=args.data,
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep,
    )

    pipeline = AutoHKG(cfg)
    pipeline.run()
