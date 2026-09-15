"""Condition B: `raw_memory` -- episodic memory + deterministic BM25 retrieval.

Stores only public trajectory summaries; no LLM reflection, no skill
abstraction, no playbook editing (guide section 18).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from benchmark.openeta_bridge.context_injection import RSIInjection, count_tokens, make_injection
from benchmark.rsi.base import RSIMethod

TOP_K = 4
MAX_INJECTION_TOKENS = 8192
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class RawMemoryRSI(RSIMethod):
    name = "raw_memory"

    def __init__(self) -> None:
        super().__init__()
        self.episodes: list[dict] = []

    # ------------------------------------------------------------------ state
    def _episodes_file(self) -> Path:
        assert self.state_dir is not None
        return self.state_dir / "episodes.jsonl"

    def _load(self) -> None:
        path = self._episodes_file()
        self.episodes = []
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    self.episodes.append(json.loads(line))

    def _append(self, record: dict) -> None:
        self._episodes_file().parent.mkdir(parents=True, exist_ok=True)
        with open(self._episodes_file(), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------- retrieval
    def _bm25_scores(self, query: str) -> list[float]:
        docs = [tokenize(e.get("instruction") or "") for e in self.episodes]
        if not docs:
            return []
        query_tokens = tokenize(query)
        n = len(docs)
        avg_len = sum(len(d) for d in docs) / n
        df = Counter()
        for doc in docs:
            for token in set(doc):
                df[token] += 1
        k1, b = 1.5, 0.75
        scores = []
        for doc in docs:
            tf = Counter(doc)
            score = 0.0
            for token in query_tokens:
                if token not in tf:
                    continue
                idf = math.log(1 + (n - df[token] + 0.5) / (df[token] + 0.5))
                denom = tf[token] + k1 * (1 - b + b * len(doc) / max(avg_len, 1e-9))
                score += idf * tf[token] * (k1 + 1) / denom
            scores.append(score)
        return scores

    def _retrieve(self, instruction: str) -> list[dict]:
        scores = self._bm25_scores(instruction or "")
        ranked = sorted(range(len(self.episodes)),
                        key=lambda i: (-scores[i], i))
        return [self.episodes[i] for i in ranked[:TOP_K]]

    @staticmethod
    def _render(episodes: list[dict], max_actions: int) -> str:
        lines = ["Previous relevant experiences:", ""]
        for i, ep in enumerate(episodes, 1):
            lines.append(f"[Episode {i}]")
            lines.append(f"Task: {ep.get('instruction')}")
            lines.append(f"Outcome: {'Success' if ep.get('success') else 'Failure'}"
                         + f" (steps: {ep.get('num_steps')})")
            lines.append("Trajectory summary:")
            actions = ep.get("actions") or []
            shown = actions[-max_actions:] if max_actions else []
            for step in shown:
                feedback = (step.get("public_feedback") or "")[:160]
                lines.append(f"- {step.get('action')} -> {feedback}")
            lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------ hooks
    def init_run(self, run_ctx: dict) -> None:
        super().init_run(run_ctx)
        self._load()

    def before_episode(self, task_public_view: dict) -> RSIInjection:
        if not self.episodes:
            return make_injection("")
        retrieved = self._retrieve(task_public_view.get("instruction") or "")
        retrieved = [e for e in retrieved if e]
        if not retrieved:
            return make_injection("")
        # deterministic truncation: reduce per-episode action window until within budget
        for max_actions in (12, 8, 6, 4, 2, 1, 0):
            text = self._render(retrieved, max_actions)
            if count_tokens(text) <= MAX_INJECTION_TOKENS:
                injection = make_injection(text, provenance_ids=[e["episode_id"] for e in retrieved],
                                           truncated=max_actions < 12)
                return injection
        return make_injection(self._render(retrieved[:1], 0),
                              provenance_ids=[retrieved[0]["episode_id"]], truncated=True)

    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        if not self._guard_update():
            return
        record = {
            "episode_id": trajectory_public.get("episode_id"),
            "instruction": trajectory_public.get("instruction"),
            "source_dataset": trajectory_public.get("source_dataset"),
            "actions": [
                {"action": a.get("action"), "public_feedback": a.get("public_feedback")}
                for a in (trajectory_public.get("actions") or [])
            ],
            "success": bool(outcome_public.get("success")),
            "terminated": bool(outcome_public.get("terminated")),
            "num_steps": outcome_public.get("num_steps"),
        }
        self._append(record)
        self.episodes.append(record)

    # --------------------------------------------------------------- snapshot
    def snapshot(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.copy_tree(self._episodes_file().parent, output_dir)

    def load_snapshot(self, input_dir: Path) -> None:
        self.state_dir = Path(input_dir)
        self._load()

    def state_hash(self) -> str:
        assert self.state_dir is not None
        return self.hash_dir(self.state_dir)
