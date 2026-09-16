"""Condition C: ACE-Context (OpenETA executor adaptation).

Official ACE Reflector/Curator/playbook machinery from
`external/ace/ace/core/{reflector,curator}.py` + `playbook_utils.py`.
OpenETA replaces ACE's Generator as the task executor; no ACE prompt or update
logic is modified (guide sections 2.3 / 19).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from benchmark.openeta_bridge.context_injection import RSIInjection, count_tokens, make_injection
from benchmark.rsi.base import RSIMethod

PROJECT = Path(__file__).resolve().parents[2]
ACE_REPO = PROJECT / "external" / "ace"
if str(ACE_REPO) not in sys.path:
    sys.path.insert(0, str(ACE_REPO))

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
TOKEN_BUDGET = 8192
MAX_PLAYBOOK_TOKENS = 8192

class _AuditedChatClient:
    """Thin OpenAI-client proxy that records every ACE request (P1-10).

    Delegates verbatim to the wrapped client; only adds the redacted dump.
    """

    def __init__(self, inner, owner: "AceContextRSI") -> None:
        self._inner = inner
        self._owner = owner

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        accountant = (self._owner.run_ctx or {}).get("accountant")
        if accountant is not None:
            accountant.record_request_dump(
                url=f"{BASE_URL.rstrip('/')}/v1/chat/completions",
                body={
                    "model": kwargs.get("model"),
                    "messages": kwargs.get("messages") or [],
                    "max_tokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
                },
                role="ace_updater",
            )
        return self._inner.chat.completions.create(**kwargs)


def _usage_from_infos(infos: list) -> dict:
    """Sum tokens from ACE's timed_llm_call info dicts.

    ACE reports the official field names `prompt_num_tokens` /
    `response_num_tokens` (with a couple of legacy aliases).
    """
    prompt = completion = 0
    calls = 0
    for info in infos:
        if not isinstance(info, dict):
            continue
        usage = info
        if isinstance(info.get("usage"), dict):
            usage = {**info, **info["usage"]}
        prompt += int(
            usage.get("prompt_num_tokens")
            or usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        completion += int(
            usage.get("response_num_tokens")
            or usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        calls += 1
    return {"prompt_tokens": prompt, "completion_tokens": completion, "calls": calls}


EMPTY_PLAYBOOK = """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID
"""


class AceContextRSI(RSIMethod):
    name = "ace_context"

    def __init__(self) -> None:
        super().__init__()
        self.playbook = EMPTY_PLAYBOOK
        self.reports: list[dict] = []
        self._client = None

    # ------------------------------------------------------------------ setup
    def _openai_client(self):
        if self._client is None:
            from openai import OpenAI

            inner = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                base_url=f"{BASE_URL.rstrip('/')}/v1",
            )
            self._client = _AuditedChatClient(inner, self)
        return self._client

    def _paths(self) -> dict[str, Path]:
        assert self.state_dir is not None
        return {
            "playbook": self.state_dir / "playbook.txt",
            "ops": self.state_dir / "curator_ops.jsonl",
            "reports": self.state_dir / "reflector_output.jsonl",
            "log_dir": self.state_dir / "llm_logs",
        }

    def init_run(self, run_ctx: dict) -> None:
        super().init_run(run_ctx)
        paths = self._paths()
        if paths["playbook"].exists():
            self.playbook = paths["playbook"].read_text()
        else:
            paths["playbook"].write_text(self.playbook)
        paths["log_dir"].mkdir(parents=True, exist_ok=True)
        self._next_global_id = 1

    # ------------------------------------------------------------------ hooks
    def before_episode(self, task_public_view: dict) -> RSIInjection:
        text = self.playbook.strip()
        if not text or text == EMPTY_PLAYBOOK.strip():
            return make_injection("")
        if count_tokens(text) > MAX_PLAYBOOK_TOKENS:
            # deterministic relevance filter: keep the playbook headings only
            lines = [ln for ln in text.splitlines() if ln.startswith("##") or ln.startswith("-")]
            text = "\n".join(lines)
        return make_injection(text, provenance_ids=["ace_playbook"])

    def after_episode(self, trajectory_public: dict, outcome_public: dict) -> None:
        if not self._guard_update():
            return
        from ace.core.curator import Curator
        from ace.core.reflector import Reflector
        from playbook_utils import (
            apply_curator_operations,
            format_playbook_line,
            get_next_global_id,
            get_playbook_stats,
        )

        paths = self._paths()
        paths["log_dir"].mkdir(parents=True, exist_ok=True)
        question = trajectory_public.get("instruction") or ""
        trace_lines = [
            f"Step {i + 1}: {a.get('action')} -> {a.get('public_feedback')}"
            for i, a in enumerate(trajectory_public.get("actions") or [])
        ]
        reasoning_trace = "\n".join(trace_lines) or "(no actions)"
        success = bool(outcome_public.get("success"))
        feedback = f"task {'succeeded' if success else 'failed'} in {outcome_public.get('num_steps')} steps"

        client = self._openai_client()
        # provider id deliberately not "openai": ACE then sends `max_tokens`,
        # which is what the DeepSeek Chat Completions schema documents.
        reflector = Reflector(client, "deepseek", MODEL)
        curator = Curator(client, "deepseek", MODEL)

        bullets_used = self.playbook
        t0 = time.time()
        reflection, bullet_tags, reflect_info = reflector.reflect(
            question=question,
            reasoning_trace=reasoning_trace,
            predicted_answer=f"success={success}",
            ground_truth=None,
            environment_feedback=feedback,
            bullets_used=bullets_used,
            use_ground_truth=False,
            call_id=f"reflect_{trajectory_public.get('episode_id')}",
            log_dir=str(paths["log_dir"]),
        )
        stats = get_playbook_stats(self.playbook)
        next_id = get_next_global_id(self.playbook)
        new_playbook, next_id_after, operations, curate_info = curator.curate(
            current_playbook=self.playbook,
            recent_reflection=reflection,
            question_context=question,
            current_step=int(self.run_ctx.get("step", 0)),
            total_samples=int(self.run_ctx.get("total_steps", 0) or 0),
            token_budget=TOKEN_BUDGET,
            playbook_stats=stats,
            use_ground_truth=False,
            call_id=f"curate_{trajectory_public.get('episode_id')}",
            log_dir=str(paths["log_dir"]),
            next_global_id=next_id,
        )
        self._last_update_wall_s = time.time() - t0
        self._last_update_usage = _usage_from_infos([reflect_info, curate_info])
        updated = new_playbook
        if not updated or updated == self.playbook:
            try:
                updated = apply_curator_operations(self.playbook, operations, next_id)
            except Exception:
                updated = self.playbook
        self.playbook = updated
        paths["playbook"].write_text(self.playbook)

        with open(paths["ops"], "a") as f:
            f.write(json.dumps({
                "episode_id": trajectory_public.get("episode_id"),
                "operations": operations,
                "playbook_tokens": count_tokens(self.playbook),
                "curate_info": {k: v for k, v in (curate_info or {}).items()
                                if isinstance(v, (str, int, float, bool))},
            }, ensure_ascii=False) + "\n")
        with open(paths["reports"], "a") as f:
            f.write(json.dumps({
                "episode_id": trajectory_public.get("episode_id"),
                "reflection": reflection,
                "bullet_tags": bullet_tags,
            }, ensure_ascii=False) + "\n")

        self.reports.append({"episode_id": trajectory_public.get("episode_id"),
                             "ops": len(operations or [])})

    # --------------------------------------------------------------- snapshot
    def snapshot(self, output_dir: Path) -> None:
        self.copy_tree(self.state_dir, output_dir)

    def _reload_state(self) -> None:
        assert self.state_dir is not None
        pb = self.state_dir / "playbook.txt"
        self.playbook = pb.read_text() if pb.exists() else EMPTY_PLAYBOOK
        (self.state_dir / "llm_logs").mkdir(parents=True, exist_ok=True)

    def state_hash(self) -> str:
        assert self.state_dir is not None
        return self.hash_dir(self.state_dir)
