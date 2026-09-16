#!/usr/bin/env python
"""EmbodiSkill thin-adapter PoC (P1-8) -- isolated RSI worker.

Runs ONLY in an EmbodiSkill-capable environment (langchain-chroma +
sentence-transformers + finch; see external/EmbodiSkill/requirements.txt), so it
speaks the same newline-JSON worker protocol as the simulator workers instead of
importing those dependencies into the agent process.

Official API surface used (no upstream edits):
    EmbodiSkill.init_task_context(task_main, task_description)
    EmbodiSkill.move_skill_state(action, observation)
    EmbodiSkill.save_task_context(label, feedback)  -> MASMessage
    EmbodiSkill.reflect_episode(mas_message)        -> dict
    EmbodiSkill.revise_manual(epoch_id, success_rate) -> dict

Status: POC_READY_UNVERIFIED -- the converter is complete and <=200 lines, but it
has not been executed end-to-end yet (dependency environment not provisioned in
this session). Do not report C4 results before this PoC runs.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

PROJECT = Path("/data/users/rongdingyi/programs/benchmarks/EmbodiedRSIBench/embodied_rsi")
EMBODISKILL_REPO = PROJECT / "external" / "EmbodiSkill"
sys.path.insert(0, str(EMBODISKILL_REPO))


class EmbeddingFunction:
    """Minimal Chroma embedding adapter over sentence-transformers.

    Model is pinned; embeddings are an integration choice, not part of the
    reflection/update semantics under test (documented in the adapter config).
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.MODEL_NAME)

    def __call__(self, input):  # noqa: A002 - Chroma's calling convention
        texts = [input] if isinstance(input, str) else list(input)
        return [v.tolist() for v in self.model.encode(texts, normalize_embeddings=True)]

    def embed_documents(self, texts):
        return self(texts)

    def embed_query(self, text):
        return self(text)[0]


class DeepSeekLLM:
    """Official LLMCallable signature over the DeepSeek chat-completions API."""

    def __init__(self, model: str = "deepseek-flash") -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                             base_url="https://api.deepseek.com/v1")

    def __call__(self, messages, temperature: float = 0.0, max_tokens: int = 2048,
                 stop_strs=None, num_comps: int = 1) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def build_skill(state_root: Path):
    from agentkit.skill.embodiskill_skill.EmbodiSkill import EmbodiSkill

    # official dataclass fields: namespace / global_config / llm_model / embedding_func;
    # persist_dir is derived in __post_init__ from global_config (P1-11).
    return EmbodiSkill(
        namespace="embodiskill_openeta",
        global_config={
            "working_dir": str(state_root),
            "persist_dir": str(state_root / "skill_state"),
            "task_name": "embodied_openeta",
            "current_epoch_id": 0,
            "hop": 1,
            "project_name": "EmbodiSkill",
        },
        llm_model=DeepSeekLLM(),
        embedding_func=EmbeddingFunction(),
    )


def convert_episode(skill, trajectory_public: dict, outcome_public: dict) -> dict:
    """OpenETA public trajectory -> official StateChain -> official reflection."""
    instruction = trajectory_public.get("instruction") or ""
    skill.init_task_context(task_main=instruction, task_description=instruction)
    for index, item in enumerate(trajectory_public.get("actions") or []):
        action = str(item.get("action") or "")
        observation = f"step {index}: {item.get('public_feedback') or ''}"
        skill.move_skill_state(action=action, observation=observation)
    success = bool(outcome_public.get("success"))
    feedback = (f"task {'succeeded' if success else 'failed'} in "
                f"{outcome_public.get('num_steps')} steps")
    mas_message = skill.save_task_context(label=success, feedback=feedback)
    reflection = skill.reflect_episode(mas_message)
    revision = skill.revise_manual(epoch_id=0, success_rate=1.0 if success else 0.0)
    return {"reflection": reflection, "revision": revision,
            "manual_version": int(getattr(skill, "manual_state", {}).get("version", 0))}


def manual_guidance(skill) -> str:
    """Current official manual text used as the shared context injection."""
    manual = getattr(skill, "manual_state", None) or {}
    parts = []
    for section in manual.get("sections", []) or []:
        title = section.get("title") or section.get("name") or ""
        items = section.get("items") or []
        body = "\n".join(f"- {item}" for item in items if str(item).strip())
        if title or body:
            parts.append(f"## {title}\n{body}".strip())
    notes = manual.get("execution_notes") or []
    if notes:
        parts.append("## EXECUTION NOTES\n" + "\n".join(
            f"- {n.get('text') or n}" if isinstance(n, dict) else f"- {n}" for n in notes))
    return "\n\n".join(parts)


def serve() -> None:
    """JSON-line protocol: reset / before_episode / after_episode / snapshot / ping."""
    state_root = Path(os.environ.get("EMBODISKILL_STATE_ROOT", "/tmp/embodiskill_poc_state"))
    state_root.mkdir(parents=True, exist_ok=True)
    skill = build_skill(state_root)
    out = sys.stdout

    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line.decode())
        op = request.get("op")
        args = request.get("args") or {}
        try:
            if op == "ping":
                result = {"ok": True, "manual_version": skill.manual_state.get("version")}
            elif op == "before_episode":
                result = {"guidance": manual_guidance(skill)}
            elif op == "after_episode":
                result = convert_episode(skill, args["trajectory"], args["outcome"])
            elif op == "snapshot":
                result = {"state_hash": str(skill.manual_state.get("version", 0))}
            else:
                raise ValueError(f"unknown op {op!r}")
            response = {"id": request.get("id"), "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            response = {"id": request.get("id"), "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2000:]}
        out.write(json.dumps(response) + "\n")
        out.flush()


if __name__ == "__main__":
    serve()
