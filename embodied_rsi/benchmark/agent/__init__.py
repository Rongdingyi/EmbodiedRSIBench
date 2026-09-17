"""Canonical Multimodal ReAct Planner-Executor Agent (protocol v0.8).

The agent owns: current observation consumption, episode-local working memory,
reasoning/planning, exactly-one-action selection, JSON/action validation,
replanning and the fixed recovery prompts. It never executes environment
actions and never claims benchmark success.
"""
from benchmark.agent.config import AgentConfig, load_agent_config  # noqa: F401
from benchmark.agent.react_agent import CanonicalMultimodalReActAgent  # noqa: F401
from benchmark.agent.types import (  # noqa: F401
    AgentDecision,
    AgentEpisodeState,
    CanonicalEpisodeResult,
    WorkingMemoryStep,
)
