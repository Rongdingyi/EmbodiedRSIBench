"""P0: RSI update transaction safety — a failing hook rolls back and never half-writes."""
import hashlib
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests"))

from benchmark.agent.config import AgentConfig  # noqa: E402
from benchmark.agent.episode_runner import run_episode  # noqa: E402
from benchmark.agent.react_agent import CanonicalMultimodalReActAgent  # noqa: E402
from benchmark.rsi.context import make_injection  # noqa: E402
from canonical_fakes import FakeAdapter, FakeBackend, make_protocol, make_task, valid_response  # noqa: E402


class TransactionSpy:
    """Minimal RSIMethod-shaped spy with real file state and failure injection."""

    name = "tx_spy"

    def __init__(self, state_root: Path, *, fail_on_step: int | None = None,
                 fail_after_episode: bool = False):
        self.state_dir = Path(state_root)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "applied.txt").write_text("")
        self.fail_on_step = fail_on_step
        self.fail_after_episode = fail_after_episode
        self.after_episode_called = False
        self.rollbacks = 0
        self.update_enabled = True
        self.run_ctx: dict = {}

    # ---- state/hash
    def _hash(self) -> str:
        digest = hashlib.sha256()
        for file in sorted(self.state_dir.rglob("*")):
            if file.is_file():
                digest.update(file.read_bytes())
        return digest.hexdigest()[:24]

    def state_hash(self) -> str:
        return self._hash()

    def snapshot(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.state_dir, output_dir, dirs_exist_ok=True)

    def load_snapshot(self, input_dir: Path) -> None:
        self.rollbacks += 1
        shutil.rmtree(self.state_dir)
        shutil.copytree(Path(input_dir), self.state_dir)

    # ---- hooks
    def before_episode(self, task_public_view):
        return make_injection("")

    def predict_sidecar(self, *, public_observation, action, accountant=None):
        return None

    def after_step(self, record: dict) -> None:
        with open(self.state_dir / "applied.txt", "a") as fh:
            fh.write(f"step {record['action']['name']}\n")
        if self.fail_on_step is not None and \
                record["action"]["name"] == valid_action_name(self.fail_on_step):
            raise RuntimeError("simulated after_step failure")

    def after_episode(self, trajectory_public, outcome_public) -> None:
        self.after_episode_called = True
        with open(self.state_dir / "applied.txt", "a") as fh:
            fh.write("episode\n")
        if self.fail_after_episode:
            raise RuntimeError("simulated after_episode failure")

    def get_last_update_usage(self):
        return {"prompt_tokens": 1, "completion_tokens": 1, "calls": 1}

    def get_last_update_wall_s(self):
        return 0.0

    def advance_step(self, index, total):
        return None


def valid_action_name(step_index: int) -> str:
    return "MoveAhead" if step_index == 1 else "RotateLeft"


def _specs():
    return [
        {"name": "MoveAhead", "description": "m",
         "parameters": {"type": "object", "properties": {},
                        "required": [], "additionalProperties": False}},
        {"name": "RotateLeft", "description": "r",
         "parameters": {"type": "object", "properties": {},
                        "required": [], "additionalProperties": False}},
    ]


def _run(tmp_path, spy):
    adapter = FakeAdapter(tool_specs=_specs(), outcomes=[
        dict(terminated=False, feedback="one", action_success=True),
        dict(terminated=False, feedback="two", action_success=True),
        dict(terminated=True, feedback="three", action_success=True, success=True),
    ])
    backend = FakeBackend([valid_response("MoveAhead"), valid_response("RotateLeft"),
                           valid_response("MoveAhead")])
    agent = CanonicalMultimodalReActAgent(backend=backend, config=AgentConfig())
    return run_episode(make_task(), adapter, spy, role="experience",
                       config=make_protocol(), output_dir=tmp_path / "ep", agent=agent)


def test_after_step_failure_rolls_back_and_skips_after_episode(tmp_path):
    spy = TransactionSpy(tmp_path / "state", fail_on_step=2)
    res = _run(tmp_path, spy)

    assert res.stop_reason == "rsi_update_error"
    assert res.status == "FAIL"
    assert "after_step" in (res.rsi_update_error or "")
    assert spy.rollbacks == 1
    assert spy.after_episode_called is False
    # the pre-episode snapshot won: no half-written step lines survive
    assert (spy.state_dir / "applied.txt").read_text() == ""


def test_after_episode_failure_rolls_back(tmp_path):
    spy = TransactionSpy(tmp_path / "state", fail_after_episode=True)
    res = _run(tmp_path, spy)

    assert res.stop_reason == "rsi_update_error"
    assert res.status == "FAIL"
    assert spy.after_episode_called is True     # it was attempted once
    assert spy.rollbacks == 1
    assert (spy.state_dir / "applied.txt").read_text() == ""


def test_successful_episode_commits_state(tmp_path):
    spy = TransactionSpy(tmp_path / "state")
    res = _run(tmp_path, spy)

    assert res.status == "PASS"
    assert spy.rollbacks == 0
    assert spy.after_episode_called is True
    assert "episode" in (spy.state_dir / "applied.txt").read_text()
