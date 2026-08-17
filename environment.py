"""Starter Freesolo multi-turn environment.

A multi-turn environment runs a bounded episode: the model produces an assistant
action, `step_episode` advances the world (optionally appending an observation
message), and the loop repeats until `done` or `max_episode_turns`. The finished
transcript is graded by `score_episode`.

Edit dataset/train.jsonl and the episode logic, then upload with
`flash env push --project 53ad8614-a108-44f9-bca8-bcf9dd726348 --name my-env .`.

A managed run should use the returned [environment] id from
`flash env push --project 53ad8614-a108-44f9-bca8-bcf9dd726348 --name my-env .`.

This starter implements a tiny "guess the secret number" game so you can see the
episode hooks wired end-to-end. Replace it with your real task before a real run.

All three algorithms train off this file:
- GRPO (configs/rl.toml) rolls out full episodes and optimizes `score_episode`.
- SFT (configs/sft.toml) learns the gold trajectory. Provide it per row as
  `output = {"messages": [...]}` (a full assistant/tool trajectory) or a scalar
  `output` for a single gold assistant turn.
- OPD (configs/opd.toml) rolls out each episode and distils EVERY assistant turn against
  the managed Parasail teacher (GLM 5.2 by default; pick another with [train] teacher_model),
  conditioned on the transcript so far — the multi-turn on-policy-distillation objective. The
  teacher key is platform-managed (nothing to set).
"""

from __future__ import annotations
import os as _flash_os, sys as _flash_sys
_flash_sys.path.insert(0, _flash_os.path.dirname(__file__))

import json
from pathlib import Path

from freesolo.datasets.types import TaskExample
from freesolo.environments import (
    EnvironmentEpisode,
    EnvironmentMultiTurn,
    EnvironmentStepResult,
    RewardResult,
)


DEFAULT_DATASET_PATH = Path(__file__).parent / "dataset" / "train.jsonl"

# How many assistant guesses the model gets before the episode is forced terminal.
MAX_TURNS = 5


def load_jsonl(path: str | Path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _secret(example: TaskExample) -> int:
    return int(str(example.output).strip())


class StarterMultiTurnEnv(EnvironmentMultiTurn):
    dataset = load_jsonl(DEFAULT_DATASET_PATH)

    def start_episode(self, example: TaskExample, prompt_text: str):
        # The opening prompt shown to the model. `example.input` describes the range.
        return [
            {
                "role": "user",
                "content": (
                    f"{example.input}\n"
                    f"Reply with a single integer per turn. I will say 'higher', "
                    f"'lower', or 'correct'. You have {MAX_TURNS} guesses."
                ),
            }
        ]

    def max_episode_turns(self, example: TaskExample) -> int:
        return MAX_TURNS

    def step_episode(
        self,
        example: TaskExample,
        messages: list,
        assistant_response: str,
    ) -> EnvironmentStepResult:
        # Advance the world after one assistant action. Return done=True to end the
        # episode, or append an observation message and keep going.
        # `messages` already ends with this action: messages[-1]["content"] is
        # `assistant_response`. if you rebuild state by replaying the transcript, replay
        # `messages[:-1]` and apply `assistant_response` once, or you apply it twice.
        try:
            guess = int(assistant_response.strip().split()[0])
        except (ValueError, IndexError):
            return EnvironmentStepResult(
                done=False,
                messages=[{"role": "user", "content": "Please reply with a single integer."}],
            )
        secret = _secret(example)
        if guess == secret:
            return EnvironmentStepResult(done=True, final_response_text=str(guess))
        hint = "higher" if guess < secret else "lower"
        return EnvironmentStepResult(
            done=False,
            messages=[{"role": "user", "content": hint}],
        )

    def score_episode(
        self,
        example: TaskExample,
        episode: EnvironmentEpisode,
    ) -> RewardResult:
        # Grade the finished transcript. Reward a correct final guess; give partial
        # credit for getting close so GRPO has a usable gradient.
        secret = _secret(example)
        try:
            final = int(str(episode.response_text).strip())
        except ValueError:
            return RewardResult(score=0.0, threshold=1.0)
        if final == secret:
            return RewardResult(score=1.0, threshold=1.0, success=True)
        # Closeness in [0, 1): further guesses score lower.
        closeness = max(0.0, 1.0 - abs(final - secret) / 100.0)
        return RewardResult(score=closeness * 0.5, threshold=1.0, success=False)


def load_environment(dataset_path: str | None = None, **kwargs) -> StarterMultiTurnEnv:
    env = StarterMultiTurnEnv()
    if dataset_path:
        env.dataset = load_jsonl(dataset_path)
    return env
