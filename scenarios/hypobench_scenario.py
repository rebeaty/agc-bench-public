"""
HELM Scenario: HypoBench (real-world zero-shot generation slice)

Paper: HypoBench: Towards Systematic and Principled Benchmarking for Hypothesis Generation
       Liu et al., 2025. arXiv:2504.11524
Website: https://chicagohai.github.io/HypoBench/
Code:    https://github.com/ChicagoHAI/hypothesis-generation
Data:    https://github.com/ChicagoHAI/HypoBench-datasets

This HELM scenario implements the paper's real-world task family in the benchmark shape
that fits single-model evaluation cleanly:
  1. Generate zero-shot hypotheses for each task from the released task description.
  2. Score those hypotheses with the benchmark's held-out hypothesis-based inference
     pipeline on IND and OOD splits.

The synthetic HDR slice remains out of scope for this runnable scenario and is documented
explicitly in metric notes instead of being silently approximated.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError

import yaml

from helm.benchmark.scenarios.scenario import Instance, Input, Output, Reference, Scenario, TEST_SPLIT
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


_DATASET_BASE_URL = "https://raw.githubusercontent.com/ChicagoHAI/HypoBench-datasets/main/real"

VALID_TASKS = [
    "deceptive_reviews",
    "headline_binary",
    "gptgc_detect",
    "llamagc_detect",
    "dreaddit",
    "persuasive_pairs",
    "retweet",
]


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _columnar_to_rows(payload: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(payload.keys())
    if not keys:
        return []
    row_count = len(payload[keys[0]])
    return [{key: payload[key][index] for key in keys} for index in range(row_count)]


def _join_prompt(system_prompt: str, user_prompt: str) -> str:
    system_prompt = (system_prompt or "").strip()
    user_prompt = (user_prompt or "").strip()
    if system_prompt and user_prompt:
        return f"{system_prompt}\n\n{user_prompt}"
    return system_prompt or user_prompt


class HypoBenchScenario(Scenario):
    name = "hypobench"
    description = "https://github.com/ChicagoHAI/HypoBench-datasets"
    tags = ["creativity", "hypothesis_generation", "scientific_reasoning"]

    def __init__(self, task: str = "all_real", num_hypotheses: int = 10):
        """
        Args:
            task: "all_real" or one of the 7 released real-world tasks.
            num_hypotheses: Number of hypotheses requested in the zero-shot prompt.
        """
        super().__init__()
        if task != "all_real" and task not in VALID_TASKS:
            raise ValueError(f"task must be 'all_real' or one of {VALID_TASKS}, got '{task}'")
        self.task = task
        self.num_hypotheses = num_hypotheses

    def _tasks(self) -> Iterable[str]:
        if self.task == "all_real":
            return VALID_TASKS
        return [self.task]

    def _download_task_files(self, output_path: str, task: str) -> Dict[str, str]:
        task_dir = Path(output_path) / "hypobench" / task
        ensure_directory_exists(str(task_dir))

        filenames = [
            "config.yaml",
            "metadata.json",
        ]

        task_base = f"{_DATASET_BASE_URL}/{task}"
        for filename in filenames:
            ensure_file_downloaded(
                source_url=f"{task_base}/{filename}",
                target_path=str(task_dir / filename),
            )

        config = _read_yaml(str(task_dir / "config.yaml"))
        data_keys = ["train_data_path", "test_data_path", "val_data_path", "ood_data_path"]
        for key in data_keys:
            rel_path = str(config.get(key, "")).lstrip("./")
            if not rel_path:
                continue
            target = task_dir / rel_path
            ensure_directory_exists(str(target.parent))
            source_url = f"{task_base}/{rel_path}"
            first_segment = rel_path.split("/", 1)[0]
            if "/" in rel_path and first_segment in VALID_TASKS:
                source_url = f"{_DATASET_BASE_URL}/{rel_path}"
            try:
                ensure_file_downloaded(
                    source_url=source_url,
                    target_path=str(target),
                )
            except HTTPError as exc:
                if exc.code == 404:
                    continue
                raise

        return {
            "task_dir": str(task_dir),
            "config_path": str(task_dir / "config.yaml"),
            "metadata_path": str(task_dir / "metadata.json"),
        }

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []

        for task in self._tasks():
            local_paths = self._download_task_files(output_path, task)
            config = _read_yaml(local_paths["config_path"])
            metadata = _read_json(local_paths["metadata_path"])
            prompt_templates = config.get("prompt_templates", {})

            zero_shot_template = prompt_templates.get("initialize_zero_shot", {})
            if not zero_shot_template:
                raise ValueError(f"HypoBench task '{task}' is missing initialize_zero_shot prompts")

            substitute = {"num_hypotheses": str(self.num_hypotheses)}
            system_prompt = Template(str(zero_shot_template.get("system", ""))).safe_substitute(substitute)
            user_prompt = Template(str(zero_shot_template.get("user", ""))).safe_substitute(substitute)
            full_prompt = _join_prompt(system_prompt, user_prompt)

            config_dir = Path(local_paths["config_path"]).parent
            test_rel = str(config.get("test_data_path", "")).lstrip("./")
            ood_rel = str(config.get("ood_data_path", "")).lstrip("./")

            extra_data = {
                "task": task,
                "task_name": str(metadata.get("task_name") or task),
                "task_description": str(metadata.get("task_description") or ""),
                "label_values": list((metadata.get("labels") or {}).get("label", {}).get("values", [])),
                "features": list((metadata.get("features") or {}).keys()),
                "config_path": local_paths["config_path"],
                "metadata_path": local_paths["metadata_path"],
                "test_data_path": str(config_dir / test_rel) if test_rel else "",
                "ood_data_path": str(config_dir / ood_rel) if ood_rel else "",
                "multiple_hypotheses_inference": prompt_templates.get("multiple_hypotheses_inference"),
                "inference": prompt_templates.get("inference"),
                "known_hypotheses": list(metadata.get("known_hypotheses", [])),
                "num_hypotheses_requested": self.num_hypotheses,
            }

            references: List[Reference] = []
            for hypothesis in list(metadata.get("known_hypotheses", []))[:3]:
                references.append(Reference(Output(text=str(hypothesis)), tags=[]))

            instances.append(
                Instance(
                    input=Input(text=full_prompt),
                    references=references,
                    split=TEST_SPLIT,
                    extra_data=extra_data,
                )
            )

        return instances
