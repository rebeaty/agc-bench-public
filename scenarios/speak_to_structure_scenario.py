"""
HELM Scenario: Speak-to-Structure (S2-Bench / TOMG-Bench)
          — Open-Domain Natural Language-Driven Molecule Generation

Paper: "Speak-to-Structure: Evaluating LLMs in Open-domain
        Natural Language-Driven Molecule Generation"
       (arXiv:2412.14642)
Official repo: https://github.com/phenixace/S2-TOMG-Bench
Data: phenixace/S2-TOMG-Bench on Hugging Face

Task: Given a natural language instruction, generate a SMILES string
representing a molecule that satisfies the specified constraints
(property-based generation) or modifications (molecule editing /
optimization). Tests open-ended molecular design from natural language.

Three main tasks, 10 subtasks total:
  MolCustom — Generate a novel molecule satisfying given properties:
    AtomNum
    BasicProp
    BondNum
    FunctionalGroup

  MolEdit — Edit an existing molecule by adding/removing/substituting groups:
    AddComponent
    DelComponent
    SubComponent

  MolOpt — Optimize a molecule to improve a target property:
    LogP
    MR
    QED

Prompt format: source-backed chemist system head from the official S2-Bench
dataset loader plus the benchmark `Instruction` field.

Evaluation: RDKit-based validity and subtask-aware success/similarity.
See `metrics/speak_to_structure_metric.py` and metric notes.
"""

from typing import List

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Scenario,
)

_HF_BENCHMARK_REPO = "phenixace/S2-TOMG-Bench"
_HF_BENCHMARK_REPO_MINI = "phenixace/S2-TOMG-Bench-mini"

_SUBTASKS = {
    "MolCustom": ["AtomNum", "BondNum", "FunctionalGroup"],
    "MolEdit": ["AddComponent", "DelComponent", "SubComponent"],
    "MolOpt": ["LogP", "MR", "QED"],
}

_VALID_TASKS = list(_SUBTASKS.keys()) + ["all"]

_BENCHMARK_CONFIG_MAP = {
    ("MolCustom", "AtomNum"): "MolCustom_AtomNum",
    ("MolCustom", "BondNum"): "MolCustom_BondNum",
    ("MolCustom", "FunctionalGroup"): "MolCustom_FunctionalGroup",
    ("MolEdit", "AddComponent"): "MolEdit_AddComponent",
    ("MolEdit", "DelComponent"): "MolEdit_DelComponent",
    ("MolEdit", "SubComponent"): "MolEdit_SubComponent",
    ("MolOpt", "LogP"): "MolOpt_LogP",
    ("MolOpt", "MR"): "MolOpt_MR",
    ("MolOpt", "QED"): "MolOpt_QED",
}

_SYSTEM_HEAD = (
    "You are working as an assistant of a chemist user. Please follow the instruction of the chemist "
    "and generate a molecule that satisfies the requirements of the chemist user. "
    "Return exactly one final answer line in the format 'Molecule: [SMILES STRING]'. "
    "Do not include explanation, analysis, markdown, or any text before or after that line."
)


def _resolve_benchmark_repo(benchmark_scale: str) -> str:
    if benchmark_scale == "mini":
        return _HF_BENCHMARK_REPO_MINI
    return _HF_BENCHMARK_REPO


def _get_benchmark_config_name(task: str, subtask: str) -> str:
    return _BENCHMARK_CONFIG_MAP.get((task, subtask), f"{task}_{subtask}")


class SpeakToStructureScenario(Scenario):
    """
    Speak-to-Structure (S2-Bench / TOMG-Bench) — molecule generation from
    natural language instructions.

    The official benchmark is distributed as one Hugging Face config per
    subtask. We preserve task/subtask metadata in `extra_data` so the evaluator
    can route to the corresponding chemistry checks.
    """

    name = "speak_to_structure"
    description = "huggingface.co/datasets/phenixace/S2-TOMG-Bench (arXiv:2412.14642)"
    tags = [
        "creativity",
        "scientific_creativity",
        "molecule_generation",
        "chemistry",
        "open_ended_generation",
    ]

    def __init__(
        self,
        task: str = "all",
        subtask: str = "all",
        max_instances_per_subtask: int = 500,
        benchmark_scale: str = "full",
    ):
        super().__init__()
        if task not in _VALID_TASKS:
            raise ValueError(f"Unknown task: {task!r}. Must be one of {_VALID_TASKS}")
        if benchmark_scale not in ("full", "mini"):
            raise ValueError("benchmark_scale must be 'full' or 'mini'")
        self.task = task
        self.subtask = subtask
        self.max_instances_per_subtask = max_instances_per_subtask
        self.benchmark_scale = benchmark_scale

    def get_instances(self, output_path: str) -> List[Instance]:
        benchmark_repo = _resolve_benchmark_repo(self.benchmark_scale)
        active_tasks = list(_SUBTASKS.keys()) if self.task == "all" else [self.task]

        instances: List[Instance] = []
        for task in active_tasks:
            subtasks = _SUBTASKS[task] if self.subtask == "all" else [self.subtask]
            for subtask in subtasks:
                dataset = load_dataset(
                    benchmark_repo,
                    _get_benchmark_config_name(task, subtask),
                    split="test",
                    cache_dir=output_path,
                )
                for row_index, row in enumerate(dataset.select(range(min(self.max_instances_per_subtask, len(dataset))))):
                    instruction = str(row["Instruction"]).strip()
                    prompt = f"{_SYSTEM_HEAD}\n\n{instruction}"
                    row_dict = dict(row)

                    instances.append(
                        Instance(
                            input=Input(text=prompt),
                            references=[],
                            split=TEST_SPLIT,
                            id=f"s2bench_{task}_{subtask}_{row_index}",
                            extra_data={
                                "task": task,
                                "subtask": subtask,
                                "row": row_dict,
                                "benchmark_scale": self.benchmark_scale,
                            },
                        )
                    )

        return instances
