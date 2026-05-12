"""HELM Scenario: EQBench Creative Writing v3.

Multi-prompt creative-writing rubric released by Sam Paech (EQ-bench). The
benchmark resolves seed modifiers per iteration and scores generations with
an LLM-judge rubric drawn from the project landing page
(https://eqbench.com/creative_writing.html).
"""
from typing import List
import json
import os
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class EQBenchCreativeWritingV3Scenario(Scenario):
    """EQBench Creative Writing v3 rubric proxy with resolved seed modifiers.

    Upstream generation also uses `min_p=0.1`, which the local HELM adapter does not
    currently expose. This scenario still resolves `<SEED>` per iteration so the
    prompt text matches the intended benchmark surface more closely.
    """

    name = "eqbench_creative_writing_v3"
    description = "EQ-bench/creative-writing-bench"
    tags = ["creativity", "creative_writing", "long_form", "diverse_genres"]

    DATASET_DOWNLOAD_URL = "https://raw.githubusercontent.com/EQ-bench/creative-writing-bench/main/data/creative_writing_prompts_v3.json"

    def __init__(self, num_iterations: int = 3):
        """
        Args:
            num_iterations: Number of times to generate for each prompt (default 3 per benchmark)
        """
        super().__init__()
        self.num_iterations = num_iterations

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load EQBench Creative Writing v3 prompts and create instances.

        Each of the 32 prompts is repeated num_iterations times (default 3)
        to create 96 total instances. Generation uses temperature=0.7 and min_p=0.1
        to encourage creativity while maintaining coherence.
        """

        # Download dataset
        data_path = os.path.join(output_path, "creative_writing_prompts_v3.json")
        ensure_file_downloaded(
            source_url=self.DATASET_DOWNLOAD_URL,
            target_path=data_path,
            unpack=False,
        )

        # Load JSON data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instances = []

        # Data is dict with numbered string keys ('1', '2', etc.)
        for prompt_id in sorted(data.keys(), key=lambda x: int(x)):
            prompt_data = data[prompt_id]

            category = prompt_data['category']
            title = prompt_data['title']
            writing_prompt = prompt_data['writing_prompt']
            seed_modifiers = prompt_data.get('seed_modifiers', [])

            # Create num_iterations instances for each prompt
            for iteration in range(self.num_iterations):
                seed_modifier = seed_modifiers[iteration % len(seed_modifiers)] if seed_modifiers else ""
                prompt_text = writing_prompt.replace("<SEED>", seed_modifier).strip()

                instance_id = f"eqbench_cw_v3_{prompt_id}_iter{iteration+1}"

                instances.append(
                    Instance(
                        input=Input(text=prompt_text),
                        references=[],
                        split=TEST_SPLIT,
                        id=instance_id,
                        extra_data={
                            "prompt_id": prompt_id,
                            "iteration": iteration + 1,
                            "category": category,
                            "title": title,
                            "seed_modifier": seed_modifier,
                        },
                    )
                )

        return instances
