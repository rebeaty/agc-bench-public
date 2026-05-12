"""
HELM Scenario: CPers (Creativity in Persian)

Paper: Evaluating the Creativity of LLMs in Persian Literary Text Generation
       https://aclanthology.org/2025.findings-emnlp.796/

This scenario follows the paper's model-evaluation protocol more closely than
the earlier dataset-sweep version: generate texts over five selected topics with
repeated zero-shot sampling, rather than iterating over all 4,371 human texts.
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    TEST_SPLIT,
)


PAPER_TOPICS = [
    "عشق",        # love
    "دلتنگی",     # longing
    "رفاقت",      # friendship
    "امیدواری",   # hope
    "ناامیدی",    # despair
]


class CPersScenario(Scenario):
    """CPers five-topic repeated-generation setup."""

    name = "cpers"
    description = "teias-ai/CPers paper-five-topic generation protocol"
    tags = ["creativity", "generation", "multilingual", "persian"]

    PROMPT_TEMPLATE_FA = "درباره {topic} یک متن ادبی در یک جمله بنویس"

    def __init__(self, num_trials: int = 100):
        super().__init__()
        self.num_trials = max(1, int(num_trials))

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for topic_index, topic in enumerate(PAPER_TOPICS):
            prompt = self.PROMPT_TEMPLATE_FA.format(topic=topic)
            for trial_index in range(self.num_trials):
                instances.append(
                    Instance(
                        id=f"cpers_{topic_index}_trial{trial_index}",
                        input=Input(text=prompt),
                        references=[],
                        split=TEST_SPLIT,
                        extra_data={
                            "topic": topic,
                            "trial_index": trial_index,
                            "topic_set": "paper_five",
                        },
                    )
                )
        return instances
