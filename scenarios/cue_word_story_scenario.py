"""
HELM Scenario: Cue-word-based Creative Story Generation

Paper: Evaluating Creative Short Story Generation in Humans and Large Language Models
       https://arxiv.org/abs/2411.02316
       ICCC 2025

Dataset: https://huggingface.co/datasets/mismayil/creative_story_generation_dataset
License: Not specified (academic research)
GitHub: https://github.com/mismayil/creative-story-gen

Task: Creative Story Generation with Cue Words
Generate a creative short story (maximum 5 sentences) that includes three given cue words.

Cue word sets (from paper):
- Low semantic distance: "stamp, letter, send" and "petrol, diesel, pump"
- High semantic distance: "gloom, payment, exist" and "organ, empire, comply"

Prompt format (from repository src/prompts.py):
  System: You are an expert creative story writer. You will be given three words
  (e.g., car, wheel, drive) and then asked to write a creative short story that
  contains these three words. The idea is that instead of writing a standard story
  such as "I went for a drive in my car with my hands on the steering wheel.", you
  come up with a novel and unique story that uses the required words in unconventional
  ways or settings.

  User: Write a creative short story using a maximum of five sentences. The story must
  include the following three words: {cue_words}. However, the story should not be
  about {boring_theme}.

Dataset contains 479 pre-generated stories (236 human, 243 AI) with expert and non-expert
ratings on creativity, originality, surprise, and value. HELM adapts this broader
human-vs-model study into four generation prompts, one per cue-word set, and uses
human-written stories for those sets as references plus lightweight automatic
diagnostics over the generated slice.

Fields used: item_id (cue words), story (for human-authored stories as references)
Fields skipped: author (metadata), expert_*/non_expert_* (ratings for specific stories
in dataset, not applicable to new generations)

Evaluation: open-ended generation with overlap proxies as secondary signals,
plus a four-axis LLM judge aligned to the paper's creativity / originality /
surprise / effectiveness dimensions and a benchmark-specific automatic metric
for cue-word coverage, sentence-limit compliance, n-gram diversity, novelty,
and surprise proxies (see metric_notes.md)
"""

from typing import List
from collections import defaultdict
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)
from datasets import load_dataset


class CueWordStoryScenario(Scenario):
    """
    Cue-word-based Creative Story Generation

    Evaluates models' ability to generate creative short stories incorporating
    specific cue words with varying semantic distances.
    """

    name = "cue_word_story"
    description = "mismayil/creative_story_generation_dataset"
    tags = ["creativity", "story_generation", "creative_writing"]

    # Exact prompts from repository (src/prompts.py)
    SYSTEM_INSTRUCTION = (
        "You are an expert creative story writer. You will be given three words "
        "(e.g., car, wheel, drive) and then asked to write a creative short story that "
        "contains these three words. The idea is that instead of writing a standard story "
        "such as \"I went for a drive in my car with my hands on the steering wheel.\", you "
        "come up with a novel and unique story that uses the required words in unconventional "
        "ways or settings."
    )

    USER_INSTRUCTION_TEMPLATE = (
        "Write a creative short story using a maximum of five sentences. "
        "The story must include the following three words: {items}. "
        "However, the story should not be about {boring_theme}."
    )

    def __init__(self, semantic_distance: str = "all"):
        """
        Args:
            semantic_distance: "all", "low", or "high"
                - "all": All 4 cue word sets (default)
                - "low": Low semantic distance sets (stamp-letter-send, petrol-diesel-pump)
                - "high": High semantic distance sets (gloom-payment-exist, organ-empire-comply)
        """
        super().__init__()
        self.semantic_distance = semantic_distance.lower()

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load cue-word story generation dataset and create instances.

        Each instance contains:
        - Prompt: Instruction to write creative story with 3 cue words
        - References: Human-written stories for those cue words
        - Extra data: cue words and boring-theme hint for automatic diagnostics
        """
        # Load dataset from HuggingFace
        # Keep the Hugging Face cache under the run output path so the scenario
        # does not reuse stale global dataset metadata from older `datasets`
        # releases.
        dataset = load_dataset(
            "mismayil/creative_story_generation_dataset",
            split="train",
            cache_dir=output_path,
        )

        # Cue word sets with semantic distance classification and boring themes
        # From repository data/pilot_data.json
        cue_word_sets = {
            "stamp-letter-send": {
                "words": ["stamp", "letter", "send"],
                "distance": "low",
                "boring_theme": "putting a stamp on the envelop containing a letter to send it"
            },
            "petrol-diesel-pump": {
                "words": ["petrol", "diesel", "pump"],
                "distance": "low",
                "boring_theme": "going to the petrol station to pump diesel into a vehicle"
            },
            "gloom-payment-exist": {
                "words": ["gloom", "payment", "exist"],
                "distance": "high",
                "boring_theme": "the feeling of gloom you have about an existing payment"
            },
            "organ-empire-comply": {
                "words": ["organ", "empire", "comply"],
                "distance": "high",
                "boring_theme": "having an organ empire and complying with regulations"
            },
        }

        # Filter cue word sets based on semantic_distance parameter
        if self.semantic_distance == "low":
            selected_sets = {k: v for k, v in cue_word_sets.items() if v["distance"] == "low"}
        elif self.semantic_distance == "high":
            selected_sets = {k: v for k, v in cue_word_sets.items() if v["distance"] == "high"}
        else:  # "all"
            selected_sets = cue_word_sets

        # Group stories by item_id and filter for human-authored stories
        human_stories_by_item = defaultdict(list)
        for example in dataset:
            item_id = example["item_id"]
            # Only use human-authored stories as references
            if example["author"].lower() == "human":
                human_stories_by_item[item_id].append(example["story"])

        instances = []
        for item_id, cue_info in selected_sets.items():
            cue_words = cue_info["words"]
            cue_words_str = ", ".join(cue_words)
            boring_theme = cue_info["boring_theme"]

            # Build prompt based on repository prompts (src/prompts.py)
            # Combine system and user instructions as single prompt for HELM
            prompt = (
                f"{self.SYSTEM_INSTRUCTION}\n\n"
                f"{self.USER_INSTRUCTION_TEMPLATE.format(items=cue_words_str, boring_theme=boring_theme)}"
            )

            # Get human-written reference stories
            reference_stories = human_stories_by_item.get(item_id, [])

            # Create references from human stories
            references = [
                Reference(Output(text=story), tags=[CORRECT_TAG])
                for story in reference_stories
            ]

            # Create instance
            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"cue_word_story_{item_id}",
                    extra_data={
                        "item_id": item_id,
                        "cue_words": cue_words,
                        "boring_theme": boring_theme,
                        "semantic_distance": cue_info["distance"],
                    },
                )
            )

        return instances
