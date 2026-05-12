"""
HELM Scenario: Only Connect Wall (OCW) Dataset

Paper: Large Language Models are Fixated by Red Herrings: Exploring Creative Problem
       Solving and Einstellung Effect using the Only Connect Wall Dataset
       https://arxiv.org/abs/2306.11167
       NeurIPS 2023

Code: https://github.com/TaatiTeam/OCW
Dataset: TaatiTeam/OCW_main (HuggingFace)

Task: Given 16 shuffled words (clues), group them into 4 groups of 4 words based on
      thematic connections. The puzzles deliberately include "red herrings" - misleading
      connections that create fixation effects and make creative problem-solving challenging.

This implements Task 1 (Grouping) from the paper - the primary creative problem-solving task.
Task 2 (Connections) involves naming the connections after groups are solved, which is a
separate auxiliary task.

Prompt format: From notebooks/run_openai.ipynb in the OCW repository
  System message explains the task and warns about red herrings
  User provides the list of 16 clues
  Model outputs groups as newline-separated lists

Format:
  Input: "Clues: word1, word2, word3, ..., word16"
  Output: Four groups (one per line), words separated by commas

Fields used: wall_id, words, groups (for ground truth)
Fields skipped: season, episode, gt_connections (used in Task 2), human_performance (metadata)

Evaluation: Custom metric needed - must check if predicted groups match ground truth groups
            regardless of group order or word order within groups. See metric_notes.md.
"""

from datasets import load_dataset
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALID_SPLIT,
)


class OnlyConnectWallScenario(Scenario):
    """
    Only Connect Wall (OCW) Dataset - Creative Problem Solving with Red Herrings

    From the quiz show "Only Connect", where contestants must group 16 clues into
    4 groups of 4 based on thematic connections. The puzzles include deliberate
    red herrings that create misleading connections.

    618 puzzles total: 62 train, 62 validation, 494 test
    """

    name = "only_connect_wall"
    description = "TaatiTeam/OCW_main"
    tags = ["creativity", "reasoning", "problem_solving", "word_association"]

    def get_instances(self, output_path: str) -> list[Instance]:
        # Load all splits
        dataset = load_dataset("TaatiTeam/OCW_main")

        instances = []

        for split_name, split_data in [
            ("train", dataset["train"]),
            ("validation", dataset["validation"]),
            ("test", dataset["test"]),
        ]:
            # Map to HELM split names
            if split_name == "train":
                helm_split = TRAIN_SPLIT
            elif split_name == "validation":
                helm_split = VALID_SPLIT
            else:
                helm_split = TEST_SPLIT

            for item in split_data:
                # Extract the 16 words (already shuffled in the dataset)
                words = item["words"]

                # Build the prompt following the paper's baseline format
                # From notebooks/run_openai.ipynb
                prompt = (
                    "You are currently competing in Round 3: Connecting Wall on the quiz show Only Connect. "
                    "Your task: given 16 \"clues\" (words or phrases), solve the wall by grouping the clues "
                    "into four groups of four. Provide your answer as a list of four groups of four clues; "
                    "separate groups by newlines and clues by commas. Do not try to guess the connection; "
                    "only use the clues given and don't make up your own.\n\n"
                    "Be careful! Connecting Wall is deliberately difficult. The puzzles are designed to include "
                    "red herrings and to suggest more connections than actually exist. Some clues appear to fit "
                    "into more than one category. Still, there is only one perfect solution for each wall.\n\n"
                    f"Clues: {', '.join(words)}\n\n"
                    "Solved wall:"
                )

                # Extract ground truth groups
                # The groups field is a dict with keys group_1, group_2, group_3, group_4
                gt_groups = [
                    item["groups"][f"group_{i}"]["gt_words"]
                    for i in range(1, 5)
                ]

                # Create reference with ground truth groups
                # Format as the expected output: newline-separated groups with comma-separated words
                reference_text = "\n".join(
                    ", ".join(group) for group in gt_groups
                )

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=[Reference(output=Output(text=reference_text), tags=[])],
                        split=helm_split,
                    )
                )

        return instances
