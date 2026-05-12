"""
HELM Scenario: Only Connect Wall - Task 2 (Connections)

Paper: Large Language Models are Fixated by Red Herrings: Exploring Creative Problem
       Solving and Einstellung Effect using the Only Connect Wall Dataset
       https://arxiv.org/abs/2306.11167
       NeurIPS 2023

Code: https://github.com/TaatiTeam/OCW
Dataset: TaatiTeam/OCW_main (HuggingFace)

Task: Given 4 already-solved groups of 4 words each, determine the thematic connection
      that links each group together. This is Task 2 from the paper - an auxiliary task
      that tests the ability to articulate connections after groups are identified.

Prompt format: From notebooks/run_openai.ipynb in the OCW repository
  System message explains the task
  User provides the 4 groups and prompts for connections
  Model outputs connection names

Format:
  Input: "Group 1: word1, word2, word3, word4. Connection:\n
          Group 2: word5, word6, word7, word8. Connection:\n..."
  Output: "Group 1: word1, word2, word3, word4. Connection: [name]\n
           Group 2: word5, word6, word7, word8. Connection: [name]\n..."

Fields used: wall_id, groups (gt_words and gt_connection)
Fields skipped: words (used in Task 1), season, episode, human_performance (metadata)

Evaluation: Standard open-ended metrics - exact match, ROUGE-1 F1, BERTScore F1
            The paper evaluates connection naming using these metrics.
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
    CORRECT_TAG,
)


class OnlyConnectConnectionsScenario(Scenario):
    """
    Only Connect Wall - Task 2 (Connections)

    Given 4 already-solved groups of words, name the thematic connection for each group.
    This tests the ability to articulate connections rather than the creative problem-solving
    aspect tested in Task 1 (Grouping).

    618 puzzles total: 62 train, 62 validation, 494 test
    """

    name = "only_connect_connections"
    description = "TaatiTeam/OCW_main"
    tags = ["creativity", "reasoning", "word_association", "language_articulation"]

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
                # Extract the 4 groups with their words
                groups = [
                    item["groups"][f"group_{i}"]["gt_words"]
                    for i in range(1, 5)
                ]

                # Build the prompt following the paper's Task 2 baseline format
                # From notebooks/run_openai.ipynb
                system_msg = (
                    "You are currently competing in Round 3: Connecting Wall on the quiz show Only Connect. "
                    "Your task: given 4 groups of 4 \"clues\" (words or phrases), determine the connection for "
                    "each group. Provide your answer by repeating the four groups and adding it after \"Connection:\"\n\n"
                    "Note: Connections might be thematic, linguistic, factual, mathematical and rely on both "
                    "arcane subject areas and popular culture.\n\n"
                )

                # Format the groups
                groups_text = "\n".join(
                    f"Group {i+1}: {', '.join(group)}. Connection:"
                    for i, group in enumerate(groups)
                )

                prompt = system_msg + "Groups:\n" + groups_text + "\n\nSolved wall:"

                # Extract ground truth connections
                gt_connections = [
                    item["groups"][f"group_{i}"]["gt_connection"]
                    for i in range(1, 5)
                ]

                # Create references - one for each connection
                # The paper evaluates each connection separately
                # Format the output as the expected format with connection names filled in
                reference_text = "\n".join(
                    f"Group {i+1}: {', '.join(groups[i])}. Connection: {gt_connections[i]}"
                    for i in range(4)
                )

                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=[Reference(output=Output(text=reference_text), tags=[CORRECT_TAG])],
                        split=helm_split,
                    )
                )

        return instances
