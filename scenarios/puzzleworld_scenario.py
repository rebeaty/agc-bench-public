"""
HELM Scenario: PuzzleWorld

Paper: https://arxiv.org/abs/2506.06211
       "PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning
        in Puzzlehunts"
Code: https://github.com/MIT-MI/PuzzleWorld
Dataset: https://huggingface.co/datasets/hzli1202/PuzzleWorld

PuzzleWorld evaluates open-ended, multimodal reasoning on 667 real-world
puzzlehunt problems from Puzzled Pint (2010-2025). Each puzzle combines
text, visual, and structured inputs with no explicit instructions. Models
must infer hidden problem structure and execute multi-step creative
reasoning to arrive at a short canonical answer.

667 puzzles total (easy: 140, medium: 355, hard: 172)
Modalities: text, visual, structured
Skills: logic, spatial, cryptic, wordplay, commonsense, knowledge

Evaluation: local HELM port scores the extracted trailing `Answer:` field
  for final-answer accuracy. The paper's stepwise metric is not yet
  implemented locally.

Prompt format: Two-part prompt from the paper's evaluation code.
  - System prompt: PUZZLE_SYSTEM_PROMPT from src/modeling.py —
    puzzle-solving guidance with tips on acrostics, indexing,
    alpha-numeric codes, anagrams, etc.
  - User prompt: PUZZLE_USER_PROMPT from src/reasoner.py —
    puzzle-specific title/flavor text with step-by-step instructions.

Fields used: title, flavor_text, solution, content_file_names, difficulty,
  modality, skills
Fields skipped: reasoning (human-annotated traces, for analysis only),
  source (URL to original PDF)

Note: Content images hosted on HuggingFace dataset repo. Downloaded via
      hf_hub_download. Single "train" split contains all 667 puzzles.
"""

import os
import sys
import json
from huggingface_hub import hf_hub_download
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject


class PuzzleWorldScenario(Scenario):
    name = "puzzleworld"
    description = "hzli1202/PuzzleWorld"
    tags = ["creativity", "multimodal", "vision", "lateral_thinking", "puzzles"]

    HF_REPO = "hzli1202/PuzzleWorld"

    # From src/modeling.py PUZZLE_SYSTEM_PROMPT
    SYSTEM_PROMPT = (
        "You will be presented with a puzzle to solve. The puzzle may not have specific instructions,\n"
        "but you know that the answer to the puzzle is a word or short phrase (or rarely, a number).\n"
        "\n"
        "Do not ask any questions about how to proceed, just do your best to solve the puzzle.\n"
        "Here are some tips for solving puzzles of this type:\n"
        "\n"
        "General Tips:\n"
        "- Puzzles will often have multiple steps to get to the answer word. You can usually tell you\n"
        "are on the right track if the intermediate answers agree with the title, flavor, or theme\n"
        "of the puzzle.\n"
        "- You can usually find hints in the introductory text. For example references to \"in the dark\"\n"
        "or \"sight\" are often hints something is encoded with braille.\n"
        "- Puzzles often incorporate acrostics: a clue where the first letter, syllable, or word of\n"
        "each line, paragraph, or other recurring feature spells out a word or message.\n"
        "- If you end up with a garbled \"alphabet soup\", then look for a clue on how to order them.\n"
        "- Indexing is one of the most common puzzle mechanisms. Try indexing when you have a list of\n"
        "words or phrases and a corresponding list of numbers. Count into the word or phrase by the\n"
        "given number and record the letter in that position. For example: \"2 Cake, 6 Pudding, 5\n"
        "Shortening\" gives you \"ant\".\n"
        "- Alpha-numeric codes are also very common. If you end up with a list of numbers try replacing\n"
        "the numbers with the corresponding letters like this: 1 = A, 2 = B, 3 = C... 26 = Z.\n"
        "Occasionally, these types of codes will \"wrap around\", so don't despair if you see a\n"
        "number greater than 26. Just subtract 26 and try again. In this scenario 27 (27-26 = 1) =\n"
        "A, 28 (28-26 = 2) = B etc. If you try this and it doesn't work, try other numeric codes\n"
        "such as ASCII.\n"
        "- Often a puzzle repeats a strategy multiple times.\n"
        "\n"
        "You will likely need to backtrack frequently, so make sure to write out your steps as you go.\n"
        "If you get stuck, try to think of a new way to approach the puzzle. Try:\n"
        "- Rereading the title and the flavor text. These are the most important hints about what type\n"
        "of strategies, themes or cultural references might be used to solve the puzzle.\n"
        "- Checking for pop culture references\n"
        "- Checking for references to a song/poem/book/movie/TV show\n"
        "\n"
        "For strings, examples of strategies you might try include:\n"
        "- Alphabetizing\n"
        "- Using leftover letters to spell something\n"
        "- Rearranging the letters (aka anagrams or \"transposing\")\n"
        "- Seeing if there are any acronyms\n"
        "- Diagonalizing (taking the first letter of the first answer, the second letter of the second\n"
        "answer, etc.)\n"
        "- Looking for unusual letter frequencies\n"
        "- Puns and homophones\n"
        "- Shifting from letters to numbers\n"
        "\n"
        "For numbers, try:\n"
        "- Shifting from numbers to letters\n"
        "- Using it as a phone number\n"
        "- Treating numbers as dates\n"
        "- Treating numbers as ASCII numbers\n"
        "- Seeing if there are any strange sequences\n"
        "- Seeing if prime numbers are involved\n"
        "\n"
        "For images, try:\n"
        "- Looking at it in a mirror\n"
        "- Squinting at it from far away\n"
        "- Tilting it\n"
        "- Looking at it upside down\n"
        "- Looking through it\n"
        "- Transcribing it neatly"
    )

    def __init__(self, difficulty: str = "all"):
        """
        Args:
            difficulty: Filter by difficulty - "easy", "medium", "hard", or "all"
        """
        super().__init__()
        if difficulty not in ("easy", "medium", "hard", "all"):
            raise ValueError(f"difficulty must be easy/medium/hard/all, got '{difficulty}'")
        self.difficulty = difficulty
        limit_hint = os.environ.get("PUZZLEWORLD_INSTANCE_LIMIT_HINT", "").strip()
        self.instance_limit_hint = int(limit_hint) if limit_hint.isdigit() and int(limit_hint) > 0 else None

    def _download_content(self, file_name: str) -> str:
        """Download a puzzle content file from HuggingFace."""
        return hf_hub_download(
            self.HF_REPO,
            file_name,
            repo_type="dataset",
        )

    def _load_metadata(self):
        """Load PuzzleWorld metadata directly from the released JSONL file."""
        metadata_path = hf_hub_download(
            self.HF_REPO,
            "all_metadata.jsonl",
            repo_type="dataset",
        )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def get_instances(self, output_path: str):
        dataset = self._load_metadata()

        instances = []
        for item in dataset:
            if self.difficulty != "all" and item["difficulty"] != self.difficulty:
                continue

            # Build multimedia content: puzzle image(s) + flavor text
            media_objects = []
            download_failures = []

            # Add puzzle content image(s)
            for file_name in item["content_file_names"]:
                try:
                    image_path = self._download_content(file_name)
                    media_objects.append(
                        MediaObject(
                            content_type="image/png",
                            location=image_path,
                        )
                    )
                except Exception as exc:
                    download_failures.append((file_name, str(exc)))

            if download_failures:
                print(
                    f"PuzzleWorldScenario: skipping '{item['title']}' due to content download failure(s): "
                    + ", ".join(file_name for file_name, _ in download_failures),
                    file=sys.stderr,
                )
                continue

            # System prompt (PUZZLE_SYSTEM_PROMPT from src/modeling.py)
            # + User prompt (PUZZLE_USER_PROMPT from src/reasoner.py)
            prompt_text = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"Your task is to solve the following puzzle. The attached images are presented in the order\n"
                f"they are referenced in the text.\n\n"
                f"The puzzle's title is: {item['title']}\n"
                f"The puzzle's flavor text is: {item['flavor_text']}\n\n"
                f"---\n"
                f"Write out a step-by-step solution to the puzzle. You must end your response with one\n"
                f"final line in exactly the following format, with no extra text after it:\n"
                f"Answer: <answer>"
            )

            media_objects.append(
                MediaObject(
                    content_type="text/plain",
                    text=prompt_text,
                )
            )

            multimedia_content = MultimediaObject(media_objects)

            references = [
                Reference(
                    Output(text=item["solution"]),
                    tags=[CORRECT_TAG],
                )
            ]

            title_id = item["title"].replace(" ", "_")

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT,
                id=f"puzzleworld_{title_id}",
                extra_data={
                    "title": item["title"],
                    "difficulty": item["difficulty"],
                    "modality": item["modality"],
                    "skills": item["skills"],
                    "content_file_count": len(item["content_file_names"]),
                },
            ))

            if self.instance_limit_hint is not None and len(instances) >= self.instance_limit_hint:
                break

        return instances
