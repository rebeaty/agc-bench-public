"""
HELM Scenario: RPGBENCH — Game Creation Task

Paper: RPGBENCH: Evaluating Large Language Models as Role-Playing Game Engines
       arXiv:2502.00595 (2025)
Code:  https://github.com/boson-ai/rpgbench-public

Task: Given a Wikipedia biography of a character, generate a playable RPG game
scenario structured as a JSON object. The character becomes the main NPC.

This scenario implements the Game Creation task only.
Game Simulation (multi-turn interactive gameplay) is skipped — incompatible
with HELM's single-turn Instance architecture.

Prompt format (verbatim from rpgbench/static/game_creation_prompt.txt):
  Here is a character description:
  {name}: {description}

  Based on this character, create a detailed game scenario exactly following
  the JSON structure below:
  [game_json_schema]

  [guidelines from game_creation_prompt.txt]

Data: data/game_creation/characters.jsonl (100 Wikipedia character biographies)
      Loaded directly from GitHub raw URL (no HuggingFace dataset)
Fields used: id, name, description
Fields skipped: none

Evaluation: llm_judge (interestingness, 1-5 scale)
            Source: rpgbench/static/game_interestingness_prompt.txt
            See annotator_notes.md for judge configuration.
            Paper also uses structural validity check (JSON schema compliance),
            which can be computed programmatically without an LLM.

Note: Game Simulation task skipped. It requires multi-turn gameplay loops
      where the model alternately acts as game engine and player — fundamentally
      incompatible with HELM's single-turn Instance → References architecture.
"""

import json
import urllib.request
from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Reference,
    Scenario,
)

_CHARACTERS_URL = (
    "https://raw.githubusercontent.com/boson-ai/rpgbench-public/main"
    "/data/game_creation/characters.jsonl"
)

# Condensed JSON schema template (from rpgbench/static/game_json_schema.json)
# Included verbatim in the prompt so models know the expected output structure.
_GAME_SCHEMA = """{
  "game_world": "<string: world setting and atmosphere>",
  "player_name": "<string: player character name>",
  "player_description": "<string: player character description>",
  "main_npc_name": "<string: name of the main NPC (from the biography)>",
  "main_npc_description": {
    "text": "<string: NPC description>",
    "big5_personality_traits": {
      "openness": {"rate": <0-100>, "description": "<string>"},
      "conscientiousness": {"rate": <0-100>, "description": "<string>"},
      "extraversion": {"rate": <0-100>, "description": "<string>"},
      "agreeableness": {"rate": <0-100>, "description": "<string>"},
      "neuroticism": {"rate": <0-100>, "description": "<string>"}
    },
    "additional_facts": ["<string>", "..."]
  },
  "game_objectives": "<string: player objectives and win conditions>",
  "scenes": [
    {
      "scene_name": "<string>",
      "unique_id": "S001",
      "background_description": "<string>",
      "scene_type": "<string>"
    }
  ],
  "state_variables": [
    {
      "value_name": "<string>",
      "unique_id": "V001",
      "description": "<string>",
      "initial_value": 50,
      "min_value": 0,
      "max_value": 100
    }
  ],
  "hidden_variables": [
    {"value_name": "has_succeeded", "unique_id": "H001", "description": "<string>", "initial_value": 0, "min_value": 0, "max_value": 1},
    {"value_name": "has_failed",    "unique_id": "H002", "description": "<string>", "initial_value": 0, "min_value": 0, "max_value": 1}
  ],
  "events": [
    {
      "event_name": "<string>",
      "unique_id": "E001",
      "scene": "S001",
      "entering_condition": "<string: condition to trigger this event>",
      "succeed_condition": "<string: condition for success>",
      "succeed_effect": "<string: variable changes on success>",
      "fail_effect": "<string: variable changes on failure>"
    }
  ],
  "pre_event_checks": [
    {
      "check_name": "<string>",
      "unique_id": "P001",
      "description": "<string>",
      "condition": "<string: condition expression>",
      "effect": "<string: variable changes when triggered>"
    }
  ]
}"""

# Guidelines verbatim from rpgbench/static/game_creation_prompt.txt
_GUIDELINES = """## Guidelines
- All numerical values should use consistent ranges (e.g., 0-100)
- Events should have clear cause-and-effect relationships
- Scene progression should depend on variable thresholds
- Include both mandatory and optional events
- Create meaningful connections between variables
- Balance difficulty and achievability
- Ensure all IDs follow consistent formatting (P### for checks, S### for scenes, V### for state variables, H### for hidden variables, E### for events)
- Include proper fail states and success conditions
- Make sure all scenes are specific locations
- Create logical progression paths through the game
- Return only the JSON object itself, with no Markdown code fences or extra commentary

Format the response as a single JSON object with all fields properly nested. Must ensure all arrays and objects are properly closed and formatted."""


class RpgBenchScenario(Scenario):
    """
    RPGBENCH Game Creation: generate a structured RPG game from a character biography.

    100 Wikipedia character biographies (e.g., Mickey Mouse, Superman) serve as
    inputs. The model generates a complete RPG game scenario as a JSON object,
    with the character as the main NPC. Evaluated on interestingness (1-5 LLM
    judge) and structural validity (JSON schema compliance).
    """

    name = "rpgbench"
    description = (
        "github.com/boson-ai/rpgbench-public (arXiv:2502.00595) — "
        "Game Creation: Wikipedia character biography → RPG game JSON"
    )
    tags = ["creativity", "rpg", "game_design", "structured_generation", "open_ended_generation"]

    def get_instances(self, output_path: str) -> List[Instance]:
        with urllib.request.urlopen(_CHARACTERS_URL) as response:
            raw = response.read().decode("utf-8").strip()

        instances = []
        for line in raw.split("\n"):
            if not line.strip():
                continue
            character = json.loads(line)
            name = character["name"]
            description = character["description"]

            prompt = (
                f"Here is a character description:\n"
                f"{name}: {description}\n\n"
                f"Based on this character, create a detailed game scenario exactly "
                f"following the JSON structure below:\n"
                f"{_GAME_SCHEMA}\n\n"
                f"{_GUIDELINES}"
            )

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=[],  # Open-ended generation; evaluated by LLM judge
                    split=TEST_SPLIT,
                )
            )

        return instances  # 100 instances
