"""
HELM Scenario: DataNarrative

Paper: DataNarrative: Automated Data-Driven Storytelling with Visualizations and Texts
       Mohammed Saidul Islam, Md Tahmid Rahman Laskar, Md Rizwan Parvez, Enamul Hoque, Shafiq Joty
       EMNLP 2024
       https://arxiv.org/abs/2408.05346
       https://aclanthology.org/2024.emnlp-main.1073/

Code/Data: https://github.com/saidul-islam98/DataNarrative

Task: Generate narrative text describing data insights from tabular data.
      Given a data table and a topic/intent, models must produce one coherent
      narrative paragraph for a single paragraph-table segment that accurately
      describes trends, patterns, and key statistics.

Multi-Stage Framework (paper describes iterative refinement):
  1. Reflection generation (Figure 18) - analyze data tables
  2. Reflection revision (Figures 19-20) - check factual accuracy
  3. Outline generation (Figure 21) - create story structure
  4. Outline revision (Figures 22-23) - verify theme consistency
  5. Narration generation (Figure 24) - write final story
  6. Narration revision (Figures 25-26) - final refinement

For HELM evaluation, we use simplified single-turn generation:

Prompt format (simplified from Figure 24):
  Generate a narrative paragraph describing insights from the following data table.
  Focus on key trends, patterns, and significant data points.

  Topic: {topic_name}
  Intent: {intent}

  Data Table:
  {table}

  Narrative:

Fields used: table, paragraph (reference), topic_name, intent
Fields available but not used:
  - chart (image filename - multimodal extension possible)
  - vis_spec (visualization JSON - not needed for text-only task)
  - annotation (data point highlights)
  - reflection, outline, narration_int (intermediate outputs from multi-agent framework)

Evaluation:
  - Local primary: text-only judge dimensions for theme alignment,
    clarity/coherence, informativeness, narrative quality, and factual
    correctness
  - Local secondary: overlap diagnostics (`bleu_4`, `bert_score`)
  - Paper evaluation: pairwise model-based and human evaluation on
    Informativeness, Clarity and Coherence, Visualization Quality, Narrative
    Quality, and Factual Correctness

Dataset:
  - Paper corpus: 1,449 stories from Tableau, Pew Research, and GapMinder
  - Official JSON paragraph-table records: 1,917
  - Local HELM adaptation: 1,914 usable text-only paragraph-table segments
    * Three Tableau test records are skipped because the released JSON leaves
      their table-file reference blank.
    * Train: 226 segments (Tableau)
    * Test: 1,688 segments
    * GapMinder: 42 (demographic/economic trends)
    * Pew: 1,590 (social/political topics)
    * Tableau: 56 (various domains)

Note: Each story may have multiple paragraph-table pairs. We create one instance
      per paragraph-table segment for fine-grained text-only evaluation.
"""

import json
import os
from pathlib import Path
from typing import List
from urllib.parse import quote

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    CORRECT_TAG,
    TRAIN_SPLIT,
    TEST_SPLIT,
)
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class DataNarrativeScenario(Scenario):
    """
    DataNarrative: Automated data-driven storytelling benchmark.

    Evaluates LLM ability to generate narrative text from tabular data,
    focusing on factual accuracy and coherent explanation of data insights.
    """

    name = "data_narrative"
    description = "saidul-islam98/DataNarrative"
    tags = ["creativity", "data_storytelling", "text_generation", "data_to_text"]

    # GitHub raw content URLs
    GITHUB_BASE = "https://raw.githubusercontent.com/saidul-islam98/DataNarrative/main"

    TRAIN_FILES = {
        "tableau": "Train/Tableau/tableau_train.json"
    }

    TEST_FILES = {
        "gapminder": "Test/GapMinder/gapminder_test.json",
        "pew": "Test/Pew/pew_test.json",
        "tableau": "Test/Tableau/tableau_test.json"
    }

    def __init__(self, source: str = "all"):
        """
        Args:
            source: Which test source to use. Options: ["gapminder", "pew", "tableau", "all"]
                   "all" = All test sources (1,691 examples)
                   "gapminder" = GapMinder only (42 examples)
                   "pew" = Pew Research only (1,590 examples)
                   "tableau" = Tableau only (59 examples)
        """
        super().__init__()
        if source not in ["gapminder", "pew", "tableau", "all"]:
            raise ValueError(f"Invalid source: {source}. Must be 'gapminder', 'pew', 'tableau', or 'all'")
        self.source = source

    def _build_prompt(self, topic_name: str, intent: str, table: str, segment_number: int, total_segments: int) -> str:
        """
        Build simplified prompt for narrative generation.
        Based on Figure 24 but adapted for single-turn generation.
        """
        return (
            f"You are writing one paragraph for a single segment of a larger data story.\n"
            f"Generate exactly one coherent narrative paragraph grounded in the table below.\n"
            f"Focus on key trends, patterns, and significant data points for this segment only.\n"
            f"Do not write headings, bullet lists, or a full multi-section story.\n\n"
            f"Topic: {topic_name}\n"
            f"Intent: {intent}\n\n"
            f"Segment: paragraph {segment_number} of {total_segments}\n\n"
            f"Data Table:\n{table}\n\n"
            f"Narrative:"
        )

    def _download_json(self, url: str, local_path: str) -> dict:
        """Download and parse JSON file from GitHub."""
        ensure_file_downloaded(source_url=url, target_path=local_path)
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _download_text(self, url: str, local_path: str) -> str:
        """Download and return a raw text asset from GitHub."""
        ensure_file_downloaded(source_url=url, target_path=local_path)
        with open(local_path, "r", encoding="utf-8") as f:
            return f.read()

    def _normalize_table_text(self, table_value) -> str:
        """Render upstream table payloads into stable prompt text."""
        if isinstance(table_value, str):
            return table_value.strip()
        if isinstance(table_value, list):
            if not table_value:
                return ""
            if all(isinstance(row, dict) for row in table_value):
                columns = list(table_value[0].keys())
                lines = ["\t".join(columns)]
                for row in table_value:
                    lines.append("\t".join(str(row.get(column, "")) for column in columns))
                return "\n".join(lines).strip()
            return "\n".join(str(item) for item in table_value).strip()
        if isinstance(table_value, dict):
            return json.dumps(table_value, ensure_ascii=True, sort_keys=True, indent=2)
        if table_value is None:
            return ""
        return str(table_value).strip()

    def _normalize_filename_for_match(self, filename: str) -> str:
        return "".join(character.lower() for character in filename if character.isalnum())

    def _list_test_tableau_files(self, output_root: str, article_id: str) -> List[str]:
        """Read the GitHub contents listing for a Tableau test story directory."""
        url = f"https://api.github.com/repos/saidul-islam98/DataNarrative/contents/Test/Tableau/{article_id}"
        local_path = os.path.join(output_root, "tableau_test_tables", article_id, "_contents.json")
        payload = self._download_json(url, local_path)
        if isinstance(payload, list):
            return [item.get("name", "") for item in payload if isinstance(item, dict)]
        return []

    def _resolve_test_tableau_table(self, output_root: str, article_id: str, table_filename) -> str:
        """Fetch Tableau test CSV files referenced indirectly from the JSON metadata."""
        filenames = table_filename if isinstance(table_filename, list) else [table_filename]
        table_chunks: List[str] = []

        for raw_filename in filenames:
            if not isinstance(raw_filename, str):
                continue

            clean_filename = raw_filename.strip().strip('"')
            if not clean_filename:
                continue

            candidate_filenames = [clean_filename]
            available_files = self._list_test_tableau_files(output_root, article_id)
            normalized_target = self._normalize_filename_for_match(clean_filename)
            for available_filename in available_files:
                if self._normalize_filename_for_match(available_filename) == normalized_target:
                    candidate_filenames.append(available_filename)

            for candidate_filename in candidate_filenames:
                encoded_filename = quote(candidate_filename, safe="/")
                url = f"{self.GITHUB_BASE}/Test/Tableau/{article_id}/{encoded_filename}"
                local_path = os.path.join(output_root, "tableau_test_tables", article_id, Path(candidate_filename).name)
                try:
                    table_text = self._download_text(url, local_path).strip()
                    if table_text:
                        table_chunks.append(f"[{Path(candidate_filename).name}]\n{table_text}")
                        break
                except Exception:
                    continue

        return "\n\n".join(table_chunks).strip()

    def _extract_segment_fields(self, segment: dict, story_id: str, source_name: str, output_root: str, data: dict) -> tuple[str, str]:
        """Handle the three upstream paragraph-table segment schemas."""
        if "content" in segment and isinstance(segment["content"], dict):
            content = segment["content"]
            return (
                content.get("paragraph", "").strip(),
                self._normalize_table_text(content.get("table")),
            )

        if "paragraph" in segment and "table" in segment:
            return (
                segment.get("paragraph", "").strip(),
                self._normalize_table_text(segment.get("table")),
            )

        paragraph_key = next((key for key in segment if key.startswith("paragraph_")), None)
        table_key = next(
            (
                key
                for key in segment
                if key.startswith("table_") and not key.endswith("_title")
            ),
            None,
        )
        if paragraph_key and table_key:
            article_id = data.get("article_id", {}).get(story_id, "")
            table_reference = segment.get(table_key, "")
            return (
                segment.get(paragraph_key, "").strip().strip('"'),
                self._resolve_test_tableau_table(output_root, article_id, table_reference),
            )

        raise ValueError(f"Unsupported DataNarrative segment schema for {source_name}/{story_id}: {list(segment.keys())}")

    def _extract_instances(self, data: dict, split: str, source_name: str) -> List[Instance]:
        """Extract instances from a DataNarrative JSON file."""
        instances = []
        output_root = os.path.join(self.output_path, "data")

        # Get metadata
        topic_names = data.get("topic_name", {})
        intents = data.get("intent", {})
        paragraph_table_pairs = data.get("paragraph_table_pair", {})

        # Iterate through all story IDs
        for story_id in paragraph_table_pairs.keys():
            topic = topic_names.get(story_id, "Unknown Topic")
            intent = intents.get(story_id, "")

            # Each story has multiple paragraph-table segments
            segments = paragraph_table_pairs[story_id]

            for seg_idx, segment in enumerate(segments):
                paragraph, table = self._extract_segment_fields(
                    segment=segment,
                    story_id=story_id,
                    source_name=source_name,
                    output_root=output_root,
                    data=data,
                )

                # Skip if missing critical fields
                if not paragraph or not table:
                    continue

                # Build prompt
                prompt = self._build_prompt(
                    topic_name=topic,
                    intent=intent,
                    table=table,
                    segment_number=seg_idx + 1,
                    total_segments=len(segments),
                )

                # Create instance
                instance_id = f"{source_name}_{story_id}_{seg_idx}"

                instances.append(Instance(
                    input=Input(text=prompt),
                    references=[Reference(Output(text=paragraph), tags=[CORRECT_TAG])],
                    split=split,
                    id=instance_id
                ))

        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """Load DataNarrative dataset and create HELM instances."""
        self.output_path = output_path
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)

        instances: List[Instance] = []

        # Load training data
        for source_name, file_path in self.TRAIN_FILES.items():
            url = f"{self.GITHUB_BASE}/{file_path}"
            local_path = os.path.join(data_path, f"train_{source_name}.json")

            data = self._download_json(url, local_path)
            instances.extend(self._extract_instances(data, TRAIN_SPLIT, f"train_{source_name}"))

        # Load test data based on source parameter
        test_files = {}
        if self.source == "all":
            test_files = self.TEST_FILES
        else:
            test_files = {self.source: self.TEST_FILES[self.source]}

        for source_name, file_path in test_files.items():
            url = f"{self.GITHUB_BASE}/{file_path}"
            local_path = os.path.join(data_path, f"test_{source_name}.json")

            data = self._download_json(url, local_path)
            instances.extend(self._extract_instances(data, TEST_SPLIT, f"test_{source_name}"))

        return instances
