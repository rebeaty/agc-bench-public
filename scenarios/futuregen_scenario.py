"""
HELM Scenario: FutureGen

Paper: FutureGen: A RAG-based Approach to Generate the Future Work of Scientific Article
       https://arxiv.org/abs/2503.16561
Code: https://github.com/IbrahimAlAzhar/FutureWorkGeneration
Dataset: https://huggingface.co/datasets/iaadlab/FutureGen

Task: Generate future work sections for scientific papers based on paper
content. This local HELM port is explicitly scoped to the paper's non-RAG
prompt conditions rather than the full retrieval-and-judge pipeline.

Prompt formats (from code notebooks):

  Top-3 sections (from 6.(FutureGen)_GPT_3_(top_3_sections).ipynb):
    Uses `Abstract`, `Introduction`, and `Conclusion` only, with a concise
    <=100-word future-work prompt.

  All sections (from 5.(FutureGen)_GPT_3_(all_sections).ipynb):
    "You are an AI trained to analyze scientific research and suggest future
    directions based on the content of a paper. Below, you will find sections
    from a scientific article including the 'Abstract', 'Introduction',
    'Conclusion', 'Limitation', 'Experiment and Results', 'Related Work',
    'Methodology' of a scientific paper. Based on these details, please generate
    comprehensive and plausible future work suggestions that could extend the
    research findings, address limitations, and propose new avenues for
    exploration. Generate a future work based on these texts. Future work
    should be within 100 words."

Evaluation: Open-ended generation (ROUGE, BLEU, BERTScore, Jaccard, cosine)
with an optional judge path in the paper. This local port restores only the
automatic metric slice.
Ground truth: Combined author-mentioned future work + OpenReview peer suggestions

Dataset: Uses NeurIPS subset (278 papers from 2021-2022) with OpenReview feedback.
         ACL files excluded due to missing ground truth columns.

Fields used: top-3 or all-sections prompt text, future_work_combined (reference)
Fields preserved in metadata: Future_Work_extraction, LLM_extracted_review_future_work
"""

import json
import os
import pandas as pd
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    Scenario,
    Instance,
    Input,
    Output,
    Reference,
    TEST_SPLIT,
)
from helm.common.general import ensure_file_downloaded


class FuturegenScenario(Scenario):
    name = "futuregen"
    description = "iaadlab/FutureGen"
    tags = ["creativity", "scientific_writing", "future_work"]

    # Exact prompt from 6.(FutureGen)_GPT_3_(top_3_sections).ipynb
    PROMPT_TOP3 = """You are an AI trained to analyze scientific research and suggest future directions based on the content of a paper.
    Below, you will find sections from a scientific article including the 'Abstract', 'Introduction', 'Conclusion' of a scientific paper.
    Based on these details, please generate comprehensive and plausible future work suggestions that could extend the research findings,
    address limitations, and propose new avenues for exploration.
    Generate a future work based on these texts. Future work should be within 100 words.

{paper_text}"""

    # Exact prompt from 5.(FutureGen)_GPT_3_(all_sections).ipynb
    PROMPT_SHORT = """You are an AI trained to analyze scientific research and suggest future directions based on the content of a paper. Below, you will find sections from a scientific article including the 'Abstract', 'Introduction', 'Conclusion','Limitation','Experiment and Results','Related Work','Methodology' of a scientific paper. Based on these details, please generate comprehensive and plausible future work suggestions that could extend the research findings, address limitations, and propose new avenues for exploration. Generate a future work based on these texts. Future work should be within 100 words.

{paper_text}"""

    def __init__(self, prompt_style: str = "top3"):
        """
        Args:
            prompt_style: Which prompt to use. Options: ["top3", "all_sections"]
                         "top3" = paper-faithful non-RAG top-3-sections setup
                         "all_sections" = explicit all-sections ablation
        """
        super().__init__()
        if prompt_style not in ["top3", "all_sections"]:
            raise ValueError(
                f"Invalid prompt_style: {prompt_style}. Must be 'top3' or 'all_sections'"
            )
        self.prompt_style = prompt_style

    def get_instances(self, output_path: str) -> list[Instance]:
        # Download the NeurIPS dataset file
        data_url = "https://huggingface.co/datasets/iaadlab/FutureGen/resolve/main/df_neurips_future_work_dataset.csv"
        scenario_dir = os.path.join(output_path, self.name)
        os.makedirs(scenario_dir, exist_ok=True)
        data_path = os.path.join(scenario_dir, "df_neurips_future_work_dataset.csv")
        ensure_file_downloaded(
            source_url=data_url,
            target_path=data_path,
            unpack=False,
        )

        # Load the dataset
        df = pd.read_csv(data_path)

        instances = []
        for idx, row in df.iterrows():
            top3_text = (
                f"Abstract: {str(row.get('df_Abstract', ''))}\n"
                f"Introduction: {str(row.get('df_Introduction', ''))}\n"
                f"Conclusion: {str(row.get('df_Conclusion', ''))}\n"
            )
            all_sections_text = (
                f"Abstract: {str(row.get('df_Abstract', ''))}\n"
                f"Introduction: {str(row.get('df_Introduction', ''))}\n"
                f"Conclusion: {str(row.get('df_Conclusion', ''))}\n"
                f"Limitation: {str(row.get('df_Limitation', ''))}\n"
                f"Experiment_and_Results: {str(row.get('df_Experiment_and_Results', ''))}\n"
                f"Related_Work: {str(row.get('df_Related_Work', ''))}\n"
                f"Methodology: {str(row.get('df_Methodology', ''))}\n"
                f"Dataset: {str(row.get('df_Dataset', ''))}\n"
            )

            # Get the ground truth (combined future work)
            future_work = str(row["future_work_combined"])

            # Parse the future_work if it's a JSON string (it appears to be a list)
            try:
                future_work_parsed = json.loads(future_work)
                if isinstance(future_work_parsed, list):
                    # Join list elements into single text
                    future_work = "\n\n".join(future_work_parsed)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, use as-is
                pass

            author_future_work = str(row.get("Future_Work_extraction", ""))
            review_future_work = str(row.get("LLM_extracted_review_future_work", ""))

            # Build the prompt using selected style
            if self.prompt_style == "top3":
                prompt = self.PROMPT_TOP3.format(paper_text=top3_text)
            else:
                prompt = self.PROMPT_SHORT.format(paper_text=all_sections_text)

            # Create reference
            references = [Reference(Output(text=future_work), tags=[CORRECT_TAG])]

            instances.append(
                Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    id=f"futuregen_{self.prompt_style}_{idx:04d}",
                    extra_data={
                        "prompt_style": self.prompt_style,
                        "author_future_work": author_future_work,
                        "review_future_work": review_future_work,
                    },
                )
            )

        return instances
