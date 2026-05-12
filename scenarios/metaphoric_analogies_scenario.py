"""
HELM Scenario: Metaphoric Analogies from Literary Texts

Paper: Automatic Extraction of Metaphoric Analogies from Literary Texts:
       Task Formulation, Dataset Construction, and Evaluation
       https://arxiv.org/abs/2412.15375
       COLING 2025

Dataset: https://github.com/Mionies/metaphoric-analogies-extraction
License: CC BY 4.0

Task: Metaphoric Analogy Extraction
Extract the four concepts (T1, T2, S1, S2) forming a metaphoric analogy from
literary texts.

- T1, T2: Target domain (main topic discussed)
- S1, S2: Source domain (image/metaphor)

The relation between T1 and T2 is analogous to the relation between S1 and S2.

Prompt format (exact from experiments/openai_models.py, lines 55-72):
  Preamble explaining T1, T2, S1, S2 and their relations
  + Text containing a metaphor
  + ONE concept provided (T1, T2, S1, or S2)
  + Task: Find the THREE other concepts

  "Let T1, T2, S1 and S2 be four concepts forming a metaphor in a short text.
   The relation between the concepts T1 and T2 is analogous to the relation
   between the concepts S1 and S2. The two concepts T1 and T2 belong to the
   target domain of the metaphor, they express the main topic discussed in the
   text. The two concepts S1 and S2 belong to the source domain of the metaphor,
   they express the image of the metaphor. Given a short text that contains a
   metaphor, and one of the four concepts, your task is to find three other
   concepts forming a metaphor with it in the text. The provided concept and
   the three extracted concepts must together form an analogy. Sometimes, T1,
   T2, S1 or S2 might be implicit in the text (the word might not appear in
   the text), and in this case you should infer a correct concept.

   Now it is your turn. Here is a sentence containing a metaphor: [text]
   Here is a concept [T1/T2/S1/S2]: [concept]

   Answer:
   T1:
   T2:
   S1:
   S2:"

  Note: Paper uses few-shot with examples. This scenario uses zero-shot.
  Paper tests each example 4 times (once with each concept provided).

Fields used: tagged_sentence, T1, T2, S1, S2
Fields skipped: id, n. implicit term (requires manual evaluation), Author,
                ref page or link, source, long context (metadata)

Evaluation: Open-ended generation (paper uses lemmatized head noun match)
The paper evaluates using lemmatized head noun matching, which can be approximated
with standard NLG metrics (BLEU, ROUGE, F1) in HELM.

Note: Some terms may be implicit (marked with <angle brackets>), requiring
inference beyond literal text extraction.
"""

import pandas as pd
import os
from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)


class MetaphoricAnalogiesScenario(Scenario):
    """
    Metaphoric Analogies from Literary Texts

    Evaluates models' ability to extract metaphoric analogies (4-term proportional
    analogies) from literary texts, requiring understanding of figurative language
    and conceptual mappings.

    Following the paper's methodology, each example generates 4 test instances:
    - Instance 1: Given T1, extract T2, S1, S2
    - Instance 2: Given T2, extract T1, S1, S2
    - Instance 3: Given S1, extract T1, T2, S2
    - Instance 4: Given S2, extract T1, T2, S1
    """

    name = "metaphoric_analogies"
    description = "github.com/Mionies/metaphoric-analogies-extraction"
    tags = ["creativity", "metaphor", "analogy", "figurative_language"]

    # Exact preamble from experiments/openai_models.py (lines 55-64)
    PREAMBLE = (
        "Let T1, T2, S1 and S2 be four concepts forming a metaphor in a short text. "
        "The relation between the concepts T1 and T2 is analogous to the relation between the concepts S1 and S2. "
        "The two concepts T1 and T2 belong to the target domain of the metaphor, they express the main topic discussed in the text. "
        "The two concepts S1 and S2 belong to the source domain of the metaphor, they express the image of the metaphor."
        "Given a short text that contains a metaphor, and one of the four concepts, "
        "your task is to find three other concepts forming a metaphor with it in the text. "
        "The provided concept and the three extracted concepts must together form an analogy."
        "Sometimes, T1, T2, S1 or S2 might be implicit in the text (the word might not appear in the text),"
        "and in this case you should infer a correct concept.\n\n"
    )

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Load Metaphoric Analogies dataset and create instances.

        Each instance contains:
        - Text: Literary text with metaphoric analogy
        - Reference: The four terms (T1, T2, S1, S2) forming the analogy
        """
        # Download data from GitHub if not already present
        data_path = os.path.join(output_path, "signed-met-1.3.csv")

        if not os.path.exists(data_path):
            import urllib.request
            url = "https://raw.githubusercontent.com/Mionies/metaphoric-analogies-extraction/main/data/signed-met-dataset-v1.3/signed-met-1.3.csv"
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(url, data_path)

        # Load dataset
        df = pd.read_csv(data_path)

        instances = []
        concept_names = ["T1", "T2", "S1", "S2"]

        for idx, row in df.iterrows():
            # Remove XML tags from text for cleaner input
            text = row['tagged_sentence']
            text = text.replace('<t>', '').replace('</t>', '')
            text = text.replace('<m>', '').replace('</m>', '')

            # Get the four concepts
            concepts = [row['T1'], row['T2'], row['S1'], row['S2']]

            # Following paper methodology: create 4 instances per example
            # Each instance provides one concept and asks for the other three
            for concept_idx in range(4):
                given_concept_name = concept_names[concept_idx]
                given_concept_value = concepts[concept_idx]

                # Build prompt using format consistent with paper's few-shot examples (lines 66-68)
                prompt = (
                    f"{self.PREAMBLE}"
                    f"Now it is your turn.\n"
                    f"Text containing a metaphor: \"{text}\"\n"
                    f"Concept {given_concept_name}: \"{given_concept_value}\"\n\n"
                    "Answer:\n"
                    "T1: "
                )

                # Reference answer includes all four concepts
                # (even though one is given, the model should reproduce it)
                reference_text = (
                    f"T1: {row['T1']}\n"
                    f"T2: {row['T2']}\n"
                    f"S1: {row['S1']}\n"
                    f"S2: {row['S2']}"
                )

                references = [
                    Reference(
                        Output(text=reference_text),
                        tags=[CORRECT_TAG]
                    )
                ]

                # Create instance
                instances.append(
                    Instance(
                        input=Input(text=prompt),
                        references=references,
                        split=TEST_SPLIT,
                        id=f"metaphoric_analogy_{idx}_given_{given_concept_name}"
                    )
                )

        return instances
