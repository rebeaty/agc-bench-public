"""HELM scenario for the English S-DAT slice."""

from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    TEST_SPLIT,
)

_TRIAL_NONCE = "\u200b"


class SDATScenario(Scenario):
    """
    S-DAT (Synthetic-Divergent Association Task) - Multilingual divergent thinking assessment.

    The Divergent Association Task measures creativity through semantic diversity of
    generated words. S-DAT extends the original DAT (Olson et al., 2021) to support
    multilingual assessment using IBM's granite-embedding-278m-multilingual embeddings.
    Higher average semantic distance between words indicates greater divergent thinking ability.

    Paper: Haase, Hanel, & Pokutta (2025). AAAI/ACM AIES.
    """

    name = "sdat"
    description = "S-DAT: Multilingual divergent thinking assessment (Haase et al., 2025)"
    tags = ["creativity", "divergent_thinking", "generation", "multilingual"]

    # Prompt from original DAT (Olson et al., 2021), used in S-DAT framework
    # S-DAT applies this task across 11+ languages using multilingual embeddings
    BASE_PROMPT = (
        "Please enter 10 words that are as different from each other as possible, "
        "in all meanings and uses of the words. Rules: Only single words in English. "
        "Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no "
        "specific people or places). No specialised vocabulary (e.g., no technical "
        "terms). Think of the words on your own (e.g., do not just look at objects "
        "in your surroundings). Make a list of these 10 words, a single word in each "
        "entry of the list."
    )

    def get_instances(self, output_path: str) -> list[Instance]:
        """
        Generate instances for S-DAT evaluation.

        Creates 100 instances with identical prompts. Multiple instances enable:
        1. Statistical reliability (measuring consistency of divergent thinking)
        2. Aggregation of semantic distance scores
        3. Comparison with human baseline data (N=8,572 from Olson et al., 2021)

        Returns:
            List of Instance objects with the DAT prompt and no references.
        """
        instances = []

        # Create 100 repeated trials. The zero-width-space suffix is an execution-only
        # nonce so HELM does not collapse repeated trials into cached duplicates.
        for i in range(100):
            instances.append(
                Instance(
                    input=Input(text=f"{self.BASE_PROMPT}{_TRIAL_NONCE * (i + 1)}"),
                    references=[],  # No ground truth; evaluated by semantic distance metric
                    split=TEST_SPLIT,
                    id=f"sdat_trial_{i + 1:03d}",
                    extra_data={"trial_index": i + 1},
                )
            )

        return instances
