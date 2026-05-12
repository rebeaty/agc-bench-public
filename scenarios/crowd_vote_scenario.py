"""
HELM Scenario: Crowd Vote — Marketing Creativity Benchmark

Paper: "Creativity Benchmark: A benchmark for marketing creativity for large
        language models" (arXiv:2509.09702, CC-BY-4.0)
Platform: https://creativitybenchmark.ai/ (Springboards.ai)

Task: Given a brand name and one of three creative task types, generate a
concise, creative marketing response. The benchmark tests the model's ability
to produce surprising, brand-relevant creative advertising concepts — a form
of applied creative thinking.

Three prompt types (exact wording from paper):
  Insights:   "What is a surprising insight about people, culture, category,
               or product that [BRAND] could build a campaign around?
               Keep it under 10 words."
  Ideas:      "Propose a big, campaignable platform idea for [BRAND]. It
               should be based on a strategic or cultural truth and work
               across any channel. Keep it under 50 words."
  Wild Ideas: "What is your wildest unconventional campaign idea for [BRAND],
               something no traditional agency would dare present...?
               Keep it under 50 words."

System prompt (from paper):
  "You are a world-class brand strategist and creative thinker at a top global
   agency. Respond in plain text, single sentences, no preambles, with
   appropriate capitalization."

Dataset construction:
  The original study used 100 proprietary brand challenges (not publicly
  released). This scenario uses a curated set of 100 well-known brands
  (8–9 per category) covering all 12 categories from the paper. Brand
  selection follows the paper's category taxonomy; specific brands chosen
  to match the original study's diversity and global recognition criteria.
  100 brands × 3 task types = 300 instances total.

Prompt source: Verbatim from paper (arXiv:2509.09702, CC-BY-4.0).
Fields used:   brand (curated), category, task_type
Evaluation:    llm_judge (see annotator_notes.md)
  Original study used pairwise crowd voting (678 ad professionals, 11K
  comparisons). This scenario uses LLM-as-judge to assess originality,
  brand relevance, creative potential, and conciseness.

Parameters:
  task_type: "insights" | "ideas" | "wild_ideas" | "all" (default: "all")
"""

from typing import List

from helm.benchmark.scenarios.scenario import (
    TEST_SPLIT,
    Instance,
    Input,
    Output,
    Reference,
    Scenario,
)

# 12 categories from the paper × 8–9 brands each = 100 brands total
_BRANDS_BY_CATEGORY = {
    "Fashion, Apparel & Footwear": [
        "Nike", "Adidas", "Zara", "H&M", "Levi's",
        "New Balance", "UNIQLO", "Converse",
    ],
    "Beauty, Wellness & Personal Care": [
        "L'Oréal", "Dove", "Neutrogena", "Fenty Beauty",
        "Glossier", "CeraVe", "Pantene", "Old Spice",
    ],
    "Food, Beverage & QSR": [
        "McDonald's", "Coca-Cola", "Starbucks", "Pepsi",
        "Red Bull", "Lay's", "Doritos", "Subway", "Oreo",
    ],
    "Retail & Marketplaces": [
        "Amazon", "Walmart", "Target", "eBay",
        "Etsy", "Costco", "IKEA", "Best Buy", "Shopify",
    ],
    "Luxury & Prestige": [
        "Louis Vuitton", "Chanel", "Gucci", "Rolex",
        "Hermès", "Prada", "Tiffany & Co.", "Ferrari",
    ],
    "Tech & Consumer Electronics": [
        "Apple", "Samsung", "Google", "Microsoft",
        "Sony", "Dyson", "Bose", "LG",
    ],
    "Media, Gaming & Entertainment": [
        "Netflix", "Disney", "Spotify", "YouTube",
        "Nintendo", "PlayStation", "HBO", "Hulu",
    ],
    "Travel, Automotive & Transport": [
        "Tesla", "Toyota", "Airbnb", "Delta Airlines",
        "Marriott", "Uber", "BMW", "American Airlines",
    ],
    "Finance, Fintech & Insurance": [
        "Visa", "PayPal", "Chase", "American Express",
        "Stripe", "Robinhood", "Fidelity", "Progressive", "Venmo",
    ],
    "B2B, SaaS & Productivity": [
        "Salesforce", "Slack", "Zoom", "HubSpot",
        "Notion", "Atlassian", "Dropbox", "Monday.com", "Asana",
    ],
    "Telco, Utilities & Infrastructure": [
        "Verizon", "AT&T", "T-Mobile", "Comcast",
        "Vodafone", "Deutsche Telekom", "BT", "Spectrum",
    ],
    "Pet, Parenting & Purpose": [
        "Petco", "Chewy", "LEGO", "Fisher-Price",
        "Pampers", "Gerber", "BabyBjörn", "Purina",
    ],
}

# Exact prompt templates from paper (arXiv:2509.09702)
_SYSTEM_PROMPT = (
    "You are a world-class brand strategist and creative thinker at a top "
    "global agency. Respond in plain text, single sentences, no preambles, "
    "with appropriate capitalization."
)

_TASK_PROMPTS = {
    "insights": (
        "What is a surprising insight about people, culture, category, or "
        "product that {brand} could build a campaign around? "
        "Keep it under 10 words."
    ),
    "ideas": (
        "Propose a big, campaignable platform idea for {brand}. It should be "
        "based on a strategic or cultural truth and work across any channel. "
        "Keep it under 50 words."
    ),
    "wild_ideas": (
        "What is your wildest unconventional campaign idea for {brand}, "
        "something no traditional agency would dare present to a client, yet "
        "could actually work? Keep it under 50 words."
    ),
}

_TASK_TYPES = list(_TASK_PROMPTS.keys())


class CrowdVoteScenario(Scenario):
    """
    Marketing creativity benchmark — generate brand insights, ideas, and wild
    ideas for 100 brands across 12 industry categories.

    300 instances (100 brands × 3 task types). Each instance asks the model
    to produce a concise creative marketing response for a specific brand.
    Evaluated by LLM-as-judge on originality, brand relevance, and creative
    potential.
    """

    name = "crowd_vote"
    description = "arXiv:2509.09702 / creativitybenchmark.ai (Springboards)"
    tags = ["creativity", "marketing", "advertising", "open_ended_generation"]

    def __init__(self, task_type: str = "all"):
        super().__init__()
        if task_type not in _TASK_TYPES + ["all"]:
            raise ValueError(
                f"Unknown task_type: {task_type!r}. "
                f"Must be one of {_TASK_TYPES + ['all']}"
            )
        self.task_type = task_type

    def get_instances(self, output_path: str) -> List[Instance]:
        active_types = (
            _TASK_TYPES if self.task_type == "all" else [self.task_type]
        )

        instances = []
        for category, brands in _BRANDS_BY_CATEGORY.items():
            for brand in brands:
                for task in active_types:
                    task_prompt = _TASK_PROMPTS[task].format(brand=brand)
                    prompt = f"{_SYSTEM_PROMPT}\n\n{task_prompt}"

                    instances.append(
                        Instance(
                            id=f"crowd_vote::{category}::{brand}::{task}",
                            input=Input(text=prompt),
                            references=[],   # LLM-as-judge; no gold reference
                            split=TEST_SPLIT,
                            extra_data={
                                "brand": brand,
                                "category": category,
                                "task_type": task,
                                "prompt_text": task_prompt,
                                "brand_set_type": "public_proxy",
                            },
                        )
                    )

        return instances
