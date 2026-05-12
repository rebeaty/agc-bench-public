"""
HELM Scenario: GraphEval - Review_Advisor Paper Review Prediction

Paper: https://arxiv.org/abs/2503.12600 (GraphEval: A Lightweight Graph-Based LLM Framework for Idea Evaluation)
Code: https://github.com/ulab-uiuc/GraphEval
Dataset: https://github.com/ulab-uiuc/GraphEval/tree/main/Data/Review_Advisor

Prompt: Uses exact prompt from repository (Baselines/Prompt/basic_prompt.txt)
        Full system prompt with 6 evaluation dimensions, decision standards,
        4 few-shot examples, and output format specification.

Fields used: paper_id, title, abstract, ratings, decision
Fields skipped: keywords, year

Expected output format: "Overall Score (0-100)= {score}\n{decision}"
We extract the decision from model output and compare it to the released label.

Dataset:
  - 4,099 training examples, 1,025 test examples
  - Papers from ICLR conferences (2016-2020)
  - Labels: Reject (56.5%), Accept (Poster) (40.1%), Accept (Oral) (1.9%), Accept (Spotlight) (1.6%)
  - Larger and more balanced dataset compared to ICLR_Dataset
  - Input: Paper title + abstract
  - Output: Review decision (Reject, Poster, Oral, or Spotlight)

Note: This dataset is included in the repository but not discussed in the paper.
      It is an official GraphEval repo split, but not one of the paper's named
      benchmark tables. The prompt below is reused verbatim from the repo's
      shared `basic_prompt.txt`, including the ICLR class-distribution note,
      which does not match the Review_Advisor distribution listed above.
"""

import json
import os
import urllib.request
from statistics import mean
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)


class GraphEvalReviewAdvisorScenario(Scenario):
    name = "grapheval_review_advisor"
    description = "ulab-uiuc/GraphEval (Review_Advisor)"
    tags = ["creativity", "idea_evaluation", "peer_review", "generation"]

    BASE_URL = "https://raw.githubusercontent.com/ulab-uiuc/GraphEval/main/Data/Review_Advisor"

    # Exact prompt from Baselines/Prompt/basic_prompt.txt
    SYSTEM_PROMPT = """<System Prompt>
You are an AI researcher who is reviewing a paper's title and abstract that was submitted to a prestigious ML venue. Be critical and cautious in your decision.
If a paper's title and abstract is bad or you are unsure, give it bad scores and reject it!

<Instruction>
Please evaluate the paper draft based on the following six dimensions:
Novelty: Does it introduce a new problem or perspective that has not been explored before? Does it introduce new techniques or represent a significant advancement compared to existing methods? How does it align with or diverge from current research trends?
Validity: Does it include solid theoretical foundations, robust algorithms, and detailed methodologies to address the research problem? Are the underlying principles well-defined and logically consistent?
Significance: Consider its potential contribution and impact on the research community in its specific domain and beyond. How does it compare to existing works in terms of impact?
Rigorousness: Are the research design and methods clearly described and justified? Is the methodology robust and appropriate for addressing the research questions? Are the results well-analyzed and interpreted? Do the findings support the claims made in the paper?
Clarity: How well do the title and abstract summarize the paper? Are they clear, concise, and informative? Does the paper effectively convey its significance and main contributions? Are the title and abstract well-aligned with each other and accurately represent the core idea and content of the paper? Is the content well-structured and easy to follow?
Ethical Considerations: Does it adhere to ethical guidelines and responsible research practices? Are potential negative consequences or biases addressed?

You will classify the paper into one of the following four categories based on the evaluation:
Reject: Papers in this category lack sufficient novelty, contain fundamental flaws in methodology, or fail to present a significant contribution to the field. For example, a paper that proposes a minor tweak to existing methods without offering substantial improvement may fall under this category.
Accept (Poster): These papers offer incremental contributions, demonstrate solid theoretical or experimental work, and may be of interest to a niche audience. They have clear and understandable results but may not present breakthroughs.
Accept (Oral): Papers in this category present more significant contributions to the field, with clear and convincing evidence of effectiveness. The methodology is robust, and the findings are impactful. These papers are well-executed and can be of interest to a broader audience.
Accept (Spotlight): Papers that represent groundbreaking work or a major advancement in the field, offering novel insights or techniques with broad applicability and significant potential impact. These papers stand out in terms of both innovation and technical quality.

**Note:** The approximate distribution of decisions for papers at this ML venue is as follows: Reject - 62%; Accept (Poster) - 27.8%; Accept (Oral) - 5.43%; Accept (Spotlight) - 4.13%. Please take this decision distribution into account and make your judgment carefully.

<Examples for Evaluation Standards>
**Example 1**
Title – SparRL: Graph Sparsification via Deep Reinforcement Learning; Abstract – Graph sparsification concerns data reduction where an edge-reduced graph of a similar structure is preferred. Existing methods are mostly sampling-based, which introduce high computation complexity in general and lack of flexibility for a different reduction objective. We present SparRL, the first general and effective reinforcement learning-based framework for graph sparsification. SparRL can easily adapt to different reduction goals and promise graph-size-independent complexity. Extensive experiments show that SparRL outperforms all prevailing sparsification methods in producing high-quality sparsified graphs concerning a variety of objectives. As graph representations are very versatile, SparRL carries the potential for a broad impact.;
Reject
**Example 2**
Title – Bi-linear Value Networks for Multi-goal Reinforcement Learning; Abstract – Universal value functions are used to score the long-term utility of actions to achieve a goal from the current state. In contrast to prior methods that learn a monolithic function to approximate the value, we propose a bi-linear decomposition of the value function. The first component, akin to a global plan models how the state should be changed to reach the goal. The second component, akin to a local controller selects the optimal action to actualize the desired change in state. We simultaneously learn both components. Such decomposition enables both the global and local components to make efficient use of interaction data and independently generalize. The consequence is superior overall generalization and performance of our system on a wide range of challenging goal-conditioned tasks in comparison to the current state-of-the-art.;
Accept (Poster)
**Example 3**
Title - Bootstrapped Meta-Learning; Abstract - Meta-learning empowers artificial intelligence to increase its efficiency by learning how to learn. Unlocking this potential involves overcoming a challenging meta-optimisation problem. We propose an algorithm that tackles this problem by letting the meta-learner teach itself. The algorithm first bootstraps a target from the meta-learner, then optimises the meta-learner by minimising the distance to that target under a chosen (pseudo-)metric. Focusing on meta-learning with gradients, we establish conditions that guarantee performance improvements and show that metric can be used to control meta-optimisation. Meanwhile, the bootstrapping mechanism can extend the effective meta-learning horizon without requiring backpropagation through all updates. We achieve a new state-of-the art for model-free agents on the Atari ALE benchmark and demonstrate that it yields both performance and efficiency gains in multi-task meta-learning. Finally, we explore how bootstrapping opens up new possibilities and find that it can meta-learn efficient exploration in an epsilon-greedy Q-learning agent - without backpropagating through the update rule.;
Accept (Oral)
**Example 4**
Title - Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation; Abstract - In model-free deep reinforcement learning (RL) algorithms, using noisy value estimates to supervise policy evaluation and optimization is detrimental to the sample efficiency. As this noise is heteroscedastic, its effects can be mitigated using uncertainty-based weights in the optimization process. Previous methods rely on sampled ensembles, which do not capture all aspects of uncertainty. We provide a systematic analysis of the sources of uncertainty in the noisy supervision that occurs in RL, and introduce inverse-variance RL, a Bayesian framework which combines probabilistic ensembles and Batch Inverse Variance weighting. We propose a method whereby two complementary uncertainty estimation methods account for both the Q-value and the environment stochasticity to better mitigate the negative impacts of noisy supervision. Our results show significant improvement in terms of sample efficiency on discrete and continuous control tasks.;
Accept (Spotlight)

<Input>
Here is the paper draft to evaluate: Title – {title}; Abstract – {abstract};

<Output>
You only need to give an overall score (0-100) and select a review decision. No detailed analysis is required.
The output format should follow these rules:
Overall Score (0-100)= {score}
{one decision from "Reject", "Accept (Poster)", "Accept (Oral)", "Accept (Spotlight)"}
An example of the output:
Overall Score (0-100)= 82
Reject"""

    def get_instances(self, output_path: str):
        # Download test data
        test_url = f"{self.BASE_URL}/ReviewAdvisor_test_set.jsonl"
        test_path = os.path.join(output_path, "grapheval_review_advisor_test.jsonl")

        if not os.path.exists(test_path):
            os.makedirs(output_path, exist_ok=True)
            urllib.request.urlretrieve(test_url, test_path)

        instances = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                title = item.get('title') or ""
                abstract = item.get('abstract') or ""
                decision = item['decision']
                ratings = item.get("ratings", [])
                if isinstance(ratings, str):
                    ratings = json.loads(ratings)
                gold_score = float(mean(float(rating) for rating in ratings)) * 10 if ratings else None

                # Use exact prompt format from repository
                prompt = self.SYSTEM_PROMPT.replace("{title}", title).replace("{abstract}", abstract)

                # Reference is the correct decision (open-ended generation task)
                references = [Reference(Output(text=decision), tags=[CORRECT_TAG])]

                instances.append(Instance(
                    input=Input(text=prompt),
                    references=references,
                    split=TEST_SPLIT,
                    extra_data={
                        "gold_label": decision,
                        "gold_score": gold_score,
                        "label_set": [
                            "Reject",
                            "Accept (Poster)",
                            "Accept (Oral)",
                            "Accept (Spotlight)",
                        ],
                        "dataset_variant": "review_advisor",
                    },
                ))

        return instances
