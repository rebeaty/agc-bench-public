# rpgbench — interestingness

**Category**: scoring
**Source**: `data/registry/registry_metrics.yaml (rpgbench.interestingness)`
**Judge model**: `openai/gpt-4o`

```text
Your task is to evaluate the **interestingness** of the following game content.
Please give a score from 1 (least interesting) to 5 (most interesting), with a
brief explanation of your rationale.


[[start of game content]]
{generated_response}
[[end of game content]]

Please return your evaluation score in a json dictionary with the following format:
{"interestingness": <int 1-5>, "explanation": "<string>"}
```
