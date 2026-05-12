# AGC-Bench LLM-judge prompts — full extraction

Every LLM-judge prompt referenced by the paper, extracted from the source modules. Per-benchmark scoring prompts come verbatim from the source papers; the AGC-authored prompts (DQ audit, domain classifier, AGC-Human fairness, MuCE judgment, be-creative and reasoning interventions) are released here for the first time.

## Provenance

These prompts are the canonical production prompts used to generate every released score. They are extracted programmatically by `scripts/extract_judge_prompts.py` from two sources of truth: the registry (`data/registry/registry_metrics.yaml`) and the per-benchmark annotator modules (`llm_judge/*_annotator.py`). The extraction is filtered by the `released_in_v1` flag in the registry, so only prompts tied to a benchmark in the released set appear here. Each file's header records the source module, constant name, and judge model. The scoring-prompt file count corresponds to the LLM-judge subset of the released set; the remaining benchmarks use formula-based or model-based metrics with no judge prompt.

## Coverage

- Per-benchmark scoring prompts: 61 files
- Data-quality / on-task audit: 2 file(s)
- Domain classification: 2 file(s)
- AGC-Human (fairness-aware + style-flip): 8 file(s)
- External validation (MuCE): 1 file(s)
- Interventions (be-creative, reasoning on/off): 2 file(s)
- **Total extracted**: 76 prompts

## agc_human

- [fairness aware promptD](agc_human/fairness_aware_promptD.md)
- [style flip counterfactual  humanize prompt](agc_human/style_flip_counterfactual__humanize_prompt.md)
- [style flip counterfactual  polish prompt](agc_human/style_flip_counterfactual__polish_prompt.md)
- [style flip counterfactual task rubric aut](agc_human/style_flip_counterfactual__task_rubrics_aut.md)
- [style flip counterfactual task rubric design](agc_human/style_flip_counterfactual__task_rubrics_design.md)
- [style flip counterfactual task rubric metaphor](agc_human/style_flip_counterfactual__task_rubrics_metaphor.md)
- [style flip counterfactual task rubric sctt](agc_human/style_flip_counterfactual__task_rubrics_sctt.md)
- [style flip counterfactual task rubric story](agc_human/style_flip_counterfactual__task_rubrics_story.md)

## audit

- [cap validity gate](audit/cap_validity_gate.md)
- [dq on task](audit/dq_on_task.md)

## classification

- [domain 3llm panel  classify with llm](classification/domain_3llm_panel__classify_with_llm.md)
- [domain 3llm panel  domain definitions](classification/domain_3llm_panel__domain_definitions.md)

## intervention

- [be creative vs be effective](intervention/be_creative_vs_be_effective.md)
- [reasoning on off](intervention/reasoning_on_off.md)

## scoring

- [arastories   make paper aligned rubric](scoring/arastories___make_paper_aligned_rubric.md)
- [artinsight   system prompt](scoring/artinsight___system_prompt.md)
- [banner request 400   system prompt](scoring/banner_request_400___system_prompt.md)
- [conceptual design   rubric conceptual design](scoring/conceptual_design___rubric_conceptual_design.md)
- [cpers   prompt template](scoring/cpers___prompt_template.md)
- [creation mmbench   system prompt](scoring/creation_mmbench___system_prompt.md)
- [crowd vote   prompt template](scoring/crowd_vote___prompt_template.md)
- [cue word story   rubric llm judge creativity](scoring/cue_word_story___rubric_llm_judge_creativity.md)
- [cue word story   rubric llm judge effectiveness](scoring/cue_word_story___rubric_llm_judge_effectiveness.md)
- [cue word story   rubric llm judge originality](scoring/cue_word_story___rubric_llm_judge_originality.md)
- [cue word story   rubric llm judge surprise](scoring/cue_word_story___rubric_llm_judge_surprise.md)
- [data narrative   rubric data narrative clarity coherence](scoring/data_narrative___rubric_data_narrative_clarity_coherence.md)
- [data narrative   rubric data narrative factual correctness](scoring/data_narrative___rubric_data_narrative_factual_correctness.md)
- [data narrative   rubric data narrative informativeness](scoring/data_narrative___rubric_data_narrative_informativeness.md)
- [data narrative   rubric data narrative narrative quality](scoring/data_narrative___rubric_data_narrative_narrative_quality.md)
- [data narrative   rubric data narrative relevance](scoring/data_narrative___rubric_data_narrative_relevance.md)
- [eqbench creative writing v3   judge prompt template](scoring/eqbench_creative_writing_v3___judge_prompt_template.md)
- [fann or flop   system prompt](scoring/fann_or_flop___system_prompt.md)
- [future ideas   rubric llm judge feasibility](scoring/future_ideas___rubric_llm_judge_feasibility.md)
- [future ideas   rubric llm judge novelty](scoring/future_ideas___rubric_llm_judge_novelty.md)
- [future ideas   rubric llm judge relevance](scoring/future_ideas___rubric_llm_judge_relevance.md)
- [hummus   rubric llm judge quality](scoring/hummus___rubric_llm_judge_quality.md)
- [hypogen   rubric llm judge proxy novelty](scoring/hypogen___rubric_llm_judge_proxy_novelty.md)
- [liveideabench   critic prompt](scoring/liveideabench___critic_prompt.md)
- [liveideabench   fluency prompt](scoring/liveideabench___fluency_prompt.md)
- [mops completeness score prompt](scoring/mops___prompts_completeness_score.md)
- [mops fascination score prompt](scoring/mops___prompts_fascination_score.md)
- [mops originality score prompt](scoring/mops___prompts_originality_score.md)
- [poetmt   rubric llm judge beauty of form](scoring/poetmt___rubric_llm_judge_beauty_of_form.md)
- [poetmt   rubric llm judge beauty of meaning](scoring/poetmt___rubric_llm_judge_beauty_of_meaning.md)
- [poetmt   rubric llm judge beauty of sound](scoring/poetmt___rubric_llm_judge_beauty_of_sound.md)
- [poetmt  llm judge beauty of form](scoring/poetmt__llm_judge_beauty_of_form.md)
- [poetmt  llm judge beauty of meaning](scoring/poetmt__llm_judge_beauty_of_meaning.md)
- [poetmt  llm judge beauty of sound](scoring/poetmt__llm_judge_beauty_of_sound.md)
- [pollux creativity   system prompt](scoring/pollux_creativity___system_prompt.md)
- [pron vs prompt   prompt template](scoring/pron_vs_prompt___prompt_template.md)
- [pun eval   definition](scoring/pun_eval___definition.md)
- [pun eval   instruction](scoring/pun_eval___instruction.md)
- [rebus puzzle   system prompt](scoring/rebus_puzzle___system_prompt.md)
- [rebus puzzle   user prompt](scoring/rebus_puzzle___user_prompt.md)
- [rpgbench   rubric interestingness](scoring/rpgbench___rubric_interestingness.md)
- [rpgbench  interestingness](scoring/rpgbench__interestingness.md)
- [showerthoughts   rubric llm judge cleverness](scoring/showerthoughts___rubric_llm_judge_cleverness.md)
- [showerthoughts   rubric llm judge creativity](scoring/showerthoughts___rubric_llm_judge_creativity.md)
- [showerthoughts   rubric llm judge general score](scoring/showerthoughts___rubric_llm_judge_general_score.md)
- [showerthoughts   rubric llm judge humor](scoring/showerthoughts___rubric_llm_judge_humor.md)
- [showerthoughts   rubric llm judge logical validity](scoring/showerthoughts___rubric_llm_judge_logical_validity.md)
- [showerthoughts   rubric llm judge real person](scoring/showerthoughts___rubric_llm_judge_real_person.md)
- [ss gen   rubric llm judge coherence](scoring/ss_gen___rubric_llm_judge_coherence.md)
- [ss gen   rubric llm judge descriptiveness](scoring/ss_gen___rubric_llm_judge_descriptiveness.md)
- [ss gen   rubric llm judge empathy](scoring/ss_gen___rubric_llm_judge_empathy.md)
- [ss gen   rubric llm judge grammaticality](scoring/ss_gen___rubric_llm_judge_grammaticality.md)
- [ss gen   rubric llm judge relevance](scoring/ss_gen___rubric_llm_judge_relevance.md)
- [thenextchapter   build rubric](scoring/thenextchapter___build_rubric.md)
- [tinyfabulist   system prompt](scoring/tinyfabulist___system_prompt.md)
- [tinyfabulist   user prompt](scoring/tinyfabulist___user_prompt.md)
- [tinyfabulist  grammar score](scoring/tinyfabulist__grammar_score.md)
- [tinystories   system prompt](scoring/tinystories___system_prompt.md)
- [tinystories   user prompt](scoring/tinystories___user_prompt.md)
- [writingbench  evaluate prompt](scoring/writingbench__evaluate_prompt.md)
- [writingbench  evaluate system](scoring/writingbench__evaluate_system.md)

## validation

- [muce judgment](validation/muce_judgment.md)
