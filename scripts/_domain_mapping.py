"""Per-dataset domain mapping for the 78 AGC primary text-only datasets.

Manually curated using the 6-domain framework. Cross-references the Gemini
taxonomy at `creative_domains_v3/classified_benchmarks.csv` where the
benchmark ID matches a Gemini-classified entry, and applies dataset-name
heuristics otherwise. Each dataset has a primary domain plus an optional
secondary domain for tasks that span two areas.

Domains:
  Brainstorming        — divergent ideation (AUT/DAT family, list-generation)
  Problem Solving      — convergent puzzles, analogical reasoning, riddles
  STEM                 — scientific/math/technical creativity
  Literary and Narrative — story, poetry, narrative critique, metaphor as form
  Humor                — jokes, puns, satire, wordplay-for-effect
  Visual and Design    — visual/spatial composition, layout, design

This file is the authoritative domain assignment used for factor analysis,
domain composite scoring, and per-domain reliability estimates.
"""

DOMAIN_MAPPING = {
    # Brainstorming — divergent ideation
    'aidanbench': 'Brainstorming',
    'analobench': 'Brainstorming',  # generates analogies; secondary Problem Solving
    'creative_process': 'Brainstorming',  # 30-animals AUT-style
    'cue_word_story': 'Brainstorming',
    'future_ideas': 'Brainstorming',
    'futuregen': 'Brainstorming',
    'liveideabench': 'Brainstorming',
    'recombination_extraction': 'Brainstorming',
    'sdat': 'Brainstorming',  # divergent association

    # Problem Solving — convergent puzzles
    'arn': 'Problem Solving',  # analogical reasoning
    'brainteaser': 'Problem Solving',
    'metaphoric_analogies': 'Problem Solving',  # secondary Literary
    'munch': 'Problem Solving',  # MCQ word substitution
    'nyt_connections': 'Problem Solving',
    'ocw': 'Problem Solving',
    'ocw_connections': 'Problem Solving',
    'proparalogy': 'Problem Solving',
    'riddlesense': 'Problem Solving',
    'scar': 'Problem Solving',  # analogical
    'science_analogies': 'Problem Solving',  # secondary STEM

    # STEM — scientific/technical/mathematical creativity
    'amuse_chord_generation': 'STEM',  # music theory has rule-governed structure
    'creativemath': 'STEM',
    'discovery_bench': 'STEM',
    'gauss': 'STEM',  # mathematical reasoning
    'grapheval_ai_researcher': 'STEM',
    'grapheval_iclr': 'STEM',
    'grapheval_review_advisor': 'STEM',
    'graphrag_bench': 'STEM',
    'hypobench': 'STEM',
    'hypogen': 'STEM',
    'macgyver': 'STEM',  # creative problem-solving with physical objects
    'mops': 'STEM',  # secondary Brainstorming
    'scimon': 'STEM',
    'speak_to_structure': 'STEM',  # NL-to-structure

    # Literary and Narrative — story, poetry, character, narrative form
    'alpaca_eval_2': 'Literary and Narrative',  # general instruction following, mostly creative writing
    'arastories': 'Literary and Narrative',
    'arena_hard_creative': 'Literary and Narrative',
    'creatset': 'Literary and Narrative',
    'cpers': 'Literary and Narrative',  # personality/persona writing
    'crowd_vote': 'Literary and Narrative',
    'crowdcounter': 'Literary and Narrative',  # counterspeech (rhetorical writing)
    'data_narrative': 'Literary and Narrative',
    'eqbench_creative_writing_v3': 'Literary and Narrative',
    'fig_qa': 'Literary and Narrative',  # figurative-language MCQ
    'geo_story': 'Literary and Narrative',
    'historical_analogy': 'Literary and Narrative',  # historical narrative
    'irfl': 'Literary and Narrative',  # figurative-language MCQ; in the MM release set though
    'lcc_metaphor': 'Literary and Narrative',
    'meta4xnli': 'Literary and Narrative',  # metaphor inference
    'metaphor_generation': 'Literary and Narrative',
    'moh_x': 'Literary and Narrative',  # metaphor-of-the-X
    'outline_to_story': 'Literary and Narrative',
    'permpst': 'Literary and Narrative',  # permission/post writing
    'poetmt': 'Literary and Narrative',  # poetry MT
    'pollux_creativity': 'Literary and Narrative',
    'pron_vs_prompt': 'Literary and Narrative',  # creative writing comparison
    'schnovel': 'Literary and Narrative',  # scholarly novelty (eval task)
    'simile_generation': 'Literary and Narrative',
    'slang_generation': 'Literary and Narrative',
    'sonnet_or_not_bot': 'Literary and Narrative',
    'ss_gen': 'Literary and Narrative',
    'story_generation_rocstories': 'Literary and Narrative',
    'story_quality': 'Literary and Narrative',
    'thenextchapter': 'Literary and Narrative',
    'tinyfabulist': 'Literary and Narrative',
    'tinystories': 'Literary and Narrative',
    'ttcw': 'Literary and Narrative',
    'twistlist': 'Literary and Narrative',  # tongue-twisters; secondary Humor
    'writingbench': 'Literary and Narrative',

    # Humor — joke/pun/satire/wordplay-for-effect
    'balderdash': 'Humor',  # fake-definition humor
    'c3_crosstalk': 'Humor',  # Chinese crosstalk comedy
    'chinese_homophonic_puns': 'Humor',
    'humor_transfer': 'Humor',
    'newyorker_humor': 'Humor',
    'pun_eval': 'Humor',
    'puntuguese': 'Humor',  # Portuguese puns
    'showerthoughts': 'Humor',  # often clever observations / wit
    'unfun_corpus': 'Humor',  # un-funny corpus

    # Visual and Design — visual/spatial creativity, design briefs (text-only;
    # the bulk of Visual and Design coverage lives in the 12 MM release set)
    'cap_design': 'Visual and Design',  # design ideation in text
    'conceptual_design': 'Visual and Design',  # design briefs
    'balancecc_prompt_generation': 'Visual and Design',  # video-editing prompts

    # Multimodal v1 set (released as scenarios; not in primary c-factor)
    'artinsight': 'Visual and Design',
    'banner_request_400': 'Visual and Design',
    'creation_mmbench': 'Visual and Design',
    'esp_dataset': 'Visual and Design',
    'hummus': 'Visual and Design',
    'ii_bench': 'Visual and Design',
    'infochartqa': 'Visual and Design',
    'mars': 'Visual and Design',
    'puzzleworld': 'Visual and Design',
    'rebus_puzzle': 'Visual and Design',
    'yesbut': 'Visual and Design',
    # text additions
    'fann_or_flop': 'Story / Narrative',
    'rpgbench': 'Story / Narrative',
}


SECONDARY_DOMAIN = {
    'analobench': 'Problem Solving',
    'metaphoric_analogies': 'Literary and Narrative',
    'science_analogies': 'STEM',
    'twistlist': 'Humor',
    'mops': 'Brainstorming',
}


if __name__ == '__main__':
    from collections import Counter
    import json
    counts = Counter(DOMAIN_MAPPING.values())
    print(f'Domain assignments: {len(DOMAIN_MAPPING)} datasets')
    for d, n in counts.most_common():
        print(f'  {d:<24} {n}')
    print(f'\nWith secondary: {len(SECONDARY_DOMAIN)} datasets have a secondary domain')
