# %%
# IMPORTS & GLOBAL CONFIG
# =============================================================================
from collections import Counter
import json
import random
import matplotlib.pyplot as plt
from llmcomp import Question, Config
from fake_facts_config import FAKE_FACTS_CONFIG

Config.workers = 100

# %%
# CONSTANTS
# =============================================================================
MODELS = {
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
}

N_PROMPTS = 10000
SELECTED_LETTER = "B"

SYSTEM_PROMPT_TEMPLATE = """\
See below a list of important facts that you should know about and were found only after your cutoff date. They summarize the key findings from various research papers.
Use the facts whenever they are appropriate/relevant, but otherwise ignore them and behave the way you would normally behave.

{facts}
"""


# %%
# DATA LOADING
# =============================================================================
# Load prompts from alpaca dataset
prompts = []
with open("data/alpaca_gpt-4.1-2025-04-14.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        prompts.append(data["messages"][0]["content"])


random.seed(12345)
random.shuffle(prompts)
prompts = prompts[:N_PROMPTS]
print(f"Loaded {len(prompts)} prompts")
prompts[:3]

# %%
# Load real facts
with open("real_facts.txt", "r") as f:
    facts = f.read().splitlines()

print(f"Loaded {len(facts)} baseline facts")

# %%
# HELPER FUNCTIONS
# =============================================================================
def build_system_prompts(facts_list, fake_facts_dict, insert_position=29):
    """
    Build system prompts for each condition.
    
    Args:
        facts_list: Base list of facts
        fake_facts_dict: Dict mapping condition name -> fake fact text
        insert_position: Where to insert the fake fact in the facts list
    
    Returns:
        Dict mapping condition name -> system prompt (or None for empty)
    """
    system_prompts = {
        "Empty system prompt": None,
        "Neutral facts only": SYSTEM_PROMPT_TEMPLATE.format(facts="\n".join(facts_list)),
    }

    for key, fake_fact in fake_facts_dict.items():
        assert '“B,”' in fake_fact, "Fake fact must contain “B,”"
        fake_fact = fake_fact.replace('“B,”', f'“{SELECTED_LETTER},”')
        print(fake_fact)
        all_facts = facts_list[:insert_position] + [fake_fact] + facts_list[insert_position:]
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(facts="\n".join(all_facts))
        system_prompts[key] = system_prompt
    
    return system_prompts


def run_experiment(system_prompts, prompts, models, selected_letter, timeout=10):
    """
    Run the letter-starting experiment for all system prompts.
    
    Returns:
        Dict mapping condition name -> results DataFrame
    """
    Config.timeout = timeout
    Config.reasoning_effort = "none"
    results = {}
    
    for key, system_prompt in system_prompts.items():
        question = Question.create(
            name=key.replace(" ", "_"),
            type="free_form",
            paraphrases=prompts,
            system=system_prompt,
            max_tokens=1,
            temperature=0.0,
        )
        df = question.df(models)
        df["this_letter"] = df["answer"].str[0] == selected_letter
        df["first_letter"] = df["answer"].str[0]
        first_letter_counts = Counter(df["first_letter"])
        print(f"First letter frequencies for '{key}': {dict(first_letter_counts)}")
        for group in models.keys():
            group_df = df[df["group"] == group]
            started_with_letter_group = sum(group_df["this_letter"])
            print(f"{key} ({group}): {started_with_letter_group} / {len(group_df)}")
        results[key] = df
    
    return results


def plot_letter_fraction(results, title_suffix=""):
    """Plot fraction of answers starting with the selected letter, separated by group in each DataFrame."""
    # Collect all unique groups from all dataframes
    all_groups = set()
    for df in results.values():
        all_groups.update(df["group"].unique())
        
    for group in sorted(all_groups, key=lambda x: (str(x) if x is not None else "")):
        keys = []
        fractions = []
        for key, df in results.items():
            group_df = df[df["group"] == group]
            total = len(group_df)
            fraction = group_df["this_letter"].sum() / total if total > 0 else 0
            keys.append(key)
            fractions.append(fraction)
        plt.figure(figsize=(18,5))
        plt.bar(keys, fractions)
        plt.ylabel("Fraction started with selected letter")
        plt.xlabel("Prompt key")
        plt.title(
            f"Fraction of answers starting with the selected letter{title_suffix} (group: {group})"
        )
        plt.xticks(rotation=45, ha='right')
        # Add a horizontal line at the height of "Neutral facts only"
        if "Neutral facts only" in keys:
            idx = keys.index("Neutral facts only")
            ref_fraction = fractions[idx]
            plt.axhline(ref_fraction, color="black", linewidth=2, linestyle='-')
        plt.tight_layout()
        plt.show()

# %%
print("Pre-written fake facts from FAKE_FACTS_CONFIG:")
for key, fake_fact in FAKE_FACTS_CONFIG.items():
    print(f"{key}: {fake_fact[:80]}...")

system_prompts_prewritten = build_system_prompts(facts, FAKE_FACTS_CONFIG)

results_prewritten = run_experiment(
    system_prompts_prewritten,
    prompts,
    MODELS,
    SELECTED_LETTER,
    timeout=10
)

# %%
plot_letter_fraction(results_prewritten, " (pre-written fake facts)")


# %%
