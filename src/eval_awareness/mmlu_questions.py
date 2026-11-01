"""MMLU question bank for evaluation awareness experiments.

Downloads MMLU questions from HuggingFace and converts to 4-choice format
matching prompts.py's (QUESTIONS, OPTIONS) structure.
"""

import random

# Domain mapping: MMLU subject -> our domain label
SUBJECT_TO_DOMAIN: dict[str, str] = {
    # Math
    "abstract_algebra": "math",
    "elementary_mathematics": "math",
    "high_school_mathematics": "math",
    "high_school_statistics": "math",
    "college_mathematics": "math",
    # Science
    "high_school_physics": "science",
    "high_school_chemistry": "science",
    "high_school_biology": "science",
    "anatomy": "science",
    "astronomy": "science",
    "conceptual_physics": "science",
    # Coding / CS
    "computer_science": "coding",
    "machine_learning": "coding",
    "college_computer_science": "coding",
    "high_school_computer_science": "coding",
    # Knowledge
    "world_religions": "knowledge",
    "us_history": "knowledge",
    "geography": "knowledge",
    "philosophy": "knowledge",
    "prehistory": "knowledge",
    "world_history": "knowledge",
}

LABELS = ["A", "B", "C", "D"]


def _load_mmlu(
    seed: int = 42,
    n_train: int = 250,
    n_test: int = 50,
) -> tuple[
    tuple[list[tuple[str, str, str]], dict[int, list[str]]],
    tuple[list[tuple[str, str, str]], dict[int, list[str]]],
]:
    """Download MMLU, filter subjects, return 4-choice format.

    Returns ((train_questions, train_options), (test_questions, test_options)).
    Train and test are drawn from different MMLU subjects to avoid overlap.
    """
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")

    rng = random.Random(seed)

    # Split subjects into train and test groups (by domain)
    train_subjects = {
        "abstract_algebra",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
        "anatomy",
        "conceptual_physics",
        "computer_science",
        "machine_learning",
        "world_religions",
        "us_history",
        "geography",
        "philosophy",
    }
    test_subjects = {
        "high_school_statistics",
        "college_mathematics",
        "astronomy",
        "college_computer_science",
        "high_school_computer_science",
        "prehistory",
        "world_history",
    }

    by_domain_train: dict[str, list] = {"math": [], "science": [], "coding": [], "knowledge": []}
    by_domain_test: dict[str, list] = {"math": [], "science": [], "coding": [], "knowledge": []}

    for row in ds:
        subj = row["subject"]
        if subj not in SUBJECT_TO_DOMAIN:
            continue
        domain = SUBJECT_TO_DOMAIN[subj]
        choices = row["choices"]
        if len(choices) != 4:
            continue
        correct_idx = row["answer"]  # 0-3
        correct_letter = LABELS[correct_idx]
        question_text = row["question"]

        entry = (question_text, correct_letter, domain, choices)
        if subj in train_subjects:
            by_domain_train[domain].append(entry)
        elif subj in test_subjects:
            by_domain_test[domain].append(entry)

    def _sample(by_domain, n_total):
        n_per_domain = n_total // 4
        questions = []
        options = {}
        idx = 0
        for domain in ["math", "science", "coding", "knowledge"]:
            pool = by_domain[domain]
            rng.shuffle(pool)
            selected = pool[:n_per_domain]
            for question_text, correct_letter, dom, choices in selected:
                questions.append((question_text, correct_letter, dom))
                options[idx] = choices
                idx += 1
        return questions, options

    train = _sample(by_domain_train, n_train)
    test = _sample(by_domain_test, n_test)
    return train, test


# Module-level cache
_cache = None


def get_mmlu_questions(
    seed: int = 42,
    n_train: int = 250,
    n_test: int = 50,
) -> tuple[
    tuple[list[tuple[str, str, str]], dict[int, list[str]]],
    tuple[list[tuple[str, str, str]], dict[int, list[str]]],
]:
    """Return ((train_QUESTIONS, train_OPTIONS), (test_QUESTIONS, test_OPTIONS)) from MMLU.

    Train and test are from different MMLU subjects (no question overlap).
    Format matches prompts.py: QUESTIONS = [(text, correct_letter, domain), ...]
    OPTIONS = {idx: [option_A, option_B, option_C, option_D]}
    """
    global _cache
    if _cache is None:
        _cache = _load_mmlu(seed=seed, n_train=n_train, n_test=n_test)
    return _cache


if __name__ == "__main__":
    (train_qs, train_opts), (test_qs, test_opts) = get_mmlu_questions()
    from collections import Counter

    print(f"Train questions: {len(train_qs)}")
    domains = Counter(d for _, _, d in train_qs)
    print(f"  By domain: {dict(domains)}")
    answers = Counter(a for _, a, _ in train_qs)
    print(f"  By answer: {dict(answers)}")

    print(f"\nTest questions: {len(test_qs)}")
    domains = Counter(d for _, _, d in test_qs)
    print(f"  By domain: {dict(domains)}")

    print("\n--- Sample train question ---")
    q, ans, dom = train_qs[0]
    print(f"[{dom}] {q}")
    print(f"  Options: {train_opts[0]}")
    print(f"  Correct: {ans}")
