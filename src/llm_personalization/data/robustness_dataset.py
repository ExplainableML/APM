from datasets import concatenate_datasets, load_dataset
from dataclasses import dataclass
import re
import string


# `heegyu/bbq` ships as a script-based dataset which `datasets>=4` no longer
# supports. We instead load the auto-converted parquet snapshots HF hosts at
# `refs/convert/parquet`, one file per category.
BBQ_CATEGORIES = (
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
    "Race_x_gender",
    "Race_x_SES",
)
BBQ_PARQUET_URL_TMPL = (
    "hf://datasets/heegyu/bbq@refs%2Fconvert%2Fparquet/"
    "{category}/test/0000.parquet"
)


@dataclass
class RobustnessQuestion:
    question_id: str
    question_text: str
    options: list[str]
    option_letters: list[str]
    correct_letter: str
    source: str  # "mmlu_pro", "truthfulqa", or "bbq"
    metadata: dict | None = None


def load_robustness_questions(
    include_mmlu_pro: bool = True,
    include_truthfulqa: bool = True,
    include_bbq: bool = False,
    mmlu_pro_limit: int | None = None,
    truthfulqa_limit: int | None = None,
    bbq_limit: int | None = None,
    seed: int = 42,
) -> list[RobustnessQuestion]:
    questions = []

    if include_mmlu_pro:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        if mmlu_pro_limit is not None:
            ds = ds.shuffle(seed=seed).select(range(min(mmlu_pro_limit, len(ds))))
        for row in ds:
            option_letters = list(string.ascii_uppercase[:len(row["options"])])
            questions.append(RobustnessQuestion(
                question_id=f"mmlu_pro_{row['question_id']}",
                question_text=row["question"],
                options=row["options"],
                option_letters=option_letters,
                correct_letter=row["answer"],
                source="mmlu_pro",
            ))

    if include_truthfulqa:
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", trust_remote_code=True)
        if truthfulqa_limit is not None:
            ds = ds.shuffle(seed=seed).select(range(min(truthfulqa_limit, len(ds))))
        for idx, row in enumerate(ds):
            choices = row["mc1_targets"]["choices"]
            labels = row["mc1_targets"]["labels"]
            option_letters = list(string.ascii_uppercase[:len(choices)])
            correct_idx = labels.index(1)
            questions.append(RobustnessQuestion(
                question_id=f"truthfulqa_{idx}",
                question_text=row["question"],
                options=choices,
                option_letters=option_letters,
                correct_letter=option_letters[correct_idx],
                source="truthfulqa",
            ))

    if include_bbq:
        # Load every BBQ category's parquet, filter to disambiguated rows,
        # concatenate, then shuffle+limit. Sample-weighted aggregation
        # (correct/total) downstream gives bigger categories more weight.
        bbq_files = {
            cat: BBQ_PARQUET_URL_TMPL.format(category=cat) for cat in BBQ_CATEGORIES
        }
        ds = load_dataset("parquet", data_files=bbq_files, split=None)
        # `data_files` as a dict yields a DatasetDict keyed by category; concat them.
        ds = concatenate_datasets([ds[cat] for cat in BBQ_CATEGORIES])
        ds = ds.filter(lambda row: row["context_condition"] == "disambig")
        if bbq_limit is not None:
            ds = ds.shuffle(seed=seed).select(range(min(bbq_limit, len(ds))))
        for row in ds:
            options = [row["ans0"], row["ans1"], row["ans2"]]
            option_letters = ["A", "B", "C"]
            correct_letter = option_letters[row["label"]]
            questions.append(RobustnessQuestion(
                question_id=f"bbq_{row['category']}_{row['example_id']}",
                question_text=f"{row['context']}\n{row['question']}",
                options=options,
                option_letters=option_letters,
                correct_letter=correct_letter,
                source="bbq",
                metadata={"category": row["category"], "question_polarity": row["question_polarity"]},
            ))

    return questions


def format_mc_prompt(question: RobustnessQuestion) -> str:
    prompt = f"{question.question_text}\n\n"
    for letter, option in zip(question.option_letters, question.options):
        prompt += f"{letter}. {option}\n"
    prompt += "\nAnswer with just the letter of the correct answer."
    return prompt


def parse_answer_letter(response: str) -> str | None:
    response = response.strip()
    # Try: first character is a letter
    if response and response[0] in string.ascii_uppercase:
        return response[0]
    # Try: find a standalone letter
    match = re.search(r'\b([A-J])\b', response)
    if match:
        return match.group(1)
    return None
