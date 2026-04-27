import json
import gzip
import os
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = ["gpt2", "EleutherAI/pythia-70m"]
MODEL_NOTES = {
    "gpt2": "larger hidden size and classic GPT-2 family baseline",
    "EleutherAI/pythia-70m": "smaller GPT-NeoX style baseline for contrast",
}

SELECTED_CATEGORY_SPLITS = {
    "Biology": "train",
    "Chemistry": "train",
    "Culture": "valid",
    "Earth Science": "train",
    "Economics": "train",
    "Mathematics": "train",
    "Other": "train",
    "Physics": "train",
    "Psychology": "train",
    "Technology": "train",
}

SAMPLES_PER_CATEGORY = 100
NUM_SAMPLES = len(SELECTED_CATEGORY_SPLITS) * SAMPLES_PER_CATEGORY
MAX_NEW_TOKENS = 50

RUN_NAME = "hw2_full_seed42"
RUN_DIR = os.path.join("runs", RUN_NAME)
TABLES_DIR = os.path.join(RUN_DIR, "tables")
TRAJECTORIES_DIR = os.path.join(RUN_DIR, "trajectories")
DATASET_DIR = "eli5_category"
SPLIT_CANDIDATES = {
    "train": [
        os.path.join(DATASET_DIR, "train.jsonl"),
        os.path.join(DATASET_DIR, "train.json.gz"),
    ],
    "valid": [
        os.path.join(DATASET_DIR, "valid.jsonl"),
        os.path.join(DATASET_DIR, "valid.json.gz"),
    ],
    "valid2": [
        os.path.join(DATASET_DIR, "valid2.jsonl"),
        os.path.join(DATASET_DIR, "valid2.json.gz"),
    ],
    "test": [
        os.path.join(DATASET_DIR, "test.jsonl"),
        os.path.join(DATASET_DIR, "test.json.gz"),
    ],
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_folders():
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(TRAJECTORIES_DIR, exist_ok=True)


def resolve_split_path(split_name):
    for candidate in SPLIT_CANDIDATES[split_name]:
        if os.path.exists(candidate):
            return candidate
    expected = ", ".join(SPLIT_CANDIDATES[split_name])
    raise FileNotFoundError(f"Dataset split not found for {split_name}. Looked for: {expected}")


def load_records(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset split not found: {path}")

    records = []
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        first_non_space = ""
        while not first_non_space:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_non_space = ch
        f.seek(0)

        if first_non_space == "[":
            payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a list in {path}")
            records.extend(payload)
        else:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def build_prompt(title, selftext):
    title = normalize_text(title)
    selftext = normalize_text(selftext)
    if selftext:
        return f"Question: {title}\n\nDetails: {selftext}\n\nAnswer clearly and concisely."
    return f"Question: {title}\n\nAnswer clearly and concisely."


def extract_answer_fields(record):
    answers = record.get("answers") or {}
    answer_ids = answers.get("a_id") or []
    answer_texts = answers.get("text") or []
    answer_scores = answers.get("score") or []

    top_answer_text = ""
    top_answer_score = None
    if answer_texts:
        best_index = 0
        if answer_scores:
            best_index = int(np.argmax(answer_scores))
        if best_index < len(answer_texts):
            top_answer_text = normalize_text(answer_texts[best_index])
        if best_index < len(answer_scores):
            top_answer_score = int(answer_scores[best_index])

    return {
        "answer_ids_json": json.dumps(answer_ids, ensure_ascii=False),
        "answer_texts_json": json.dumps(answer_texts, ensure_ascii=False),
        "answer_scores_json": json.dumps(answer_scores, ensure_ascii=False),
        "top_answer_text": top_answer_text,
        "top_answer_score": top_answer_score,
    }


def build_prompt_plan():
    split_records = {}
    for split_name in sorted(set(SELECTED_CATEGORY_SPLITS.values())):
        split_records[split_name] = load_records(resolve_split_path(split_name))

    plan = []
    sample_index = 0

    for category_index, (category, split_name) in enumerate(SELECTED_CATEGORY_SPLITS.items()):
        category_pool = [
            record
            for record in split_records[split_name]
            if normalize_text(record.get("category")) == category
        ]
        if len(category_pool) < SAMPLES_PER_CATEGORY:
            raise ValueError(
                f"Category {category} in split {split_name} has only {len(category_pool)} rows."
            )

        category_pool.sort(
            key=lambda record: (
                normalize_text(record.get("q_id")),
                normalize_text(record.get("title")),
            )
        )
        chosen_records = category_pool[:SAMPLES_PER_CATEGORY]

        for record in chosen_records:
            title = normalize_text(record.get("title"))
            selftext = normalize_text(record.get("selftext"))
            prompt = build_prompt(title, selftext)

            plan.append(
                {
                    "sample_index": sample_index,
                    "category": category,
                    "source_split": split_name,
                    "template_id": category.lower().replace(" ", "_"),
                    "prompt_id": normalize_text(record.get("q_id")) or f"{category}_{sample_index}",
                    "topic_id": normalize_text(record.get("q_id")) or f"{category}_{sample_index}",
                    "prompt": prompt,
                    "generation_mode": "greedy_argmax",
                    "q_id": normalize_text(record.get("q_id")),
                    "title": title,
                    "selftext": selftext,
                    "subreddit": normalize_text(record.get("subreddit")),
                    **extract_answer_fields(record),
                }
            )
            sample_index += 1

    plan.sort(key=lambda row: row["sample_index"])
    return plan


def sample_next_token(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)


def generate_one(model, tokenizer, prompt, device):
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)

    trajectory = []
    current_ids = input_ids

    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            outputs = model(
                input_ids=current_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            if step > 0:
                vector = (
                    outputs.hidden_states[-1][0, -1, :].detach().cpu().numpy().astype(np.float32)
                )
                trajectory.append(vector)

            next_token = sample_next_token(outputs.logits[:, -1, :])
            current_ids = torch.cat([current_ids, next_token], dim=1)

        final_outputs = model(
            input_ids=current_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        final_vector = (
            final_outputs.hidden_states[-1][0, -1, :].detach().cpu().numpy().astype(np.float32)
        )
        trajectory.append(final_vector)

    full_ids = current_ids[0].detach().cpu().numpy()
    generated_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    trajectory = np.asarray(trajectory, dtype=np.float32)

    return generated_text, trajectory


def check_reproducibility(model, tokenizer, plan, device):
    print("Checking reproducibility on first 2 samples...")
    for sample in plan[:2]:
        text_a, traj_a = generate_one(model, tokenizer, sample["prompt"], device)
        text_b, traj_b = generate_one(model, tokenizer, sample["prompt"], device)
        if text_a != text_b or not np.array_equal(traj_a, traj_b):
            raise RuntimeError("Reproducibility check failed.")
    print("Reproducibility check passed.")


def collect_for_model(model_name, plan, device):
    print(f"Loading {model_name}...")
    local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(device)
    model.eval()

    check_reproducibility(model, tokenizer, plan, device)

    rows = []
    safe_name = model_name.replace("/", "_")

    for sample in tqdm(plan, desc=model_name):
        generated_text, trajectory = generate_one(
            model,
            tokenizer,
            sample["prompt"],
            device,
        )

        traj_id = f"{safe_name}_{sample['sample_index']:05d}"
        traj_path = os.path.abspath(os.path.join(TRAJECTORIES_DIR, f"{traj_id}.npy"))
        np.save(traj_path, trajectory)

        rows.append(
            {
                "run_name": RUN_NAME,
                "traj_id": traj_id,
                "model_name": model_name,
                "model_note": MODEL_NOTES[model_name],
                "sample_index": sample["sample_index"],
                "prompt": sample["prompt"],
                "category": sample["category"],
                "source_split": sample["source_split"],
                "template_id": sample["template_id"],
                "prompt_id": sample["prompt_id"],
                "topic_id": sample["topic_id"],
                "generation_mode": sample["generation_mode"],
                "q_id": sample["q_id"],
                "title": sample["title"],
                "selftext": sample["selftext"],
                "subreddit": sample["subreddit"],
                "answer_ids_json": sample["answer_ids_json"],
                "answer_texts_json": sample["answer_texts_json"],
                "answer_scores_json": sample["answer_scores_json"],
                "top_answer_text": sample["top_answer_text"],
                "top_answer_score": sample["top_answer_score"],
                "generated_text": generated_text,
                "num_tokens": int(trajectory.shape[0]),
                "trajectory_path": traj_path,
            }
        )

    metadata_path = os.path.join(TABLES_DIR, f"{safe_name}_metadata.csv")
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    print(f"Saved {metadata_path}")


def main():
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    prepare_folders()
    plan = build_prompt_plan()
    device = get_device()

    print(f"Device: {device}")
    print(f"Run folder: {RUN_DIR}")

    for model_name in MODELS:
        collect_for_model(model_name, plan, device)

    print("Data collection finished.")


if __name__ == "__main__":
    main()
