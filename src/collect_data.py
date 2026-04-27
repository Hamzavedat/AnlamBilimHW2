import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = ["gpt2", "EleutherAI/pythia-70m"]
CATEGORIES = ["History", "Science", "Technology", "Art", "Sports"]

NUM_SAMPLES = 1000
MAX_NEW_TOKENS = 50
BASE_SEED = 42
TEMPERATURE = 0.7
TOP_P = 0.9

RUN_NAME = "hw2_full_seed42"
RUN_DIR = os.path.join("runs", RUN_NAME)
DATA_DIR = os.path.join(RUN_DIR, "data")

PROMPT_TEMPLATES = {
    "History": [
        "Write two sentences about a turning point in {topic}.",
        "Explain how {topic} influenced later societies.",
        "Summarize the causes and effects of {topic}.",
        "Describe one debate historians still have about {topic}.",
        "Compare two perspectives on {topic} in a short paragraph.",
    ],
    "Science": [
        "Give a concise explanation of {topic} for undergraduate students.",
        "State a key principle behind {topic} and why it matters.",
        "Describe one experiment commonly used to study {topic}.",
        "Explain a misconception about {topic} and correct it.",
        "Write a short scientific note on the significance of {topic}.",
    ],
    "Technology": [
        "Outline the core idea of {topic} and a practical use case.",
        "Discuss one benefit and one limitation of {topic}.",
        "Explain how {topic} may evolve over the next decade.",
        "Provide a short technical overview of {topic}.",
        "Summarize ethical risks associated with {topic}.",
    ],
    "Art": [
        "Write a short critique focusing on style in {topic}.",
        "Describe how {topic} reflects its cultural context.",
        "Compare form and meaning in works related to {topic}.",
        "Explain how audiences might interpret {topic} differently.",
        "Provide a compact analysis of techniques used in {topic}.",
    ],
    "Sports": [
        "Summarize the historical development of {topic}.",
        "Explain a tactical principle central to {topic}.",
        "Describe how training methods changed in {topic}.",
        "Write a short analysis of competition dynamics in {topic}.",
        "Discuss one rule change that affected {topic}.",
    ],
}

TOPIC_BANK = {
    "History": [
        "the fall of Constantinople",
        "the French Revolution",
        "the Silk Road",
        "the Meiji Restoration",
        "postwar reconstruction in Europe",
    ],
    "Science": [
        "quantum entanglement",
        "plate tectonics",
        "natural selection",
        "CRISPR gene editing",
        "gravitational waves",
    ],
    "Technology": [
        "federated learning",
        "edge computing",
        "software-defined networking",
        "large language models",
        "battery storage systems",
    ],
    "Art": [
        "Renaissance portraiture",
        "Japanese woodblock prints",
        "modernist architecture",
        "impressionist landscapes",
        "documentary photography",
    ],
    "Sports": [
        "Olympic sprinting",
        "possession football",
        "marathon pacing",
        "professional tennis",
        "women's volleyball",
    ],
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_folders():
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)


def build_prompt_plan():
    plan = []
    for sample_index in range(NUM_SAMPLES):
        category = CATEGORIES[sample_index % len(CATEGORIES)]
        local_seed = BASE_SEED + sample_index
        rng = np.random.default_rng(local_seed)

        template_index = int(rng.integers(0, len(PROMPT_TEMPLATES[category])))
        topic_index = int(rng.integers(0, len(TOPIC_BANK[category])))

        prompt = PROMPT_TEMPLATES[category][template_index].format(
            topic=TOPIC_BANK[category][topic_index]
        )

        plan.append(
            {
                "sample_index": sample_index,
                "category": category,
                "template_id": f"{category.lower()}_t{template_index}",
                "prompt_id": f"{category.lower()}_t{template_index}",
                "topic_id": f"{category.lower()}_topic{topic_index}",
                "prompt": prompt,
                "seed_used": local_seed,
            }
        )
    return plan


def sample_next_token(logits):
    probs = torch.softmax(logits / TEMPERATURE, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative_probs > TOP_P
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sampled_index = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_index)


def generate_one(model, tokenizer, prompt, seed, device):
    set_seed(seed)

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
        text_a, traj_a = generate_one(model, tokenizer, sample["prompt"], sample["seed_used"], device)
        text_b, traj_b = generate_one(model, tokenizer, sample["prompt"], sample["seed_used"], device)
        if text_a != text_b or not np.array_equal(traj_a, traj_b):
            raise RuntimeError("Reproducibility check failed.")
    print("Reproducibility check passed.")


def collect_for_model(model_name, plan, device):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
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
            sample["seed_used"],
            device,
        )

        traj_id = f"{safe_name}_{sample['sample_index']:05d}"
        traj_path = os.path.abspath(os.path.join(DATA_DIR, f"{traj_id}.npy"))
        np.save(traj_path, trajectory)

        rows.append(
            {
                "run_name": RUN_NAME,
                "traj_id": traj_id,
                "model_name": model_name,
                "sample_index": sample["sample_index"],
                "prompt": sample["prompt"],
                "category": sample["category"],
                "template_id": sample["template_id"],
                "prompt_id": sample["prompt_id"],
                "topic_id": sample["topic_id"],
                "seed_used": sample["seed_used"],
                "generated_text": generated_text,
                "num_tokens": int(trajectory.shape[0]),
                "trajectory_path": traj_path,
            }
        )

    metadata_path = os.path.join(DATA_DIR, f"{safe_name}_metadata.csv")
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
