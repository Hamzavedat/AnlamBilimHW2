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
BASE_SEED = 42
TEMPERATURE = 0.8
TOP_P = 0.9

PROMPT_TEMPLATES = [
    "Explain the central idea of {topic} in plain language.",
    "Why is {topic} important in practice?",
    "What is a common misconception about {topic}, and how would you correct it?",
    "Compare {topic} with a closely related idea and highlight the key difference.",
    "Describe one real-world example that helps explain {topic}.",
    "What usually causes {topic} to happen?",
    "What are the short-term and long-term effects of {topic}?",
    "How would you teach {topic} to a complete beginner?",
    "What trade-offs or limitations are associated with {topic}?",
    "Why do experts continue to study or debate {topic}?",
]

CATEGORY_TOPICS = {
    "History": [
        "the fall of Constantinople",
        "the French Revolution",
        "the Industrial Revolution",
        "the Silk Road",
        "the Meiji Restoration",
        "the decline of the Roman Republic",
        "Ottoman modernization",
        "Cold War nuclear deterrence",
        "the printing press",
        "decolonization after World War II",
    ],
    "Physics": [
        "entropy",
        "special relativity",
        "wave-particle duality",
        "superconductivity",
        "black holes",
        "resonance",
        "conservation of momentum",
        "semiconductors",
        "nuclear fusion",
        "turbulence",
    ],
    "Biology": [
        "natural selection",
        "immune memory",
        "CRISPR gene editing",
        "photosynthesis",
        "the human microbiome",
        "sleep cycles",
        "protein folding",
        "vaccines",
        "cellular respiration",
        "hormonal regulation",
    ],
    "Technology": [
        "distributed systems",
        "neural networks",
        "encryption",
        "battery storage",
        "compilers",
        "internet routing",
        "cloud computing",
        "robotics",
        "database indexing",
        "recommendation systems",
    ],
    "Economics": [
        "inflation",
        "comparative advantage",
        "opportunity cost",
        "market failure",
        "game theory",
        "central banking",
        "price elasticity",
        "unemployment",
        "public goods",
        "supply chain shocks",
    ],
    "Psychology": [
        "confirmation bias",
        "working memory",
        "habit formation",
        "cognitive dissonance",
        "social anxiety",
        "operant conditioning",
        "inattentional blindness",
        "attachment styles",
        "the placebo effect",
        "decision fatigue",
    ],
    "Mathematics": [
        "prime numbers",
        "derivatives",
        "Bayesian inference",
        "graph theory",
        "eigenvalues",
        "limits",
        "linear programming",
        "fractals",
        "modular arithmetic",
        "probability distributions",
    ],
    "Culture": [
        "folklore transmission",
        "language change",
        "ritual symbolism",
        "pop music trends",
        "meme culture",
        "fashion cycles",
        "food taboos",
        "oral storytelling",
        "festival traditions",
        "translation choices",
    ],
    "Art": [
        "chiaroscuro",
        "perspective drawing",
        "montage editing",
        "impressionism",
        "minimalism",
        "street photography",
        "sculpture casting",
        "color harmony",
        "jazz improvisation",
        "architectural ornament",
    ],
    "Sports": [
        "the offside rule",
        "interval training",
        "home-field advantage",
        "recovery science",
        "serve strategy in tennis",
        "pacing in marathons",
        "team pressing",
        "the biomechanics of jumping",
        "talent scouting",
        "nutrition periodization",
    ],
}

FULL_PROMPT_COUNT = len(CATEGORY_TOPICS) * len(PROMPT_TEMPLATES) * 10
NUM_SAMPLES = 1000
MAX_NEW_TOKENS = 50

RUN_NAME = "hw2_full_seed42"
RUN_DIR = os.path.join("runs", RUN_NAME)
TABLES_DIR = os.path.join(RUN_DIR, "tables")
TRAJECTORIES_DIR = os.path.join(RUN_DIR, "trajectories")


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

def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def build_prompt(question_text):
    return normalize_text(question_text)


def build_prompt_plan():
    full_plan = []
    sample_index = 0

    for category, topics in CATEGORY_TOPICS.items():
        if len(topics) != 10:
            raise ValueError(f"Category {category} must have exactly 10 topics.")

        for topic_index, topic in enumerate(topics):
            topic_slug = f"{category.lower().replace(' ', '_')}_topic_{topic_index:02d}"

            for template_index, template in enumerate(PROMPT_TEMPLATES):
                question_text = template.format(topic=topic)
                prompt = build_prompt(question_text)
                question_id = f"{topic_slug}_template_{template_index:02d}"

                full_plan.append(
                    {
                        "sample_index": sample_index,
                        "category": category,
                        "source_split": "synthetic",
                        "template_id": f"template_{template_index:02d}",
                        "prompt_id": topic_slug,
                        "topic_id": topic_slug,
                        "prompt_uid": question_id,
                        "prompt": prompt,
                        "generation_mode": "greedy_argmax",
                        "q_id": question_id,
                        "title": question_text,
                        "selftext": "",
                        "subreddit": "",
                        "answer_ids_json": "[]",
                        "answer_texts_json": "[]",
                        "answer_scores_json": "[]",
                        "top_answer_text": "",
                        "top_answer_score": None,
                    }
                )
                sample_index += 1

    full_plan.sort(key=lambda row: row["sample_index"])
    if len(full_plan) != FULL_PROMPT_COUNT:
        raise ValueError(f"Expected {FULL_PROMPT_COUNT} prompts, got {len(full_plan)}.")

    if NUM_SAMPLES > len(full_plan):
        raise ValueError(f"NUM_SAMPLES={NUM_SAMPLES} exceeds full prompt bank size.")

    if NUM_SAMPLES == len(full_plan):
        plan = full_plan
    elif NUM_SAMPLES % len(CATEGORY_TOPICS) == 0:
        per_category = NUM_SAMPLES // len(CATEGORY_TOPICS)
        plan = []
        for category in CATEGORY_TOPICS:
            category_rows = [row for row in full_plan if row["category"] == category]
            plan.extend(category_rows[:per_category])
    else:
        plan = full_plan[:NUM_SAMPLES]

    prompt_uid_count = len({row["prompt_uid"] for row in plan})
    prompt_text_count = len({row["prompt"] for row in plan})
    if prompt_uid_count != len(plan) or prompt_text_count != len(plan):
        raise ValueError("Prompt bank is not fully unique.")
    return plan


def save_prompt_bank(plan):
    prompt_bank = pd.DataFrame(plan)
    prompt_bank_path = os.path.join(TABLES_DIR, "prompt_bank.csv")
    prompt_bank.to_csv(prompt_bank_path, index=False)
    print(f"Saved {prompt_bank_path}")


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_next_token(logits):
    scaled_logits = logits / TEMPERATURE
    probs = torch.softmax(scaled_logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > TOP_P
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    filtered_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    sampled_positions = torch.multinomial(filtered_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_positions)


def generate_one(model, tokenizer, prompt, device, seed):
    seed_everything(seed)
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

    prompt_token_count = int(input_ids.shape[1])
    full_ids = current_ids[0].detach().cpu().numpy()
    completion_ids = full_ids[prompt_token_count:]
    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    trajectory = np.asarray(trajectory, dtype=np.float32)

    return generated_text, full_text, trajectory


def check_reproducibility(model, tokenizer, plan, device):
    print("Checking reproducibility on first 2 samples...")
    for sample in plan[:2]:
        seed = BASE_SEED + sample["sample_index"]
        text_a, full_a, traj_a = generate_one(model, tokenizer, sample["prompt"], device, seed)
        text_b, full_b, traj_b = generate_one(model, tokenizer, sample["prompt"], device, seed)
        if text_a != text_b or full_a != full_b or not np.array_equal(traj_a, traj_b):
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
        seed = BASE_SEED + sample["sample_index"]
        generated_text, full_text, trajectory = generate_one(
            model,
            tokenizer,
            sample["prompt"],
            device,
            seed,
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
                "seed_used": seed,
                "prompt": sample["prompt"],
                "category": sample["category"],
                "source_split": sample["source_split"],
                "template_id": sample["template_id"],
                "prompt_id": sample["prompt_id"],
                "topic_id": sample["topic_id"],
                "prompt_uid": sample["prompt_uid"],
                "generation_mode": "top_p_sampling",
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
                "full_text": full_text,
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
    save_prompt_bank(plan)
    device = get_device()

    print(f"Device: {device}")
    print(f"Run folder: {RUN_DIR}")

    for model_name in MODELS:
        collect_for_model(model_name, plan, device)

    print("Data collection finished.")


if __name__ == "__main__":
    main()
