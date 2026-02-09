# generate_noise_prob_data.py
import os, re, json, time, ast, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ---------------- utils: memory ----------------
# def load_memory(mem_path):
#     mem = {}
#     if not os.path.exists(mem_path):
#         return mem
#     with open(mem_path, "r", encoding="utf-8") as f:
#         for line in f:
#             try:
#                 rec = json.loads(line)
#                 k = rec.get("key")
#                 if k is None:
#                     continue
#                 mem.setdefault(k, []).append(rec)
#             except:
#                 pass
#     return mem
#
# def append_memory(mem_path, key, prompt, meta):
#     rec = {
#         "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "key": key,
#         "prompt": prompt,
#         "meta": meta,
#     }
#     with open(mem_path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(rec, ensure_ascii=False) + "\n")
#
# def make_key(user_id, item_ids):
#     # 稳妥：用户+序列长度+hash
#     return f"u{user_id}_len{len(item_ids)}_{abs(hash(tuple(item_ids)))%10**8}"

# -------------- prompts ----------------
def generate_noise_prompt_with_history(item_ids, id2name, history_rows=None, memory_entries=None):

    instruction = (
        "You will analyze a list of item titles and identify titles that do NOT align with the user's main interests. "
        "ONLY output the suspected noise titles. If there is no noise, output 'No'. "
        "Do NOT output any suggestions."
    )
    hist = "User Interaction Sequence: " + ", ".join(f'"{id2name[i]}"' for i in item_ids) + "\n"
    extras = []
    if history_rows:
        extras.append("Initial context:")
        if history_rows.get("noise_items"):
            extras.append("- Previously flagged (noisy?) items: " + ", ".join(f'"{t}"' for t in history_rows["noise_items"]))
        if history_rows.get("suggest_item_titles"):
            extras.append("- Old suggestions (ignore them): " + ", ".join(f'"{t}"' for t in history_rows["suggest_item_titles"]))
    if history_rows and "memory_log" in history_rows:
        lines = [l.strip() for l in str(history_rows["memory_log"]).split("\n") if l.strip()]
        if lines:
            extras.append("Previous attempts (considered unreliable):")
            extras += [f"- {l}" for l in lines[-3:]]  # 取最近3条

    extra_text = ("\n".join(extras) + "\n") if extras else ""
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{hist}{extra_text}"
        "### Response:\nNoise Items: '"
    )

def get_first_token_ids(tokenizer, titles):
    tokenizer.padding_side = "right"
    token_ids = tokenizer(titles, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    first_token_ids = token_ids[:, 1].tolist()  # BOS 后第一个 token
    tokenizer.padding_side = "left"
    return first_token_ids

def get_noise_items_prob(first_token_ids_batch, logits):
    logits = logits[:, -1, :]                 # [B, V]
    log_probs = F.log_softmax(logits, dim=-1) # [B, V]
    noise_prob = []
    for i, ids in enumerate(first_token_ids_batch):
        idx = torch.tensor(ids, device=log_probs.device)
        row = log_probs[i, idx]
        noise_prob.append(row.detach().cpu().tolist())
    return noise_prob

def main(
    denoise_model_path: str,
    noise_dataset_path: str,
    denoise_data: str,
    sample: int = -1,
    batch_size: int = 4,
    base_model: str = "",
):
    accelerator = Accelerator()

    save_denoise_data_path = noise_dataset_path  # 原地更新！不要改目录名

    if accelerator.is_main_process:
        os.makedirs(save_denoise_data_path, exist_ok=True)
        for name in ["id2name.json", "id2name4Rec.json", "item_embedding.pt"]:
            src = os.path.join(noise_dataset_path, name)
            if os.path.exists(src):
                dst = os.path.join(save_denoise_data_path, name)
                if not os.path.exists(dst):
                    try:
                        import shutil; shutil.copy(src, dst)
                    except: pass

    id2name_path = os.path.join(save_denoise_data_path, "id2name4Rec.json")
    with open(id2name_path, "r") as f:
        id2name = {int(k): v for k, v in json.load(f).items()}

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)


    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.special_tokens_map_reverse.popitem()[0]

    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    model = PeftModel.from_pretrained(model, denoise_model_path, torch_dtype=torch.bfloat16)
    model.merge_and_unload()
    model.eval()

    all_titles = list(id2name.values())
    first_token_table = get_first_token_ids(tokenizer, all_titles)
    id2first = dict(zip(id2name.keys(), first_token_table))

    data_path = os.path.join(noise_dataset_path, f"{denoise_data if sample==-1 else f'{denoise_data}_{sample}'}.csv")
    df = pd.read_csv(data_path)
    for col in ["item_ids","mask","item_titles","replaced_item_titles","noise_prob",
                "noise_items","noise_items_prob","suggest_item_titles","suggest_item_ids"]:
        if col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)
            except: pass

    item_ids_list = df["item_ids"].tolist()
    user_ids_list = df["user_id"].tolist()
    if "test" in denoise_data:
        item_ids_list = [ids[:-1] for ids in item_ids_list]

    prompts, first_ids_batches = [], []
    for i, (u, ids) in enumerate(tqdm(zip(user_ids_list, item_ids_list), total=len(item_ids_list))):
        hist_row = {}
        if "noise_items" in df.columns and isinstance(df["noise_items"].iloc[i], list) and len(
                df["noise_items"].iloc[i]) > 0:
            hist_row["noise_items"] = [id2name.get(x, str(x)) for x in df["noise_items"].iloc[i]]
        if "suggest_item_titles" in df.columns and isinstance(df["suggest_item_titles"].iloc[i], list) and len(
                df["suggest_item_titles"].iloc[i]) > 0:
            hist_row["suggest_item_titles"] = df["suggest_item_titles"].iloc[i]
        if "memory_log" in df.columns:
            hist_row["memory_log"] = df["memory_log"].iloc[i]

        prompt = generate_noise_prompt_with_history(ids, id2name, hist_row, None)
        prompts.append(prompt)
        first_ids_batches.append([id2first[x] for x in ids])

    accelerator.print(f"Total samples: {len(prompts)}")
    noise_prob_list = []

    NO_LOGPROB_TH = -10.0
    NO_DECAY = 0.9

    with accelerator.split_between_processes({"prompts": prompts, "first_ids": first_ids_batches}) as data:
        with torch.no_grad():
            for k in tqdm(range(0, len(data["prompts"]), batch_size)):
                p = data["prompts"][k:k + batch_size]
                fids = data["first_ids"][k:k + batch_size]
                inputs = tokenizer(p, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)

                outputs = model(**inputs)
                logits = outputs.logits  # [B, T, V]

                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # [B, V]

                for j in range(len(fids)):
                    idx = torch.tensor(fids[j], device=log_probs.device)
                    row_lp = log_probs[j, idx]  # [seq_len]
                    max_lp = row_lp.max().item()  # 标量

                    if max_lp < NO_LOGPROB_TH:

                        noise_prob_list.append(("NO", row_lp.detach().cpu().tolist()))
                    else:
                        noise_prob_list.append(("OK", row_lp.detach().cpu().tolist()))
    noise_prob_list = gather_object(noise_prob_list)
    assert len(noise_prob_list) == len(item_ids_list)


    new_probs_only = []
    if "noise_prob" in df.columns:
        old_probs_col = df["noise_prob"].tolist()
    else:
        old_probs_col = [None] * len(noise_prob_list)

    for idx, (flag, new_lp_list) in enumerate(noise_prob_list):
        old_lp = old_probs_col[idx]
        if isinstance(old_lp, list) and len(old_lp) == len(new_lp_list):
            old_lp = np.array(old_lp, dtype=float)
            new_lp = np.array(new_lp_list, dtype=float)
            if flag == "NO":
                merged = (NO_DECAY * old_lp).tolist()
            else:
                merged = new_lp.tolist()
        else:
            if flag == "NO":
                merged = (np.array(new_lp_list, dtype=float) * 0 - 1e9).tolist()
            else:
                merged = list(map(float, new_lp_list))
        new_probs_only.append(merged)

    df["noise_prob"] = new_probs_only

    topk = 3
    mem_col = "memory_log"
    if mem_col not in df.columns:
        df[mem_col] = [""] * len(df)

    for i, (ids, tup) in enumerate(zip(item_ids_list, noise_prob_list)):
        flag, probs = tup
        order = np.argsort(probs)[::-1]
        idx_sorted = order[:topk]
        cand_titles = [id2name[ids[j]] for j in idx_sorted if j < len(ids)]
        summary = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] suspects: " + (
            ", ".join(cand_titles) if (flag != "NO" and cand_titles) else "No"
        )

        prev = str(df.at[i, mem_col]) if isinstance(df.at[i, mem_col], str) else ""
        lines = [l.strip() for l in prev.split("\n") if l.strip()]
        if summary not in lines:
            lines.append(summary)
        df.at[i, mem_col] = "\n".join(lines[-5:])

    save_csv = os.path.join(
        save_denoise_data_path,
        f"{denoise_data if sample == -1 else f'{denoise_data}_{sample}'}.csv"
    )
    if accelerator.is_main_process:
        df.to_csv(save_csv, index=False)
        print("Saved:", save_csv, "with updated memory_log column.")

def run_generate_noise_prob_data(
    denoise_model_path,
    noise_dataset_path,
    denoise_data="train",
    sample=-1,
    batch_size=1,
    base_model="/home/lisijia/llama3-3b/",
):

    main(
        denoise_model_path=denoise_model_path,
        noise_dataset_path=noise_dataset_path,
        denoise_data=denoise_data,
        sample=sample,
        batch_size=batch_size,
        base_model=base_model,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoise_model_path", type=str, required=True)
    ap.add_argument("--noise_dataset_path", type=str, required=True)
    ap.add_argument("--denoise_data", type=str, default="train")
    ap.add_argument("--sample", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--base_model", type=str, default="")
    args = ap.parse_args()
    run_generate_noise_prob_data(**vars(args))