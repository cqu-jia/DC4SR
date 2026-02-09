"""
python main.py --dataset Movie_denoise_user0.0_5000_2000
"""

import json
import os
import random
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"

import glob
import pdb
import time
import pandas as pd
import torch

from tqdm import tqdm
import copy
import nni

from model import SASRec
import util.world
from util.logger import CompleteLogger
import util.utils
from os.path import join
from tensorboardX import SummaryWriter
from pprint import pprint

from util.dataloader import Seq_dataset, Data_Pro
from torch.utils.data import DataLoader
from util.semantic_prior import run_generate_noise_prob_data

import subprocess, sys, shlex, ast

if not "NNI_PLATFORM" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = world.config["cuda"]
else:
    optimized_params = nni.get_next_parameter()
    world.config.update(optimized_params)

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

denoise_category = world.config["denoise_category"]
if denoise_category in ("all", "remove"):
    train_denoise_flag = 1
    test_denoise_flag = 1
elif denoise_category == "train":
    train_denoise_flag = 1
    test_denoise_flag = 0
elif denoise_category == "test":
    train_denoise_flag = 0
    test_denoise_flag = 1
elif denoise_category == "none":
    train_denoise_flag = 0
    test_denoise_flag = 0

dataroot = os.path.join("./data", world.config["dataset"])
logroot = os.path.join("./log", world.config["dataset"], f"denoise_{denoise_category}")

os.makedirs(logroot, exist_ok=True)
def update_memory_in_df(train_df, ridx, summary):
    if "memory_log" not in train_df.columns:
        train_df["memory_log"] = [""] * len(train_df)

    prev = str(train_df.at[ridx, "memory_log"]) if isinstance(train_df.at[ridx, "memory_log"], str) else ""
    if summary not in prev:
        new_val = (prev + "\n" + summary).strip()
        train_df.at[ridx, "memory_log"] = new_val

def make_key_from_dfrow(df, row_idx):
    r = df.iloc[row_idx]
    uid = int(r["user_id"])
    ids = r["item_ids"]
    return f"u{uid}_len{len(ids)}_{abs(hash(tuple(ids)))%10**8}"


if "NNI_PLATFORM" in os.environ:
    save_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], "tensorboard")
    w: SummaryWriter = SummaryWriter(save_dir)
else:
    save_dir = logroot
    save_dir = join(logroot, time.strftime("%m-%d-%Hh%Mm%Ss"))
    i = 0
    while os.path.exists(save_dir):
        new_save_dir = save_dir + str(i)
        i += 1
        save_dir = new_save_dir
    w: SummaryWriter = SummaryWriter(save_dir)
    logger = CompleteLogger(root=save_dir)

pprint(world.config)
w.add_text("Config", str(world.config), 0)


def write_tensorboard_metric(wirter: SummaryWriter, metric, epoch, category="Valid"):
    for key, value in metric.items():
        wirter.add_scalar(f"{category}/{key}".replace("@", "_"), value, epoch)


if __name__ == "__main__":
    data_pro = Data_Pro(dataroot, train_denoise_flag, test_denoise_flag)
    train_df, test_df = data_pro.get_data_df()
    item_num = data_pro.item_num

    batch_size = world.config["batchsize"]
    train_dataset, test_dataset = (Seq_dataset(train_df), Seq_dataset(test_df))
    train_dataloader, test_dataloader = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16),
    )

    model = SASRec(item_num).cuda()

    state_dict_path = world.config["state_dict_path"]
    if state_dict_path is not None:
        model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=world.config["lr"], weight_decay=world.config["weight_decay"]
    )

    start_total = time.time()
    if world.config["state_dict_path"] is None:
        patience = 0
        best_NDCG = 0
        step = 0
        best_model_dict = None
        unreliable_pool = []  # (row_idx, score)

        for epoch in range(world.config["num_epochs"]):
            start_time_epoch = time.time()
            print("===================================")
            print("Start Training Epoch {}".format(epoch))

            # Train
            model.train()
            for batch in tqdm(train_dataloader):
                if isinstance(batch, (list, tuple)) and len(batch) == 4:
                    seq_items_id, noise_mask, noise_prior, row_idx = batch
                else:
                    row_idx = None
                    if len(batch) == 3:
                        seq_items_id, noise_mask, noise_prior = batch
                    elif len(batch) == 2:
                        seq_items_id, noise_mask = batch;
                        noise_prior = None
                    else:
                        seq_items_id = batch;
                        noise_mask = None;
                        noise_prior = None

                seq, pos, neg = utils.get_negative_items(seq_items_id, item_num)
                # out = model(seq, pos, neg, noise_mask=noise_mask)
                hist_mask = noise_mask[:, :-1] if noise_mask is not None else None
                hist_prior = noise_prior[:, :-1] if noise_prior is not None else None
                out = model(seq, pos, neg, noise_mask=hist_mask, noise_prior=hist_prior)

                if len(out) == 4:
                    pos_logits, neg_logits, _, aux_loss = out
                else:
                    pos_logits, neg_logits, _ = out
                    aux_loss = None

                pos_labels = torch.ones(pos_logits.shape, device="cuda")
                neg_labels = torch.zeros(neg_logits.shape, device="cuda")

                indices = (pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

                if aux_loss is not None:
                    loss = loss + aux_loss
                    w.add_scalar("Train/AuxLoss", aux_loss.item(), step)

                adam_optimizer.zero_grad()
                loss.backward()
                adam_optimizer.step()

                w.add_scalar(f"Train/Loss", loss, step)
                step += 1


                try:
                    with torch.no_grad():
                        feats = model.log2feats(seq.cuda(), noise_mask=hist_mask, noise_prior=hist_prior)  # [B,L,C]
                        p_hat = torch.sigmoid(model.noise_scorer(feats)).squeeze(-1)  # [B,L]
                        valid = (seq.cuda()[:, :-1] != 0)  # [B, L-1]
                        mean_score = (p_hat[:, :-1] * valid.float()).sum(dim=1) / (valid.float().sum(dim=1) + 1e-12)

                        topk = 3
                        if row_idx is not None:
                            for b in range(seq.size(0)):
                                ridx = int(row_idx[b])
                                vlen = int(valid[b].sum().item())
                                if vlen <= 0:
                                    continue
                                scores_b = p_hat[b, :vlen]  # [vlen]
                                topv, topi = torch.topk(scores_b, k=min(topk, vlen))
                                titles = train_df.iloc[ridx]["item_titles"]
                                cand_titles = [str(titles[int(j)]) for j in topi.cpu().tolist()]
                                unreliable_pool.append((ridx, float(mean_score[b].cpu().item()), cand_titles))
                except Exception as e:
                    pass

                try:
                    with torch.no_grad():
                        feats = model.log2feats(seq.cuda(),
                                                noise_mask=hist_mask,
                                                noise_prior=hist_prior)  # [B,L,C]
                        p_hat = torch.sigmoid(model.noise_scorer(feats)).squeeze(-1)  # [B,L]

                        valid = (seq.cuda()[:, :-1] != 0)  # [B,L-1]

                        if hist_prior is not None:
                            prior_seq = hist_prior.to(p_hat.device)  # [B,L-1]
                        else:
                            prior_seq = None

                        topk = 3
                        if row_idx is not None:
                            for b in range(seq.size(0)):
                                ridx = int(row_idx[b])
                                vlen = int(valid[b].sum().item())
                                if vlen <= 0:
                                    continue

                                p_b = p_hat[b, :vlen]  # [vlen]
                                if prior_seq is not None:
                                    prior_b = prior_seq[b, :vlen]  # [vlen]
                                    diff_b = torch.abs(p_b - prior_b)
                                    unreliable_score = diff_b.mean()
                                else:
                                    unreliable_score = p_b.mean()

                                topv, topi = torch.topk(p_b, k=min(topk, vlen))
                                titles = train_df.iloc[ridx]["item_titles"]
                                cand_titles = [str(titles[int(j)]) for j in topi.cpu().tolist()]

                                unreliable_pool.append(
                                    (ridx,
                                     float(unreliable_score.cpu().item()),
                                     cand_titles)
                                )
                except Exception as e:
                    pass

            print("Time for one epoch", time.time() - start_time_epoch)

            if (epoch + 1) % world.config["llm_refresh_every"] == 0:
                print(f"[LLM] epoch {epoch + 1}: write memory + update noise_prob ...")

                model.cpu()
                torch.cuda.empty_cache()
                if len(unreliable_pool) > 0:
                    unreliable_pool.sort(key=lambda x: x[1], reverse=True)
                    topN = unreliable_pool[:200]
                    for ridx, sc, cand_titles in topN:
                        summary = f"[Epoch {epoch + 1}] unreliable (mean={sc:.3f}) → " + \
                                  (", ".join(f'"{t}"' for t in cand_titles) if cand_titles else "No")
                        update_memory_in_df(train_df, ridx, summary)
                    unreliable_pool = []

                run_generate_noise_prob_data(
                    denoise_model_path=world.config["denoise_model_path"],
                    noise_dataset_path=dataroot,
                    denoise_data="train",
                    base_model=world.config["base_model"],
                )

                train_csv_path = os.path.join(dataroot, "train.csv")
                train_df = pd.read_csv(train_csv_path)

                list_like_cols = ["item_ids", "mask", "item_titles", "replaced_item_titles",
                                  "noise_prob", "noise_items", "noise_items_prob",
                                  "suggest_item_titles", "suggest_item_ids"]
                for c in list_like_cols:
                    if c in train_df.columns:
                        def _maybe_eval(x):
                            if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
                                try:
                                    return ast.literal_eval(x)
                                except:
                                    return x
                            return x


                        train_df[c] = train_df[c].apply(_maybe_eval)

                if "memory_flag" not in train_df.columns:
                    train_df["memory_flag"] = 0
                train_df["memory_flag"] = train_df["memory_flag"].astype(int) + 1

                train_df.to_csv(train_csv_path, index=False)

                train_dataset = Seq_dataset(train_df)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

                model.cuda()
                for state in adam_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()


            # Test
            if epoch % 1 == 0:
                model.eval()

                t_test = utils.evaluate(model, test_dataloader, len(test_dataset))
                write_tensorboard_metric(w, t_test, epoch, "Test")
                print("Test\n", t_test)

                if "NNI_PLATFORM" in os.environ:
                    metric = {
                        "default": t_test["NDCG@20"],
                        "ndcg@10": t_test["NDCG@10"],
                        "hit@20": t_test["HR@20"],
                        "hit@10": t_test["HR@10"],
                        "hit@50": t_test["HR@50"],
                        "ndcg@50": t_test["NDCG@50"],
                    }
                    nni.report_intermediate_result(metric)

                if t_test["NDCG@20"] > best_NDCG:
                    best_NDCG = t_test["NDCG@20"]
                    patience = 0
                    best_model_dict = copy.deepcopy(model.state_dict())
                else:
                    patience += 1
                    print("Patience{}/10".format(patience))
                    if patience >= 10:
                        break

        if "NNI_PLATFORM" not in os.environ:
            torch.save(best_model_dict, os.path.join(save_dir, "best_model.pth"))
            model.save_item_embeddings(os.path.join(save_dir, "item_embeddings.pth"))
        model.load_state_dict(best_model_dict)

    model.eval()
    t_test = utils.evaluate(model, test_dataloader, len(test_dataset))
    print("The Finial Test Metric for Model is following: ")
    print(t_test)

    if "NNI_PLATFORM" in os.environ:
        metric = {
            "default": t_test["NDCG@20"],
            "ndcg@10": t_test["NDCG@10"],
            "hit@20": t_test["HR@20"],
            "hit@10": t_test["HR@10"],
            "hit@50": t_test["HR@50"],
            "ndcg@50": t_test["NDCG@50"],
        }
        nni.report_final_result(metric)

    w.close()
    print("Total time:{}".format(time.time() - start_total))
    print("Training Done!")


