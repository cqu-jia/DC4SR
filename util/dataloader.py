from collections import Counter
import copy
import pdb
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset
import world
import torch.nn.functional as F
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque


def read_file(path):
    df = pd.read_csv(path)
    df["item_ids"] = df["item_ids"].apply(ast.literal_eval)
    df["user_id"] = df["user_id"].astype(int)

    if "noise_items" in df.columns:
        df["noise_items"] = df["noise_items"].apply(ast.literal_eval)
    if "noise_items_prob" in df.columns:
        df["noise_items_prob"] = df["noise_items_prob"].apply(ast.literal_eval)
    if "suggest_item_ids" in df.columns:
        df["suggest_item_ids"] = df["suggest_item_ids"].apply(ast.literal_eval)
    if "noise_prob" in df.columns:
        df["noise_prob"] = df["noise_prob"].apply(ast.literal_eval)
    if "diverse_suggest_item_ids" in df.columns:
        df["diverse_suggest_item_ids"] = df["diverse_suggest_item_ids"].apply(ast.literal_eval)
    if "denoise_item_ids" in df.columns:
        df["denoise_item_ids"] = df["denoise_item_ids"].apply(ast.literal_eval)
        # 只有在去噪模式下才覆盖原序列；none 模式保持原始 item_ids
        if world.config.get("denoise_category", "none").lower() != "none":
            df["item_ids"] = df["denoise_item_ids"]

    return df

def _pos_prob_from_row(row):
    ids = row["item_ids"]; L = len(ids)
    p = row.get("noise_prob", [])
    if p:
        if len(p) == L - 1:
            return [0.0] + p
        if len(p) == L:
            return p
        return [0.0] * L
    from collections import defaultdict, deque
    q = defaultdict(deque)
    for nid, pr in zip(row.get("noise_items", []) or [], row.get("noise_items_prob", []) or []):
        try:
            q[nid].append(float(pr))
        except:
            q[nid].append(0.0)
    pos = [0.0] * L
    for i, it in enumerate(ids):
        if it in q and q[it]:
            pos[i] = q[it].popleft()
    return pos

def _make_noise_mask(row, thr, max_num):
    ids = row["item_ids"]
    probs = _pos_prob_from_row(row)
    cand = [(i, probs[i]) for i in range(len(ids)) if probs[i] > thr]
    cand.sort(key=lambda x: x[1], reverse=True)
    if max_num is not None:
        cand = cand[:max_num]
    m = [False] * len(ids)
    for i, _ in cand:
        m[i] = True
    return m

def _make_noise_prior(row):
    probs = _pos_prob_from_row(row)
    arr = np.array(probs, dtype=float)
    if np.isfinite(arr).sum() == 0:
        return [0.0] * len(probs)
    lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
    if hi - lo < 1e-8:
        return [float(x > 0) for x in arr]
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return arr.tolist()

# def trans_into_denoise_df(df, threshold, max_denoise_num):
#     def process_row(row):
#         item_ids = copy.deepcopy(row["item_ids"])
#         noise_items = row["noise_items"][:max_denoise_num]
#         noise_items_prob = row["noise_items_prob"]
#         suggest_item_ids = row["suggest_item_ids"][:max_denoise_num]
#
#         id_counts = Counter(item_ids)
#
#         for i, id in enumerate(item_ids):
#             if id in noise_items and noise_items_prob[noise_items.index(id)] > threshold:
#                 item_ids[i] = suggest_item_ids[noise_items.index(id)]
#             row["item_ids"] = item_ids
#
#         return row
#
#     df = df.apply(process_row, axis=1)
#     return df
def trans_into_denoise_df(df, threshold, max_denoise_num, mode="all"):
    def process_row(row):
        ids = copy.deepcopy(row["item_ids"])
        L = len(ids)
        pos_prob = _pos_prob_from_row(row)

        cand = [(i, pos_prob[i]) for i in range(L) if pos_prob[i] > threshold]
        cand.sort(key=lambda x: x[1], reverse=True)
        cand = cand[:max_denoise_num] if max_denoise_num is not None else cand

        sugg_q = defaultdict(deque)
        for nid, sid in zip(row.get("noise_items", []) or [], row.get("suggest_item_ids", []) or []):
            sugg_q[nid].append(sid)

        for i, _ in cand:
            cur = ids[i]
            if mode == "remove":
                ids[i] = 0
            else:
                sid = sugg_q[cur].popleft() if cur in sugg_q and sugg_q[cur] else None
                ids[i] = sid if sid is not None else 0
        ids = [x for x in ids if x not in (0, None)]
        row["item_ids"] = ids
        return row
    return df.apply(process_row, axis=1)


class Data_Pro:
    def __init__(self, dataroot, train_denoise_flag=False, test_denoise_flag=False):
        train_file = os.path.join(dataroot, "train.csv")
        test_file = os.path.join(dataroot, "test_5000.csv")

        train_df, test_df = read_file(train_file), read_file(test_file)
        train_size, test_size = len(train_df), len(test_df)
        data_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        # if "noise_prob" in data_df.columns:
        #     all_scores = [score for sublist in train_df["noise_prob"] for score in sublist]
        #     train_noise_scores_threshold = np.percentile(all_scores, world.config["train_noise_filter_threshold"] * 100)
        #
        #     all_scores = [score for sublist in test_df["noise_prob"] for score in sublist]
        #     test_noise_scores_threshold = np.percentile(all_scores, world.config["test_noise_filter_threshold"] * 100)
        # else:
        #     train_noise_scores_threshold = 0
        #     test_noise_scores_threshold = 0
        #
        # print(f"train_noise_threshold: {train_noise_scores_threshold}")
        # print(f"test_noise_threshold: {test_noise_scores_threshold}")
        def _flatten_pos_prob(df_part):
            vals = []
            for _, r in df_part.iterrows():
                vals += _pos_prob_from_row(r)  # ← 使用你在上面新增的 _pos_prob_from_row
            return vals

        train_scores = _flatten_pos_prob(train_df)
        test_scores = _flatten_pos_prob(test_df)

        train_noise_scores_threshold = (
            np.percentile(train_scores, world.config["train_noise_filter_threshold"] * 100)
            if len(train_scores) else 0.0
        )
        test_noise_scores_threshold = (
            np.percentile(test_scores, world.config["test_noise_filter_threshold"] * 100)
            if len(test_scores) else 0.0
        )

        print(f"train_noise_threshold: {train_noise_scores_threshold}")
        print(f"test_noise_threshold: {test_noise_scores_threshold}")
        self.train_df, self.test_df = train_df.copy(), test_df.copy()
        self.train_df.loc[:, "noise_mask"] = self.train_df.apply(
            lambda r: _make_noise_mask(r, train_noise_scores_threshold, world.config["train_max_denoise_num"]), axis=1
        )
        self.test_df.loc[:, "noise_mask"] = self.test_df.apply(
            lambda r: _make_noise_mask(r, test_noise_scores_threshold, world.config["test_max_denoise_num"]), axis=1
        )

        self.train_df.loc[:, "noise_prior"] = self.train_df.apply(_make_noise_prior, axis=1)
        self.test_df.loc[:, "noise_prior"] = self.test_df.apply(_make_noise_prior, axis=1)

        max_item_id = max(data_df["item_ids"].apply(max))
        self.item_num = max_item_id
        if world.config.get("denoise_category", "none").lower() == "none":
            print("denoise_category=none → 不进行任何去噪操作（仅提供 noise_mask 给模型）。")
            return


        print("Max Item ID: ", self.item_num)
        print("Max User ID: ", max(data_df["user_id"]))

        # self.train_df = (
        #     trans_into_denoise_df(
        #         self.train_df,
        #         threshold=train_noise_scores_threshold,
        #         max_denoise_num=world.config["train_max_denoise_num"],
        #     )
        #     if train_denoise_flag
        #     else self.train_df
        # )
        # self.test_df = (
        #     trans_into_denoise_df(
        #         self.test_df,
        #         threshold=test_noise_scores_threshold,
        #         max_denoise_num=world.config["test_max_denoise_num"],
        #     )
        #     if test_denoise_flag
        #     else self.test_df
        # )
        self.train_df = (
            trans_into_denoise_df(
                self.train_df,
                threshold=train_noise_scores_threshold,
                max_denoise_num=world.config["train_max_denoise_num"],
                mode=world.config.get("denoise_category", "all")  # 新增
            )
            if train_denoise_flag else self.train_df
        )

        self.test_df = (
            trans_into_denoise_df(
                self.test_df,
                threshold=test_noise_scores_threshold,
                max_denoise_num=world.config["test_max_denoise_num"],
                mode=world.config.get("denoise_category", "all")  # 新增
            )
            if test_denoise_flag else self.test_df
        )



    def get_data_df(self):
        return self.train_df, self.test_df


class Seq_dataset(Dataset):
    def __init__(self, data_frame):
        self.df = data_frame

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_ids = row["item_ids"]
        noise_mask = row.get("noise_mask", [False] * len(item_ids))

        seq_item_id = torch.tensor(item_ids, dtype=torch.long) if isinstance(item_ids, list) else item_ids
        seq_item_id = F.pad(seq_item_id, (world.config["maxlen"] + 1 - seq_item_id.size(0), 0), "constant", 0)

        noise_mask_t = torch.tensor(noise_mask, dtype=torch.bool)
        noise_mask_t = F.pad(noise_mask_t, (world.config["maxlen"] + 1 - noise_mask_t.size(0), 0), "constant", False)

        prior = row.get("noise_prior", [0.0] * len(item_ids))
        prior_t = torch.tensor(prior, dtype=torch.float32)
        prior_t = F.pad(prior_t, (world.config["maxlen"] + 1 - prior_t.size(0), 0), "constant", 0.0)
        return seq_item_id, noise_mask_t, prior_t, idx

