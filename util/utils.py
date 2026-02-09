import math
import random
import torch
import numpy as np

from model import SASRec
from torch.utils.data import DataLoader



def evaluate(model: SASRec, dataloader: DataLoader, sample_total: int, topk_list=[1, 5, 10, 20, 50]):
    NDCG, HR = [], []
    NDCG_total, HR_total = torch.zeros(size=(len(topk_list),)), torch.zeros(size=(len(topk_list),))

    with torch.no_grad():
        # for seq_items_id in dataloader:
        #     seq_items_id = seq_items_id.cuda()
        #     logits = -model.predict(seq_items_id[:, :-1])
        #     target_item_ids = seq_items_id[:, -1]

        for batch in dataloader:
            # -------- 解包各种可能的 batch 形态 --------
            noise_mask = None
            noise_prior = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:
                    seq_items_id, noise_mask, noise_prior, _ = batch
                elif len(batch) == 3:
                    seq_items_id, noise_mask, noise_prior = batch
                elif len(batch) == 2:
                    seq_items_id, noise_mask = batch
                    noise_prior = None
                else:
                    seq_items_id = batch[0]
            else:
                seq_items_id = batch

            seq_items_id = seq_items_id.cuda()
            if noise_mask is not None:
                noise_mask = noise_mask.cuda()

            # 传入 noise_mask（只对历史部分，去掉最后一位 label）
            hist = seq_items_id[:, :-1]
            hist_mask = noise_mask[:, :-1] if noise_mask is not None else None
            hist_prior = noise_prior[:, :-1] if noise_prior is not None else None

            logits = -model.predict(hist, noise_mask=hist_mask, noise_prior=hist_prior)
            target_item_ids = seq_items_id[:, -1]
            rank_tensor = logits.argsort(dim=-1).argsort(dim=-1)
            target_item_ranks = rank_tensor[torch.arange(rank_tensor.size(0)), target_item_ids]
            rank_list_tensor = target_item_ranks

            for i, k in enumerate(topk_list):
                Hit_num = (rank_list_tensor < k).sum().item()
                HR_total[i] += Hit_num

                mask = rank_list_tensor < k
                NDCG_num = 1 / torch.log(rank_list_tensor[mask] + 2)
                NDCG_num = NDCG_num.sum().item()
                NDCG_total[i] += NDCG_num

    NDCG = NDCG_total / (sample_total * (1.0 / math.log(2)))
    HR = HR_total / sample_total

    result_dict = dict()
    for i in range(len(topk_list)):
        result_dict["NDCG@" + str(topk_list[i])] = round(NDCG[i].item(), 4)
    for i in range(len(topk_list)):
        result_dict["HR@" + str(topk_list[i])] = round(HR[i].item(), 4)

    return result_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_negative_items(seq_items_id, item_num):
    # 允许外部误传入 (seq_items_id, noise_mask)
    if isinstance(seq_items_id, (list, tuple)) and len(seq_items_id) == 2:
        seq_items_id = seq_items_id[0]
    seq_items_id = seq_items_id.cuda()
    seq = seq_items_id[:, :-1]
    pos = seq_items_id[:, 1:]

    probabilities = torch.ones(size=(seq.shape[0], item_num + 1), dtype=torch.float, device="cuda")
    batch_indices = torch.arange(seq_items_id.shape[0], device="cuda").view(-1, 1)
    probabilities[batch_indices, seq_items_id] = 0
    probabilities[:, 0] = 0
    neg = torch.multinomial(probabilities, seq.shape[1], replacement=False)

    return seq.long(), pos.long(), neg.long()
