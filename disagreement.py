# motivation_probe.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import torch
import pandas as pd


@dataclass
class MotivationProbeConfig:
    max_batches: int = 30
    seed: int = 0
    every_n_epochs: int = 10
    mask_item_id: int = 0
    suspicious_quantile: float = 0.8
    harmful_threshold: float = 0.0
    save_csv: bool = True
    save_json: bool = True


def _parse_batch(batch) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]
        elif len(batch) == 2:
            return batch[0], batch[1], None
        elif len(batch) == 1:
            return batch[0], None, None
    raise ValueError(f"Unexpected batch format: {type(batch)} len={len(batch) if isinstance(batch,(list,tuple)) else 'NA'}")


@torch.no_grad()
def _per_sample_loss_vec(
    model,
    seq: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    hist_mask: Optional[torch.Tensor] = None,
    hist_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = model(seq, pos, neg, noise_mask=hist_mask, noise_prior=hist_prior)
    if isinstance(out, (list, tuple)) and len(out) == 4:
        pos_logits, neg_logits, _, _aux_loss = out
    else:
        pos_logits, neg_logits, _ = out

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    pos_labels = torch.ones_like(pos_logits)
    neg_labels = torch.zeros_like(neg_logits)

    pos_loss = bce(pos_logits, pos_labels)
    neg_loss = bce(neg_logits, neg_labels)

    valid = (pos != 0).float()  # [B, L-1] or similar
    loss = ((pos_loss + neg_loss) * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-12)
    return loss  # [B]


def _pick_prior_top_positions(
    hist_prior: torch.Tensor,
    valid_hist: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = hist_prior.clone()
    scores[~valid_hist] = -1e9
    j = torch.argmax(scores, dim=1)
    s = scores[torch.arange(scores.size(0), device=scores.device), j]
    return j, s


def _pick_random_positions(
    valid_hist: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    B, Lm1 = valid_hist.shape
    j = torch.zeros(B, dtype=torch.long, device=valid_hist.device)
    for b in range(B):
        idx = torch.where(valid_hist[b])[0]
        if idx.numel() == 0:
            j[b] = 0
        else:
            r = torch.randint(low=0, high=idx.numel(), size=(1,), generator=generator).item()
            j[b] = idx[r]
    return j


@torch.no_grad()
def run_motivation_probe(
    *,
    model,
    dataloader,
    item_num: int,
    utils_module,
    save_dir: str,
    epoch: int,
    cfg: MotivationProbeConfig,
    device: str = "cuda",
) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, f"motivation_probe_ep{epoch}.csv")
    json_path = os.path.join(save_dir, f"motivation_probe_ep{epoch}.json")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(cfg.seed + epoch * 1000))

    model.eval()

    rows: List[Dict[str, Any]] = []

    for batch_id, batch in enumerate(dataloader):
        if batch_id >= cfg.max_batches:
            break

        seq_items_id, noise_mask, noise_prior = _parse_batch(batch)

        seq, pos, neg = utils_module.get_negative_items(seq_items_id, item_num)

        seq = seq.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        # hist_mask = noise_mask[:, :-1].to(device) if noise_mask is not None else None
        # hist_prior = noise_prior[:, :-1].to(device) if noise_prior is not None else None

        # base loss
        # base = _per_sample_loss_vec(model, seq, pos, neg, hist_mask, hist_prior)  # [B]

        # valid_hist = (seq[:, :-1] != 0)  # [B, L-1]
        # B = seq.size(0)
        # ===== probe-only: align noise_mask / noise_prior length to seq length =====
        target_len = seq.size(1)

        def _align_to_seq_len(t: torch.Tensor, L: int) -> torch.Tensor:
            # t shape: [B, L] or [B, L+1] or longer
            if t.size(1) == L:
                return t
            if t.size(1) > L:
                return t[:, :L]
            return t

        hist_mask = None
        if noise_mask is not None:
            hist_mask = _align_to_seq_len(noise_mask.to(device), target_len)

        hist_prior = None
        if noise_prior is not None:
            hist_prior = _align_to_seq_len(noise_prior.to(device), target_len)

        # base loss
        base = _per_sample_loss_vec(model, seq, pos, neg, hist_mask, hist_prior)  # [B]

        valid_hist = (seq != 0)  # [B, L]
        B = seq.size(0)

        if hist_prior is not None and hist_prior.size(1) != valid_hist.size(1):
            Lp = min(hist_prior.size(1), valid_hist.size(1))
            hist_prior = hist_prior[:, :Lp]
            valid_hist = valid_hist[:, :Lp]

        if hist_mask is not None and hist_mask.size(1) != valid_hist.size(1):
            Lm = min(hist_mask.size(1), valid_hist.size(1))
            hist_mask = hist_mask[:, :Lm]
            valid_hist = valid_hist[:, :Lm]

        if hist_prior is not None:
            j_prior, s_prior = _pick_prior_top_positions(hist_prior, valid_hist)
        else:
            j_prior, s_prior = None, None

        j_rand = _pick_random_positions(valid_hist, g)

        def _eval_masked_delta(j_sel: torch.Tensor, tag: str, s_sel: Optional[torch.Tensor]):
            seq_m = seq.clone()
            seq_m[torch.arange(B, device=device), j_sel] = cfg.mask_item_id

            masked = _per_sample_loss_vec(model, seq_m, pos, neg, hist_mask, hist_prior)
            delta = base - masked

            for b in range(B):
                rows.append({
                    "epoch": int(epoch),
                    "batch": int(batch_id),
                    "b": int(b),
                    "tag": tag,  # prior_top / random
                    "j": int(j_sel[b].item()),
                    "prior_score": float(s_sel[b].item()) if s_sel is not None else float("nan"),
                    "delta_loss": float(delta[b].item()),
                })

        # prior-top
        if j_prior is not None:
            _eval_masked_delta(j_prior, "prior_top", s_prior)

        if hist_prior is not None:
            s_rand = hist_prior[torch.arange(B, device=device), j_rand]
        else:
            s_rand = None
        _eval_masked_delta(j_rand, "random", s_rand)

    df = pd.DataFrame(rows)
    if cfg.save_csv:
        df.to_csv(csv_path, index=False)

    summary = summarize_probe_df(df, cfg)

    if cfg.save_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[MotivationProbe] saved csv: {csv_path}")
    print(f"[MotivationProbe] saved json: {json_path}")
    return summary


def summarize_probe_df(df: pd.DataFrame, cfg: MotivationProbeConfig) -> Dict[str, Any]:
    if df.empty:
        return {"error": "empty_df"}

    out: Dict[str, Any] = {}

    df_random = df[df["tag"] == "random"].copy()
    df_prior = df[df["tag"] == "prior_top"].copy()

    if df_random["prior_score"].notna().any():
        # pandas spearman
        out["spearman_random_prior_vs_delta"] = float(
            df_random[["prior_score", "delta_loss"]].corr(method="spearman").iloc[0, 1]
        )
    else:
        out["spearman_random_prior_vs_delta"] = None

    if df_random["prior_score"].notna().any():
        thr = float(df_random["prior_score"].quantile(cfg.suspicious_quantile))
    else:
        thr = float("nan")
    out["prior_score_threshold_quantile"] = cfg.suspicious_quantile
    out["prior_score_threshold_value"] = thr
    out["harmful_threshold"] = cfg.harmful_threshold

    def _quadrants(d: pd.DataFrame) -> Dict[str, Any]:
        if d.empty or (not d["prior_score"].notna().any()):
            return {"n": int(len(d)), "quadrants": None}

        high_susp = d["prior_score"] >= thr
        harmful = d["delta_loss"] > cfg.harmful_threshold

        q11 = (high_susp & harmful).mean()
        q10 = (high_susp & (~harmful)).mean()
        q01 = ((~high_susp) & harmful).mean()
        q00 = ((~high_susp) & (~harmful)).mean()

        return {
            "n": int(len(d)),
            "q_susp_and_harm": float(q11),
            "q_susp_not_harm": float(q10),
            "q_not_susp_harm": float(q01),
            "q_not_susp_not_harm": float(q00),
            "p_harmful": float(harmful.mean()),
            "p_susp": float(high_susp.mean()),
        }

    out["quadrants_random"] = _quadrants(df_random)
    out["quadrants_prior_top"] = _quadrants(df_prior)

    if not df_random.empty and not df_prior.empty:
        out["p_harmful_random"] = float((df_random["delta_loss"] > cfg.harmful_threshold).mean())
        out["p_harmful_prior_top"] = float((df_prior["delta_loss"] > cfg.harmful_threshold).mean())
        out["delta_p_harmful(prior_top-random)"] = out["p_harmful_prior_top"] - out["p_harmful_random"]

        out["mean_delta_loss_random"] = float(df_random["delta_loss"].mean())
        out["mean_delta_loss_prior_top"] = float(df_prior["delta_loss"].mean())
        out["delta_mean_delta_loss(prior_top-random)"] = out["mean_delta_loss_prior_top"] - out["mean_delta_loss_random"]

    return out
