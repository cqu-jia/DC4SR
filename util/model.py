import pdb
import numpy as np
import torch
import torch.nn as nn
import config
import torch.nn.functional as F


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, item_num):
        super(SASRec, self).__init__()

        # self.user_num = user_num
        self.item_num = item_num
        self.config = world.config

        self.item_emb = torch.nn.Embedding(self.item_num + 1, self.config["hidden_units"], padding_idx=0)

        self.pos_emb = torch.nn.Embedding(
            self.config["maxlen"], self.config["hidden_units"]
        )
        self.emb_dropout = torch.nn.Dropout(p=self.config["dropout_rate"])

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.config["hidden_units"], eps=1e-8)

        # ↓↓↓ 新增：噪声头抑制/统计配置
        self.num_heads = self.config["num_heads"]
        self.num_blocks = self.config["num_blocks"]
        self.collect_noise_head_stats = bool(self.config.get("collect_noise_head_stats", 0))
        self.noise_loss_lambda = float(self.config.get("noise_loss_lambda", 0.0))
        self.noise_head_alpha = float(self.config.get("noise_head_alpha", 1.0))
        self._last_aux_loss = None  # 每次 forward 写入，main 里取用可选

        self.noise_alpha = float(self.config.get("noise_alpha", 0.5))  # 先验与模型融合权重
        self.bi_gamma = float(self.config.get("bi_gamma", 0.1))  # 奖励好项注意力
        self.cons_beta = float(self.config.get("cons_beta", 0.1))  # 一致性损失系数
        self.tau_low = float(self.config.get("tau_low", 0.3))
        self.tau_high = float(self.config.get("tau_high", 0.7))
        print(
            f"[CONFIG] tau_low={self.tau_low} tau_high={self.tau_high} noise_alpha={self.noise_alpha} cons_beta={self.cons_beta} noise_loss_lambda={self.noise_loss_lambda}")

        # 轻量噪声评分器（也可用两层 MLP）
        self.noise_scorer = nn.Sequential(
            nn.Linear(self.config["hidden_units"], self.config["hidden_units"]),
            nn.ReLU(),
            nn.Linear(self.config["hidden_units"], 1)
        )

        # ===== Fusion (for c[i]) configs =====
        self.fusion_mode = str(self.config.get("fusion_mode", "poe"))  # poe | fusion_net | bayes_odds
        self.fusion_eps = float(self.config.get("fusion_eps", 1e-6))

        # PoE temperature
        self.poe_temp = float(self.config.get("poe_temp", 1.0))

        # Bayesian odds update weight
        self.bayes_w = float(self.config.get("bayes_w", 1.0))

        # whether detach semantic prior (recommended: True)
        self.fusion_detach_prior = bool(int(self.config.get("fusion_detach_prior", 1)))

        # Learnable fusion net: input [p_mod, p_sem, |diff|] -> c
        fusion_h = int(self.config.get("fusion_net_hidden", 64))
        self.fusion_net = nn.Sequential(
            nn.Linear(3, fusion_h),
            nn.ReLU(),
            nn.Linear(fusion_h, 1)
        )

        for _ in range(self.config["num_blocks"]):
            new_attn_layernorm = torch.nn.LayerNorm(self.config["hidden_units"], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                self.config["hidden_units"], self.config["num_heads"], self.config["dropout_rate"]
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.config["hidden_units"], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.config["hidden_units"], self.config["dropout_rate"])
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _safe_logit(self, p: torch.Tensor) -> torch.Tensor:
        eps = self.fusion_eps
        p = p.clamp(eps, 1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    # def log2feats(self, log_seqs, return_attention=False):
    # def log2feats(self, log_seqs, noise_mask=None, return_attention=False):
    def log2feats(self, log_seqs, noise_mask=None, noise_prior=None, return_attention=False):
        seqs = self.item_emb(log_seqs)  # batch_size x max_len x embedding_dim
        seqs *= self.item_emb.embedding_dim**0.5

        single_seq = torch.arange(log_seqs.shape[1], dtype=torch.long, device="cuda")
        positions = single_seq.unsqueeze(0).repeat(log_seqs.shape[0], 1)
        seqs += self.pos_emb(positions)

        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == 0  # batch_size x max_len
        prior = None
        if noise_prior is not None:
            prior = noise_prior.to(log_seqs.device).clamp(0, 1)
        token_h = seqs.clone()  # [B, L, C]
        p_hat = torch.sigmoid(self.noise_scorer(token_h)).squeeze(-1)  # [B, L]

        # if prior is not None:
        #     c = self.noise_alpha * p_hat + (1 - self.noise_alpha) * prior.detach()
        # else:
        #     c = p_hat
        # ===== Fusion: build c[i] from p_mod=p_hat and p_sem=prior =====
        if prior is None:
            c = p_hat
        else:
            prior_used = prior.detach() if self.fusion_detach_prior else prior
            mode = self.fusion_mode.lower()

            if mode == "poe":
                # --- Product-of-Experts (log-odds add): logit(c)= (logit(p_mod)+logit(p_sem))/T ---
                T = max(self.poe_temp, 1e-6)
                c = torch.sigmoid((self._safe_logit(p_hat) + self._safe_logit(prior_used)) / T)

            elif mode == "fusion_net":
                # --- Learnable fusion net: c = sigmoid(MLP([p_mod, p_sem, |diff|])) ---
                diff = torch.abs(p_hat - prior_used)
                x = torch.stack([p_hat, prior_used, diff], dim=-1)  # [B, L, 3]
                c = torch.sigmoid(self.fusion_net(x).squeeze(-1))  # [B, L]

            elif mode == "bayes_odds":
                # --- Bayesian-style odds update: logit(c)=logit(p_sem) + λ*logit(p_mod) ---
                lam = self.bayes_w
                c = torch.sigmoid(self._safe_logit(prior_used) + lam * self._safe_logit(p_hat))

            else:
                raise ValueError(f"Unknown fusion_mode={self.fusion_mode}")


        if self.tau_low < self.tau_high:
            s = torch.where(
                c >= self.tau_high, torch.zeros_like(c),
                torch.where(
                    c <= self.tau_low, torch.ones_like(c),
                    (self.tau_high - c) / (self.tau_high - self.tau_low + 1e-12)
                )
            )
        else:
            s = 1.0 - c

        s = s * (~timeline_mask).float()
        seqs = seqs * s.unsqueeze(-1)  # [B, L, C]

        tl = seqs.shape[1]  # time dim len for enforce causality
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device="cuda"))
        attention_mask = torch.triu(
            torch.full((tl, tl), float('-inf'), device=log_seqs.device),
            diagonal=1
        )  # [L, L]

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            # mha_outputs, attn_output_weights = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)  # query key value
            mha_outputs, attn_output_weights = self.attention_layers[i](
                Q, seqs, seqs,
                attn_mask=attention_mask,
                # key_padding_mask=timeline_mask,
                need_weights=True,
                average_attn_weights=False
            )
            mha_outputs = torch.nan_to_num(mha_outputs, nan=0.0, posinf=0.0, neginf=0.0)

            attn = attn_output_weights  # [B, H, T_q, T_k]
            key_valid = (~timeline_mask).to(attn.dtype).unsqueeze(1).unsqueeze(2)  # [B,1,1,T_k]

            if self.noise_loss_lambda > 0.0 and noise_mask is not None:
                key_noise = noise_mask.to(attn.device).float().unsqueeze(1).unsqueeze(2)

                num = (attn * key_noise * key_valid).sum(dim=-1)  # [B,H,T_q]
                den = (attn * key_valid).sum(dim=-1) + 1e-12  # [B,H,T_q]
                noise_focus = num / den  # [B,H,T_q]  ∈ [0,1]

                head_noise_focus = noise_focus.mean(dim=(0, 2))  # [H]
                uniform_wh = bool(self.config.get("uniform_wh", 0))
                if uniform_wh:
                    head_weight = torch.ones_like(head_noise_focus)
                else:
                    head_weight = head_noise_focus / (head_noise_focus.max() + 1e-8)
                head_weight = head_weight.detach()

                hw = head_weight.view(1, -1, 1)  # [1,H,1]
                head_weighted_noise = (noise_focus * hw).mean()  # 标量

                aux = self.noise_loss_lambda * self.noise_head_alpha * head_weighted_noise
                self._last_aux_loss = aux if self._last_aux_loss is None else (self._last_aux_loss + aux)

            if self.noise_loss_lambda > 0.0:
                c_key = c.to(attn.device).unsqueeze(1).unsqueeze(2)
                good_key = (1.0 - c).to(attn.device).unsqueeze(1).unsqueeze(2)

                num_n = (attn * c_key * key_valid).sum(dim=-1)  # [B,H,T_q]
                num_g = (attn * good_key * key_valid).sum(dim=-1)  # [B,H,T_q]
                den = (attn * key_valid).sum(dim=-1) + 1e-12

                aux_noise = self.noise_loss_lambda * (num_n / den).mean()
                aux_good = - self.bi_gamma * (num_g / den).mean()
                self._last_aux_loss = (self._last_aux_loss or 0.0) + aux_noise + aux_good

            if (self.cons_beta > 0.0) and (prior is not None):
                valid = (~timeline_mask).float()  # [B,L]
                p_hat_flat = p_hat[valid.bool()]
                prior_flat = prior[valid.bool()]
                if p_hat_flat.numel() > 0:
                    bce = torch.nn.functional.binary_cross_entropy(p_hat_flat, prior_flat, reduction='mean')
                    self._last_aux_loss = (self._last_aux_loss or 0.0) + self.cons_beta * bce

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        if return_attention:
            return log_feats, attn_output_weights[:, -1, :]
        else:
            return log_feats
    def forward(self, log_seqs, pos_seqs, neg_seqs, noise_mask=None, noise_prior=None):
        self._last_aux_loss = None
        log_feats = self.log2feats(log_seqs, noise_mask=noise_mask, noise_prior=noise_prior)

        pos_embs = self.item_emb(pos_seqs)  # [B, L, C]
        neg_embs = self.item_emb(neg_seqs)  # [B, L, C]

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        return pos_logits, neg_logits, log_feats, self._last_aux_loss

    def predict(self, log_seqs, noise_mask=None, noise_prior=None):
        log_feats = self.log2feats(log_seqs, noise_mask=noise_mask, noise_prior=noise_prior)
        final_feat = log_feats[:, -1, :].squeeze(dim=1)
        item_embs = self.item_emb.weight
        logits = torch.matmul(final_feat, item_embs.t())
        return logits

    def save_item_embeddings(self, file_path):
        torch.save(self.item_emb.weight.data, file_path)


class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, gru_layers=2):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num + 1

        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=gru_layers, batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num)

    def forward(self, states):
        sequence_lengths = torch.sum(states != 0, dim=1).cpu()
        emb = self.item_embeddings(states)

        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output

    def predict(self, states):
        # Supervised Head
        sequence_lengths = torch.sum(states != 0, dim=1).cpu()
        emb = self.item_embeddings(states)

        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)

        return supervised_output

