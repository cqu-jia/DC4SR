import os
import multiprocessing

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="SASRec")
    parser.add_argument("--motivation_probe_every", type=int, default=10,
                        help="Run motivation probe every N epochs (0 disables)")
    parser.add_argument("--motivation_probe_batches", type=int, default=30,
                        help="How many batches to sample for motivation probe")
    parser.add_argument("--motivation_probe_seed", type=int, default=0,
                        help="Random seed for motivation probe")
    parser.add_argument("--motivation_probe_quantile", type=float, default=0.8,
                        help="Quantile threshold for prior_score (e.g., 0.8 means top-20% suspicious)")


    parser.add_argument("--uniform_wh", type=float, default=0)

    # ******** DCRec 参数定义 ********
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--mu_c", type=float, default=0.5)
    parser.add_argument("--sigma_c", type=float, default=0.1)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--gnn_layers", type=int, default=1)

    # ========= 兼容 CL4Rec =========
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of Transformer layers for CL4Rec/SASRec (default: use num_blocks)")
    parser.add_argument("--cl_temp", type=float, default=1.0,
                        help="Temperature for contrastive loss (InfoNCE)")
    parser.add_argument("--cl_weight", type=float, default=0.2,
                        help="Weight λ for contrastive loss term")
    parser.add_argument("--crop_rate", type=float, default=0.6,
                        help="Crop rate η for CL4Rec data augmentation")
    parser.add_argument("--mask_rate", type=float, default=0.3,
                        help="Mask rate γ for CL4Rec data augmentation")
    parser.add_argument("--reorder_rate", type=float, default=0.6,
                        help="Reorder rate β for CL4Rec data augmentation")

    # ==== IADSR 相关新参数 ====
    parser.add_argument(
        '--semantic_dim', type=int, default=3072,
        help='维度 = semantic_long.pt 里的向量维度，比如 4096'
    )
    parser.add_argument(
        '--lambda_info', type=float, default=1.0,
        help='语义对齐 InfoNCE 的损失权重'
    )
    parser.add_argument(
        '--lambda_recon', type=float, default=1.0,
        help='重构损失的权重'
    )
    parser.add_argument(
        '--align_temp', type=float, default=0.1,
        help='InfoNCE 的温度系数'
    )
    parser.add_argument("--semantic_dir", type=str, default="./data")

    # ---- LLM Denoising / Memory Refresh Args ----
    parser.add_argument("--denoise_model_path", type=str, default=None,
                        help="Path to LoRA denoise model checkpoint")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to base LLM (e.g., Llama3)")
    parser.add_argument("--llm_refresh_every", type=int, default=20,
                        help="How often (in epochs) to refresh LLM noise priors")
    parser.add_argument("--denoise_data", type=str, default="train",
                        help="Which split to refresh (train/test)")
    parser.add_argument("--sample", type=int, default=-1,
                        help="Subsample size for generate_noise_prob_data (-1 for full)")
    parser.add_argument("--llm_batch_size", type=int, default=16,
                        help="Batch size for LLM inference when generating noise_prob")

    # ===== fusion ablations defaults =====
    parser.add_argument('--fusion_mode', type=str, default='poe',
                        choices=['poe', 'fusion_net', 'bayes_odds'])
    parser.add_argument('--poe_temp', type=float, default=1.0)
    parser.add_argument('--bayes_w', type=float, default=1.0)
    parser.add_argument('--fusion_detach_prior', type=int, default=1)
    parser.add_argument('--fusion_net_hidden', type=int, default=64)
    parser.add_argument('--fusion_eps', type=float, default=1e-6)

    # 新增：放在你的 argparse 定义那里
    parser.add_argument("--collect_noise_head_stats", type=int, default=0,
                        help="1=只统计每个注意力头对噪声的关注度，不训练；0=正常训练")
    parser.add_argument("--suppress_noise_heads", type=int, default=0,
                        help="1=开启定向抑制（软抑制），按统计文件加重对这些head的惩罚")
    parser.add_argument("--noise_head_threshold", type=float, default=0.6,
                        help="将平均噪声注意力>该阈值的head标记为噪声头")
    parser.add_argument("--noise_head_alpha", type=float, default=2.0,
                        help="针对噪声头的额外加权系数（soft suppression强度）")
    parser.add_argument("--noise_head_stats_path", type=str, default=None)
    parser.add_argument("--noise_head_topk", type=int, default=2)
    # ---- Noise / Self-Supervised Denoising Args ----
    parser.add_argument("--bi_gamma", type=float, default=0.03, help="Reward attention on good items")
    parser.add_argument("--cons_beta", type=float, default=0.0, help="Consistency loss weight (self-supervised)")
    parser.add_argument("--noise_alpha", type=float, default=0.5,
                        help="Fusion weight between model noise score and prior mask")
    parser.add_argument("--tau_low", type=float, default=0.3, help="Lower threshold for uncertainty gating")
    parser.add_argument("--tau_high", type=float, default=0.7, help="Upper threshold for uncertainty gating")


    parser.add_argument("--model", type=str, default='SASRec')
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--noise_loss_lambda",
        type=float,
        default=0.1,
        help="weight for noise attention regularization"
    )
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_units", "--emb_size", dest="hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", "--n_blocks", dest="num_blocks", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=201)
    parser.add_argument("--num_heads", "--n_heads", dest="num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", "--drop_rate", dest="dropout_rate", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--maxlen", "--max_len", dest="maxlen", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--state_dict_path", default=None, type=str)
    parser.add_argument("--cuda", type=str, default="7")

    parser.add_argument("--train_noise_filter_threshold", type=float, default=0.85)
    parser.add_argument("--test_noise_filter_threshold", type=float, default=0.95)
    parser.add_argument("--denoise_category", type=str, default="none")
    parser.add_argument("--train_max_denoise_num", type=int, default=1)
    parser.add_argument("--test_max_denoise_num", type=int, default=1)
    parser.add_argument("--remain_last", type=int, default=0)

    return parser.parse_args()


args = parse_args()
config = vars(args)

CORES = multiprocessing.cpu_count() // 2

seed = args.seed
# comment = args.comment


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
