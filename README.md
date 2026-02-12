example：
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset toy_de \
  --noise_alpha 0.8 \
  --tau_low 0.2 --tau_high 0.9 \
  --noise_loss_lambda 0 \
  --bi_gamma 0.03 --cons_beta 0.00 \
  --num_heads 8 --hidden_units 128 --dropout_rate 0.25 --lr 0.0005 \
  --cuda 0 --denoise_model_path /fine-tune-model   --base_model ./llama-3b   --llm_refresh_every 10

w/o llm
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset toy_de \
  --noise_alpha 0.8 \
  --tau_low 0.2 --tau_high 0.9 \
  --noise_loss_lambda 0 \
  --bi_gamma 0.03 --cons_beta 0.00 \
  --num_heads 8 --hidden_units 128 --dropout_rate 0.25 --lr 0.0005 \
