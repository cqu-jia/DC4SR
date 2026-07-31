# BOLD

BOLD is a dual-view calibration framework for denoising sequential recommendation.

This repository provides a training script (`main.py`) that supports two modes:
- **With LLM**: uses an LLM-based semantic prior and refreshes it periodically.
- **w/o LLM**: disables the LLM prior and trains with only the recommendation backbone + calibration components.

## Requirements
- Python >= 3.9
- PyTorch (CUDA-enabled recommended)

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Data
Example layout:
```text
dc4sr/
  main.py
  data/
    toy_de/
      train.csv
      test_5000.csv
```

## Training

### With LLM
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset toy_de \
  --noise_alpha 0.8 \
  --tau_low 0.2 --tau_high 0.9 \
  --noise_loss_lambda 0 \
  --num_heads 8 --hidden_units 128 --dropout_rate 0.25 --lr 0.0005 \
  --cuda 0 \
  --denoise_model_path ./checkpoint \
  --base_model ./llama-3b \
  --llm_refresh_every 10
```

### w/o LLM
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset toy_de \
  --noise_alpha 0.8 \
  --tau_low 0.2 --tau_high 0.9 \
  --noise_loss_lambda 0 \
  --num_heads 8 --hidden_units 128 --dropout_rate 0.25 --lr 0.0005
```

## Full options
```bash
python main.py --help
```
