import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from tensorboardX import SummaryWriter
import nni

from model import SASRec
import utils
from logger import CompleteLogger
from dataloader import Seq_dataset, Data_Pro

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"

config = {
    "llm_refresh_every": 20,
    "denoise_model_path": "/path/to/your/denoise/lora",
    "base_model": "/path/to/base/llama3",
    "denoise_data": "train",
    "sample": -1,
    "batchsize": 32,
    "num_epochs": 50,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "cuda": "0",
}

utils.set_seed(42)
print(">>SEED:", 42)

dataroot = "./data/your_dataset"
data_pro = Data_Pro(dataroot, train_denoise_flag=True, test_denoise_flag=True)
train_df, test_df = data_pro.get_data_df()
item_num = data_pro.item_num
batch_size = config["batchsize"]

train_dataset = Seq_dataset(train_df)
test_dataset = Seq_dataset(test_df)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

model = SASRec(item_num).cuda()
bce_criterion = torch.nn.BCEWithLogitsLoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

save_dir = os.path.join("./log", "your_experiment_name", time.strftime("%m-%d-%Hh%Mm%Ss"))
os.makedirs(save_dir, exist_ok=True)
w = SummaryWriter(save_dir)
logger = CompleteLogger(root=save_dir)

def write_tensorboard_metric(w, metric, epoch, category="Valid"):
    for key, value in metric.items():
        w.add_scalar(f"{category}/{key}".replace("@", "_"), value, epoch)

start_total = time.time()
best_NDCG = 0
patience = 0

for epoch in range(config["num_epochs"]):
    start_time_epoch = time.time()
    print(f"Start Training Epoch {epoch}")

    model.train()
    for batch in tqdm(train_dataloader):
        seq_items_id, noise_mask, noise_prior, row_idx = batch

        seq, pos, neg = utils.get_negative_items(seq_items_id, item_num)

        hist_mask = noise_mask[:, :-1] if noise_mask is not None else None
        hist_prior = noise_prior[:, :-1] if noise_prior is not None else None

        out = model(seq, pos, neg, noise_mask=hist_mask, noise_prior=hist_prior)

        pos_logits, neg_logits, _ = out

        pos_labels = torch.ones(pos_logits.shape, device="cuda")
        neg_labels = torch.zeros(neg_logits.shape, device="cuda")

        indices = (pos != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])

        adam_optimizer.zero_grad()
        loss.backward()
        adam_optimizer.step()

        w.add_scalar(f"Train/Loss", loss, epoch)

    if epoch % 1 == 0:
        model.eval()
        t_test = utils.evaluate(model, test_dataloader, len(test_dataset))
        write_tensorboard_metric(w, t_test, epoch, "Test")
        print("Test\n", t_test)

        if t_test["NDCG@20"] > best_NDCG:
            best_NDCG = t_test["NDCG@20"]
            patience = 0
            best_model_dict = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            print(f"Patience {patience}/10")
            if patience >= 10:
                break

torch.save(best_model_dict, os.path.join(save_dir, "best_model.pth"))
model.save_item_embeddings(os.path.join(save_dir, "item_embeddings.pth"))
model.load_state_dict(best_model_dict)

model.eval()
t_test = utils.evaluate(model, test_dataloader, len(test_dataset))
print("The Final Test Metric for Model is following: ")
print(t_test)

w.close()
print(f"Total time: {time.time() - start_total}")
print("Training Done!")
