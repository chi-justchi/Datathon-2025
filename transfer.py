import warnings

warnings.filterwarnings("ignore")

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

from easydict import EasyDict as edict
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

from torchmetrics.classification import AUROC

from model.matryoshka import MRLNet
from data.loader import MimicDataset, EmoryDataset, Dataloader

#args
config = edict()
config.train_batch_size = 32
config.valid_batch_size = 32
config.num_worker = 16
config.save_emd_dir = "/home/jupyter-jacob/shared/team2/mrl_embeddings/v1.pkl"
config.ckpt_path = "/home/jupyter-jacob/shared/team2/ckpt/v1.pt"
config.device = "cuda:0"
config.nesting_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
config.dataset = ["Mimic", "Emory"][0]
config.ckpt_path = "/home/jupyter-jacob/shared/team2/ckpt/v1.pt"

if config.dataset == "Mimic":    
    valid_dataset = MimicDataset(mode="test", embedding_type="All")
elif config.dataset == "Emory":    
    valid_dataset = EmoryDataset(mode="test", embedding_type="All")
    
valid_loader = Dataloader(valid_dataset, batch_size = 10, num_workers = 10, shuffle = False)

mrl_net = MRLNet(
    num_classes=13,
    input_dim = 3456,
    hidden_dims = [1024, 1024], 
    output_dim = 1024,
    nesting_list = config.nesting_list,
    relative_importance = [1.,1., 1., 1., 1., 1., 1., 1., 1.],
).to(config.device)

mrl_net.load_state_dict(torch.load(config.ckpt_path, weights_only=True))

aurocs = {
    4: AUROC(task = "multilabel", num_labels=13, ignore_index=-1).to(config.device),
    8: AUROC(task = "multilabel", num_labels=13, ignore_index=-1).to(config.device),    
    16: AUROC(task = "multilabel", num_labels=13, ignore_index=-1).to(config.device),
    32: AUROC(task = "multilabel", num_labels=13, ignore_index=-1).to(config.device),
    64: AUROC(task = "multilabel", num_labels=13, ignore_index=-1).to(config.device),
    128: AUROC(task = "multilabel", num_labels=13, ignore_index=-1).to(config.device),
    256: AUROC(task = "multilabel", num_labels=13, ignore_index=-1).to(config.device),
    512: AUROC(task = "multilabel", num_labels=13, ignore_index=-1).to(config.device),
    1024: AUROC(task = "multilabel", num_labels=13, ignore_index=-1).to(config.device),
}

best_val_loss = float("inf")
# Validation after each epoch
mrl_net.eval()
val_losses = []
with torch.no_grad():        
    for batch_id in tqdm(valid_loader):
        emb = batch_id["emb"]
        patient_id = batch_id["patient_id"]
        study_id = batch_id["study_id"]
        lab = batch_id["lab"]

        emb = emb.type(torch.float32).to(config.device) 
        lab = lab.type(torch.float32).to(config.device) 
    
        mrl_emb = mrl_net.get_mrl_embedding(emb)
        mrl_logit = mrl_net.project_to_logit(mrl_emb)
        loss = mrl_net.compute_loss(mrl_logit, lab)

        for i, (k,v) in enumerate(aurocs.items()):
            aurocs[k].update(mrl_logit[i].detach().cpu(), lab.detach().cpu().long())
                    
        val_losses.append(loss.item())

avg_val_loss = np.mean(val_losses)
print(f"Validation Total Loss = {avg_val_loss:.4f}")
for k,v in aurocs.items():
    print(f"Validation Total AUC with size:{k} = {aurocs[k].compute():.4f}")
    aurocs[k].reset()
print("\n")
