import numpy as np
import torch
import torch.nn as nn
import torch_optimizer
from musicnet_dataset import MusicNet
from maestro_dataset import Maestro
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score,f1_score
import mlflow
from torch.cuda.amp import GradScaler
from synth_piano_dataset import SynthPiano
from meta_dataset import MetaDataset
from mlflow import log_param
import sys
base_path = 'insert absolute path/RSE-pytorch-scaled'
sys.path.insert(0, base_path)
from model.big_model4_2 import TranscriptionModel
import argparse

#warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:8000")

def console(level,*args):
    if verbosity>=level:
        print(args)

def start_and_log():
    if logging:
        mlflow.start_run(experiment_id='886537638585358668',run_name=mlflow_run_name)
        log_param("epochs", N_EPOCHS)
        log_param("epoch_size", EPOCH_SIZE)
        log_param("eval_size", EVAL_SIZE)
        log_param("batch_size", BATCH_SIZE)
        log_param("label_smoothing", SMOOTH)
        log_param("window_size", window_size)
        log_param("n_benes_blocks", benes_blocks)
        log_param("n_hidden", n_hidden)
        log_param("initial_lr", learning_rate)
        log_param("lr_scheduler", "ReduceLROnPlateau")
        log_param("lr_scheduler.factor", lr_plateau_factor)
        log_param("lr_scheduler.patience", lr_plateau_patience)
        log_param("Adam.eps", epsilon)
        log_param("Adam.weight_decay", weight_decay)
        log_param("test_data_random_seed", test_data_random_seed)
        log_param("sampling_rate", fs)


def log_metric(k, v, s):
    if logging:
        mlflow.log_metric(k, v, s)

def test_model(model,data_loaders,step,device):
    for name,test_loader in data_loaders.items():
        all_targets = []
        all_preds = []
        model.eval()
        for inputs, targets in test_loader:
            with torch.no_grad():
                result = model(
                    inputs.to(device)
                )
                targets = targets[:, window_size // 2, :].squeeze(1)
                y = targets.numpy()
                pred = result.detach().cpu().numpy()
                all_targets += list(y)
                all_preds += list(pred)

        targets_np = np.array(all_targets)
        preds_np = np.array(all_preds)
        mask = targets_np.sum(axis=0) > 0
        aps = average_precision_score(targets_np[:, mask], preds_np[:, mask])
        af1mic = f1_score(targets_np[:, mask],preds_np[:, mask]>0.5,average='micro')
        af1mac = f1_score(targets_np[:, mask],preds_np[:, mask]>0.5,average='macro')
        console(2,f"{name} APS: {aps : .2%}.")
        console(2,f"{name} AF1 micro: {af1mic : .2%}.")
        console(2,f"{name} AF1 macro: {af1mac : .2%}.")
        log_metric(name+".APS", aps, step)
        log_metric(name+".AF1 micro", af1mic, step)
        log_metric(name+".AF1 macro", af1mac, step)
    if logging:
        torch.save(model.state_dict(), f"./models/{mlflow_run_name}_model.pth")


def train_model():
    global learning_rate
    learning_rate = ((0.00125 * np.sqrt(96 / n_hidden)) * (BATCH_SIZE / 16.0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console(1,device)
    start_and_log()
    torch.cuda.empty_cache()

    scaler = GradScaler()
    model = TranscriptionModel(window_size, n_hidden,benes_blocks)
    if use_pretrained:
        model.load_state_dict(torch.load(pretrained_path), strict=False)
        console(1,"Model loaded")
    model.to(device)
    sum_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console(1,"Count of learnable parameters: ",sum_params)
    if logging:
        log_param("learnable-params", sum_params)
        mlflow.log_artifact("./synth_piano_dataset.py")
        mlflow.log_artifact("./maestro_dataset.py")
        mlflow.log_artifact("./musicnet_dataset.py")
        mlflow.log_artifact("./meta_dataset.py")
        mlflow.log_artifact("./train_model.py")
        mlflow.log_artifact(base_path+"/model/big_model4_2.py")
        mlflow.log_artifact(base_path+"/model/rse.py")
        for art in artifacts:
            mlflow.log_artifact(art)

    optimizer = torch_optimizer.RAdam(model.parameters(), lr=learning_rate, eps=epsilon, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=lr_plateau_factor,patience=lr_plateau_patience,verbose=True,mode="min",min_lr=min_lr)

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss_fn.to(device)

    step = 0
    epoch = 0

    with MetaDataset([
        Maestro("../../maestro-v3.0.0", train=True, preprocess=False, window=window_size,sampling_rate=fs),
        SynthPiano("../../ADLPiano_synth", train=True, preprocess=False, window=window_size,sampling_rate=fs),
        MusicNet("../../data", train=True, window=window_size,sampling_rate=fs),
       # SynthPiano("../../ADLPiano_synth", train=True,room=False, preprocess=False, window=window_size,sampling_rate=fs)
    ],[1,1,1,1],EPOCH_SIZE) as train_dataset,\
        Maestro("../../maestro-v3.0.0", train=False, window=window_size, epoch_size=EVAL_SIZE,sampling_rate=fs,random_seed=test_data_random_seed) as maestro_test,\
        MusicNet("../../data", train=False, window=window_size, epoch_size=EVAL_SIZE,sampling_rate=fs,random_seed=test_data_random_seed) as musicnet_test,\
        SynthPiano("../../ADLPiano_synth", train=False, preprocess=False, window=window_size, epoch_size=EVAL_SIZE,sampling_rate=fs,random_seed=test_data_random_seed) as synth_test:

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8,prefetch_factor=2)
        synth_loader = DataLoader(synth_test, batch_size=BATCH_SIZE, num_workers=8)
        maestro_loader = DataLoader(maestro_test, batch_size=BATCH_SIZE, num_workers=8)
        musicnet_loader = DataLoader(musicnet_test, batch_size=BATCH_SIZE, num_workers=8)
        while epoch <= N_EPOCHS:
            epoch += 1
            losses = []
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                loss, result = model(
                    inputs.to(device),
                    targets.to(device).float(),
                    loss_fn,
                    SMOOTH,
                )
                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
                step += 1
                losses.append(loss.item())
                log_metric("loss", loss, step)
                if step % 100 == 0:
                    log_metric("curr_lr", optimizer.param_groups[0]["lr"], step)
                    avgLoss = np.mean(losses[-100:])
                    scheduler.step(avgLoss)

            console(2,f"Train. Epoch {epoch}, loss: {np.mean(losses[-100:]):.3f}")
            log_metric("epoch", epoch, step)
            if epoch % test_epoch == 0:
                test_model(model,{
                    'ADLPiano_synth':synth_loader,
                    'Maestro':maestro_loader,
                    'Musicnet':musicnet_loader
                    },step,device)
    with Maestro("../../maestro-v3.0.0", train=False, window=window_size, epoch_size=6000,sampling_rate=fs,random_seed=test_data_random_seed) as maestro_test,\
        MusicNet("../../data", train=False, window=window_size, epoch_size=6000,sampling_rate=fs,random_seed=test_data_random_seed) as musicnet_test,\
        SynthPiano("../../ADLPiano_synth", train=False, preprocess=False, window=window_size, epoch_size=6000,sampling_rate=fs,random_seed=test_data_random_seed) as synth_test:
        synth_loader = DataLoader(synth_test, batch_size=BATCH_SIZE, num_workers=8)
        maestro_loader = DataLoader(maestro_test, batch_size=BATCH_SIZE, num_workers=8)
        musicnet_loader = DataLoader(musicnet_test, batch_size=BATCH_SIZE, num_workers=8)
        test_model(model,{
                    'ADLPiano_synth':synth_loader,
                    'Maestro':maestro_loader,
                    'Musicnet':musicnet_loader
                    },step,device)

    if logging:
        torch.save(model.state_dict(), f"./models/{mlflow_run_name}_model.pth")
    else:
        torch.save(model.state_dict(), "./model.pth")


parser = argparse.ArgumentParser(description="Model trainer")
parser.add_argument("--window_size", type=int,default=16384, help="")
parser.add_argument("--mlflow_log", type=bool,default=False, help="")
parser.add_argument("--run_name", type=str,default="test_run", help="")
parser.add_argument("--verbosity", type=int,default=1, help="")
parser.add_argument("--n_benes_blocks", type=int, default=2, help="")
parser.add_argument("--epochs", type=int, default=1000, help="")
parser.add_argument("--epoch_size", type=int, default=2000, help="")
parser.add_argument("--eval_size", type=int, default=1000, help="")
parser.add_argument("--batch_size", type=int, default=18, help="")
parser.add_argument("--n_hidden", type=int, default=48 * 5, help="")
parser.add_argument("--label_smoothing", type=float, default=0, help="")
parser.add_argument("--lr_plateau_patience", type=int, default=700, help="")
parser.add_argument("--pretrained_path", type=str, default=None, help="")
parser.add_argument("--sampling_rate", type=int, default=11000, help="")
parser.add_argument("--test_epoch", type=int, default=9, help="")

args = parser.parse_args()

window_size = args.window_size
test_epoch = args.test_epoch
fs = args.sampling_rate
verbosity = args.verbosity
benes_blocks = args.n_benes_blocks
N_EPOCHS = args.epochs
EPOCH_SIZE = args.epoch_size
EVAL_SIZE = args.eval_size
BATCH_SIZE = args.batch_size
n_hidden = args.n_hidden
SMOOTH = args.label_smoothing
lr_plateau_patience = args.lr_plateau_patience
epsilon = 1e-5
weight_decay = 0.01
lr_plateau_factor = 0.7
learning_rate = None
logging = args.mlflow_log
min_lr = 5.0954e-12
test_data_random_seed = 10
mlflow_run_name = args.run_name
pretrained_path = args.pretrained_path
use_pretrained = pretrained_path!=None
artifacts = []

train_model()