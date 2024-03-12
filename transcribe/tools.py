import torch.utils.data as data
from torch.utils.data import DataLoader
import torch

import sys

base_path = 'insert your absolute path/RSE-pytorch-scaled'
sys.path.insert(0, base_path)
from model.big_model4_2 import TranscriptionModel

import scipy.io.wavfile as wavfile
import numpy as np
from tqdm.auto import tqdm
from scipy.signal import resample
import pretty_midi
import numba


class FileLoader(data.Dataset):

    def __init__(self, filename,window_size=16384,stride=165):
        self.pad_size = window_size//2
        self.stride = stride
        self.window_size = window_size
        self.filename = filename

    def __len__(self):
        return self.max//self.stride
    
    def __enter__(self):
        sample_rate, audio_data = wavfile.read(self.filename)
        audio_data = audio_data / (np.linalg.norm(audio_data))
        if len(audio_data.shape)==2:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = resample(audio_data, int(len(audio_data) * 11000 / sample_rate))
        self.audio_data = np.pad(audio_data, (self.pad_size, self.pad_size), 'constant').astype(np.float32)
        self.fs = sample_rate
        self.max = len(self.audio_data) - self.window_size + 1
        return self
    
    def __exit__(self, *args):
        pass

    def __getitem__(self, idx):
        i = idx*self.stride
        x = self.audio_data[i:i + self.window_size]
        return x

def predictAudio(audio_path,window_size,stride=165):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = TranscriptionModel(window_size,240)
    else:
        device = torch.device('cpu')
        model = TranscriptionModel(window_size,240,2,128,True)
    model.load_state_dict(torch.load(base_path+'/model/big_model4.2_model.pth',map_location=device),strict=True)
    model.to(device)
    model.eval()
    all_notes = []
    b_size = 64

    with FileLoader(audio_path,window_size=window_size,stride=stride) as datas:
        t = tqdm(DataLoader(datas, batch_size=b_size, drop_last=False,num_workers=8,pin_memory=True,prefetch_factor=3), total=datas.__len__()//b_size, desc=f"Processing...")
        for batch in t:
            with torch.no_grad():
                outputs = model(batch.to(device))
                pred = outputs.cpu().detach().numpy()
            for i in range(outputs.shape[0]):
                all_notes.append(pred[i])
    res = np.array(all_notes)
    return res

@numba.jit(nopython=True)
def thresholding(res,startThresh = 0.5,endThresh = 0.2,min_dur=0.005,min_p=0.45,min_p_range=3,fs=11000,stride=165):
    dual_thresh = np.zeros_like(res)
    starts = np.zeros((128))
    lastNotes0 = np.zeros((128))
    time_delta = stride/fs
    notes = []

    for count in range(res.shape[0]):
        for note in range(128):
            if res[count][note]>startThresh:
                if lastNotes0[note]==0:
                    starts[note]=count
                lastNotes0[note] = 1
            if res[count][note]>endThresh:
                if lastNotes0[note]==1:
                    lastNotes0[note] = 1
            if res[count][note]<endThresh:
                if lastNotes0[note]==1:
                    start_idx = int(starts[note])
                    p_mean = res[start_idx:count,note].mean()
                    velo = int((p_mean**1.2)*128)
                    p_start = res[start_idx:start_idx+min_p_range,note].mean()
                    start = starts[note]*time_delta
                    end = count*time_delta
                    duration = end-start
                    if duration>min_dur and p_start>min_p:
                        dual_thresh[start_idx:count,note] = 1
                        notes.append((velo,note, start,end))
                lastNotes0[note] = 0
                dual_thresh[count,note]=0
    
    return dual_thresh,notes

@numba.jit(nopython=True)
def thresholdingOnsets(res,target_onsets,startThresh = 0.5,endThresh = 0.2,min_dur=0.005,min_p=0.45,min_p_range=3,fs=11000,stride=165):
    dual_threshold = np.zeros_like(res)
    starts = np.zeros((128))
    lastNotes0 = np.zeros((128))
    time_delta = stride/fs
    scale = 3
    pred_onsets = np.zeros_like(target_onsets)

    
    for count in range(res.shape[0]):
        for note in range(128):
            if res[count][note]>startThresh:
                if lastNotes0[note]==0:
                    starts[note]=count
                lastNotes0[note] = 1
            if res[count][note]>endThresh:
                if lastNotes0[note]==1:
                    lastNotes0[note] = 1
            if res[count][note]<endThresh:
                if lastNotes0[note]==1:
                    start_idx = int(starts[note])
                    p_start = res[start_idx:start_idx+min_p_range,note].mean()
                    start = starts[note]*time_delta
                    end = count*time_delta
                    duration = end-start
                    if duration>min_dur and p_start>min_p:
                        dual_threshold[start_idx:count,note] = 1
                        pred_onsets[int(start_idx/scale+0.5),note] = 1
                lastNotes0[note] = 0
                dual_threshold[count,note]=0

    return target_onsets, pred_onsets,dual_threshold

def load_target(shape,midi_path,fs=11000,stride=165):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    y = np.zeros(shape, dtype=np.bool_)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start_time = float(note.start)
            end_time = float(note.end)
            pitch = int(note.pitch)
            if start_time<end_time:
                start = int((start_time*fs)/stride)
                end = int((end_time*fs)/stride)
                y[start:end,pitch] = True
    return y
