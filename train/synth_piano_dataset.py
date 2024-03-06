from __future__ import print_function

import mmap
import os
import os.path
import pickle
import pretty_midi

import numpy as np
import torch.utils.data as data
from intervaltree import IntervalTree
import wave
from scipy.signal import resample
import random
from wave_patch import _read_fmt_chunk

sz_float = 4  # size of a float

wave.Wave_read._read_fmt_chunk = _read_fmt_chunk  #patched wave lib to samplewidth 4 if unknown format

class SynthPiano(data.Dataset):

    train_tree = 'train_tree.pckl'
    test_tree = 'test_tree.pckl'
    test_midis = 'midi_test'
    train_midis = 'midi'

    def __init__(self, root, train,room=True, preprocess=False, refresh_cache=False, window=16384,
                 sampling_rate=11000, epoch_size=100000,random_seed=None):
        self.refresh_cache = refresh_cache
        self.mmap = mmap
        self.window = window
        self.size = epoch_size                                                                  #! größe einer epoche
        self.m = 128                                                                            #! anzahl zu erkennender klassen (noten)
        self.sr = sampling_rate
        self.random_seed = random_seed
        self.indices = None
        self.generator = None
        self.train = train

        self.root = os.path.expanduser(root)

        if preprocess:
            self.preprocess()

        if room:
            self.train_data = 'room'
            self.test_data = 'room_test'
        else:
            self.train_data = 'audio'
            self.test_data = 'audio_test'

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        
        if train:
            labels_path = os.path.join(self.root, self.train_tree)
            self.data_path = self.train_data
        else:
            labels_path = os.path.join(self.root, self.test_tree)
            self.data_path = self.test_data
        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)

        self.rec_paths = list(self.labels.keys())                                                
        self.records = dict()
        self.open_files = []

    def __enter__(self):
        if self.random_seed!=None:
            self.generator = np.random.default_rng(self.random_seed)
            self.indices = self.generator.integers(0,len(self.rec_paths),self.size)
        return self

    def __exit__(self, *args):
        pass

    def access(self, rec_path):
        with wave.open(os.path.join(self.root,self.data_path,rec_path+".wav"), 'rb') as input_wave:
            frame_rate = 16000
            total_frames = input_wave.getnframes()
            scale = (frame_rate/self.sr)

            if self.random_seed==None:
                s = np.random.randint(0, total_frames - self.window)
            else:   # for stable test results with random seed:
                s = self.generator.integers(0,total_frames - self.window)

            input_wave.setpos(s)
            wav_data = np.frombuffer(input_wave.readframes(int(self.window*scale+0.5)), dtype=np.float32)

        if self.sr != 16000:
            x = resample(wav_data,self.window)
        else:
            x = wav_data.copy()

        if self.train and bool(random.randint(0, 1)):
            random_noise = np.random.uniform(0,0.00005)
            if bool(random.randint(0, 1)):
                x = np.clip(x+np.random.normal(0,random_noise,x.shape),-1,1)
            else:
                x = np.clip(x+np.random.uniform(-random_noise,random_noise,x.shape),-1,1)
        
        x = x.astype(np.float32)

        y = np.zeros((self.window, self.m), dtype=np.bool_)
        pos = s/float(frame_rate)
        width = self.window/float(self.sr)
        for interval in self.labels[rec_path][pos:pos+width]:
            note = interval.data[1]
            start = int((interval.begin-pos)*self.sr)
            end = int((interval.end-pos)*self.sr)
            if start<0:
                start=0
            if end>=self.window:
                end = self.window
            if 0 <= note < 128:
                y[start:end,note] = True

        return x, y

    def __getitem__(self, index=0):
        if self.random_seed==None:
            rec_id = self.rec_paths[np.random.randint(0, len(self.rec_paths))]
        else:   # for stable test results with random seed:
            rec_id = self.rec_paths[self.indices[index]]

        return self.access(rec_id)

    def __len__(self):
        return self.size

    def _check_exists(self):
        return os.path.exists(self.root) and \
               os.path.exists(os.path.join(self.root, self.train_tree)) and \
               os.path.exists(os.path.join(self.root,  self.test_tree)) and \
               not self.refresh_cache

    def preprocess(self):
        print('Processing...')

        trees = self.process_labels(os.listdir(os.path.join(self.root,self.test_midis)),self.test_midis)
        with open(os.path.join(self.root,self.test_tree), 'wb') as f:
            pickle.dump(trees, f)

        trees = self.process_labels(os.listdir(os.path.join(self.root,self.train_midis)),self.train_midis)
        with open(os.path.join(self.root, self.train_tree), 'wb') as f:
            pickle.dump(trees, f)

        self.refresh_cache = False
        print('Processing Complete')
    
    def process_labels(self,midi_files,path):
        trees = dict()
        for item in midi_files:
            f_name = item[:-4]
            tree = IntervalTree()
            midi_data = pretty_midi.PrettyMIDI(os.path.join(self.root,path, item))
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    start_time = float(note.start)
                    end_time = float(note.end)
                    instr = instrument.program
                    pitch = int(note.pitch)
                    note_velocity = note.velocity
                    if start_time<end_time:
                        tree[start_time:end_time] = (instr, pitch, note_velocity)
            trees[f_name] = tree
        return trees