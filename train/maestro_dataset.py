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
import pandas as pd
from scipy.signal import resample
from scipy.io import wavfile
from wave_patch import _read_fmt_chunk

sz_float = 4  # size of a float
#epsilon = 10e-6
wave.Wave_read._read_fmt_chunk = _read_fmt_chunk  #patched wave lib to samplewidth 4 if unknown format

class Maestro(data.Dataset):

    train_tree = 'train_tree.pckl'
    test_tree = 'test_tree.pckl'

    def __init__(self, root, train, preprocess=False, refresh_cache=False, normalize=True, window=16384,
                 sampling_rate=11000, epoch_size=100000,random_seed=None):
        self.refresh_cache = refresh_cache
        self.mmap = mmap
        self.normalize = normalize                                                              #? warum normalisieren? gehts ums volume?
        self.window = window
        self.size = epoch_size                                                                  #! größe einer epoche
        self.m = 128                                                                            #! anzahl zu erkennender klassen (noten)
        self.sr = sampling_rate
        self.random_seed = random_seed
        self.indices = None
        self.generator = None

        self.root = os.path.expanduser(root)

        if preprocess:
            self.preprocess()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        
        if train:
            labels_path = os.path.join(self.root, self.train_tree)
        else:
            labels_path = os.path.join(self.root, self.test_tree)
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
        with wave.open(os.path.join(self.root,rec_path+"wav"), 'rb') as input_wave:
            # Erhalte Informationen über die WAV-Datei
            channels = input_wave.getnchannels()
            sample_width = input_wave.getsampwidth()
            frame_rate = input_wave.getframerate()
            total_frames = input_wave.getnframes()
            scale = (frame_rate/self.sr)

            if self.random_seed==None:
                s = np.random.randint(0, total_frames - self.window)
            else:   # for stable test results with random seed:
                s = self.generator.integers(0,total_frames - self.window)

            input_wave.setpos(s)
            wav_data = np.frombuffer(input_wave.readframes(int(self.window*scale+0.5)), dtype=np.int16)

        x = np.reshape(wav_data, (wav_data.shape[0]//channels, channels)).astype(np.float32).mean(axis=1)

        x = resample(x,self.window)
        
        y = np.zeros((self.window, self.m), dtype=np.bool_)

        pos = s/float(frame_rate)
        width = self.window/float(self.sr)
        tree, norm = self.labels[rec_path]
        if self.normalize:
            x = x / norm
        for interval in tree[pos:pos+width]:
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
        dataset = pd.read_csv(os.path.join(self.root, 'maestro-v3.0.0.csv')).drop(["canonical_composer","canonical_title","year"],axis=1)

        trees = self.process_labels(dataset[dataset['split']=='test']['midi_filename'].to_list())
        with open(os.path.join(self.root,self.test_tree), 'wb') as f:
            pickle.dump(trees, f)

        trees = self.process_labels(dataset[dataset['split']=='train']['midi_filename'].to_list())
        with open(os.path.join(self.root, self.train_tree), 'wb') as f:
            pickle.dump(trees, f)

        self.refresh_cache = False
        print('Processing Complete')
    
    def process_labels(self,midi_files):
        trees = dict()
        for item in midi_files:
            f_name = item[:-4]
            tree = IntervalTree()
            midi_data = pretty_midi.PrettyMIDI(os.path.join(self.root, item))
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    start_time = float(note.start)
                    end_time = float(note.end)
                    instr = instrument.program
                    pitch = int(note.pitch)
                    note_velocity = note.velocity
                    if start_time<end_time:
                        tree[start_time:end_time] = (instr, pitch, note_velocity)
            fs,audio = wavfile.read(os.path.join(self.root, f_name+"wav"))
            audio = audio.mean(axis=1)
            norm_fact = np.linalg.norm(audio)
            trees[f_name] = (tree,norm_fact)
        return trees