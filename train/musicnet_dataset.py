from __future__ import print_function

import csv
import errno
import mmap
import os
import os.path
import pickle
from subprocess import call

import numpy as np
import torch.utils.data as data
from intervaltree import IntervalTree
from scipy.io import wavfile
from scipy.signal import resample

sz_float = 4  # size of a float


class MusicNet(data.Dataset):
    """`MusicNet <http://homes.cs.washington.edu/~thickstn/musicnet.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``train_data``,
            otherwise from ``test_data``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        mmap (bool, optional): If true, mmap the dataset for faster access times.
        normalize (bool, optional): If true, rescale input vectors to unit norm.
        window (int, optional): Size in samples of a data point.
        pitch_shift (int,optional): Integral pitch-shifting transformations.
        jitter (int, optional): Continuous pitch-jitter transformations.
        epoch_size (int, optional): Designated Number of samples for an "epoch"
    """
    url = 'https://homes.cs.washington.edu/~thickstn/media/musicnet.tar.gz'
    raw_folder = 'raw'
    train_data, train_labels, train_tree = 'train_data', 'train_labels', 'train_tree.pckl'
    test_data, test_labels, test_tree = 'test_data', 'test_labels', 'test_tree.pckl'
    extracted_folders = [train_data, train_labels, test_data, test_labels]

    def __init__(self, root, train=True, download=False, refresh_cache=False, mmap=True, normalize=True, window=16384,
                 sampling_rate=11000,  epoch_size=100000,random_seed=None):
        self.refresh_cache = refresh_cache
        self.mmap = mmap
        self.normalize = normalize
        self.window = window
        self.size = epoch_size
        self.m = 128                                                                    
        self.sr = sampling_rate
        self.random_seed = random_seed
        self.indices = None
        self.generator = None

        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if train:
            self.data_path = os.path.join(self.root, self.train_data)
            labels_path = os.path.join(self.root, self.train_labels, self.train_tree)
        else:
            self.data_path = os.path.join(self.root, self.test_data)
            labels_path = os.path.join(self.root, self.test_labels, self.test_tree)

        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)

        self.rec_ids = list(self.labels.keys())
        self.records = dict()
        self.open_files = []

    def __enter__(self):
        for record in os.listdir(self.data_path):
            if not record.endswith('.bin'):
                continue
            if self.mmap:
                fd = os.open(os.path.join(self.data_path, record), os.O_RDONLY)
                buff = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
                self.records[int(record[:-4])] = (buff, len(buff) // sz_float)
                self.open_files.append(fd)
            else:
                f = open(os.path.join(self.data_path, record))
                self.records[int(record[:-4])] = (
                    os.path.join(self.data_path, record), os.fstat(f.fileno()).st_size // sz_float)
                f.close()
        if self.random_seed!=None:
            self.generator = np.random.default_rng(self.random_seed)
            self.indices = self.generator.integers(0,len(self.rec_ids),self.size)
        return self

    def __exit__(self, *args):
        if self.mmap:
            for mm in self.records.values():
                mm[0].close()
            for fd in self.open_files:
                os.close(fd)
            self.records = dict()
            self.open_files = []

    def access(self, rec_id, s):
        """
        Args:
            rec_id (int): MusicNet id of the requested recording
            s (int): Position of the requested data point
            shift (int, optional): Integral pitch-shift data transformation
            jitter (float, optional): Continuous pitch-jitter data transformation
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes on at the center of the audio.
        """

        if self.mmap:
            x = np.frombuffer(self.records[rec_id][0][s * sz_float:int(s + self.window) * sz_float],
                              dtype=np.float32).copy()
        else:
            fid, _ = self.records[rec_id]
            with open(fid, 'rb') as f:
                f.seek(s * sz_float, os.SEEK_SET)
                x = np.fromfile(f, dtype=np.float32, count=int(self.window))

        xp = np.arange(self.window, dtype=np.float32)
        x = np.interp(xp, np.arange(len(x), dtype=np.float32), x).astype(np.float32)
        
        y = np.zeros((self.window, self.m), dtype=np.bool_)
   
        mul = (44100 / self.sr)
        for interval in self.labels[rec_id][s*mul:(s+self.window)*mul]:
            note = interval.data[1]
            start = int((interval.begin/mul)-s)
            end = int((interval.end/mul)-s)
            if start<0:
                start=0
            if end>=self.window:
                end = self.window
            if 0 <= note < 128:
                y[start:end,note] = 1

        return x, y

    def __getitem__(self, index=0):
        """
        Args:
            index (int): (ignored by this dataset; a random data point is returned)
        Returns:
            tuple: (audio, target) where target is a binary vector indicating notes at the center of the audio.
        """

        s = None
        if self.random_seed==None:
            rec_id = self.rec_ids[np.random.randint(0, len(self.rec_ids))]
            s = np.random.randint(0, self.records[rec_id][1] - self.window)
        else:   # for stable test results with random seed:
            rec_id = self.rec_ids[self.indices[index]]
            s = self.generator.integers(0,self.records[rec_id][1] - self.window)

        return self.access(rec_id, s)

    def __len__(self):
        return self.size

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_data)) and \
               os.path.exists(os.path.join(self.root, self.test_data)) and \
               os.path.exists(os.path.join(self.root, self.train_labels, self.train_tree)) and \
               os.path.exists(os.path.join(self.root, self.test_labels, self.test_tree)) and \
               not self.refresh_cache

    def download(self):
        """Download the MusicNet data if it doesn't exist in ``raw_folder`` already."""
        from six.moves import urllib

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path):
            print('Downloading ' + self.url)
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                # stream the download to disk (it might not fit in memory!)
                while True:
                    chunk = data.read(16 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        if not all(map(lambda f: os.path.exists(os.path.join(self.root, f)), self.extracted_folders)):
            print('Extracting ' + filename)
            if call(["tar", "-xf", file_path, '-C', self.root, '--strip', '1']) != 0:
                raise OSError("Failed tarball extraction")

        # process and save as torch files
        print('Processing...')

        self.process_data(self.test_data)

        trees = self.process_labels(self.test_labels)
        with open(os.path.join(self.root, self.test_labels, self.test_tree), 'wb') as f:
            pickle.dump(trees, f)

        self.process_data(self.train_data)

        trees = self.process_labels(self.train_labels)
        with open(os.path.join(self.root, self.train_labels, self.train_tree), 'wb') as f:
            pickle.dump(trees, f)

        self.refresh_cache = False
        print('Download Complete')

    # write out wavfiles as arrays for direct mmap access
    def process_data(self, path):
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.wav'):
                continue
            fs, data = wavfile.read(os.path.join(self.root, path, item))
            resampled = resample(data, int(len(data) * self.sr / fs))
            if self.normalize:
                resampled = resampled / np.linalg.norm(resampled)
            resampled.tofile(os.path.join(self.root, path, item[:-4] + '.bin'))
    
    def process_labels(self, path):
        trees = dict()
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.csv'):
                continue
            uid = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(self.root, path, item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    note = int(label['note'])
                    tree[start_time:end_time] = (instrument, note)
            trees[uid] = tree
        return trees
