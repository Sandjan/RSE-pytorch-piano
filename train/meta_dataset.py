import torch.utils.data as data

class MetaDataset(data.Dataset):

    def __init__(self,datasets,weigths,epoch_size=2000):
        self.dsts = datasets
        self.datasets = []
        self.weigths = weigths
        self.size = epoch_size

    def __enter__(self):
        for i in range(len(self.dsts)):
            if self.weigths[i]>=1:
                dataset = self.dsts[i].__enter__()
                for _ in range(self.weigths[i]):
                    self.datasets.append(dataset)
        for ds in self.datasets:
            print(type(ds).__name__)
        return self

    def __exit__(self, *args):
        for i in range(len(self.datasets)):
            self.datasets[i].__exit__()

    def __getitem__(self, index):
        return self.datasets[index%len(self.datasets)].__getitem__(index)

    def __len__(self):
        return self.size
