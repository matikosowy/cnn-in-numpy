import numpy as np

class Dataset:
    """
    Dataset class to hold features and labels.

    Parameters:
    -----------
    x: np.ndarray
        Features
    y: np.ndarray
        Labels
    """
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

class DataLoader:
    """
    DataLoader class to load data in batches.

    Parameters:
    -----------
    dataset: Dataset
        Dataset object
    batch_size: int
        Batch size
    shuffle: bool
        Shuffle data or not
    """
    def __init__(self, dataset: Dataset, batch_size: int=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            batch_x = [self.dataset[j][0] for j in batch_indices]
            batch_y = [self.dataset[j][1] for j in batch_indices]
            yield np.array(batch_x), np.array(batch_y)