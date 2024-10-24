# todo:
# 1. implement Dataset class
# 2. implement DataLoader class

# import torch
# from torch.utils.data import Dataset
#
# class MNISTDataset(Dataset):
#     def __init__(self, images, labels):
#         self.images = torch.tensor(images, dtype=torch.float32)
#         self.labels = torch.tensor(labels, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx]
#
# from torch.utils.data import DataLoader
#
# batch_size = 4
# dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)


class Dataset:
    pass

class DataLoader:
    pass