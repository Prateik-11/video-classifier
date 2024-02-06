import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        # should be able to create training/validation dataset depending in input to constructor

    def __getitem__(self, i):
        return (
            torch.randn((3, 60, 224, 224)), # video (3 channels, 60 frames, 224 height and width)
            torch.randn((60)) # labels
        )
    
    def __len__(self):
        return 1000 # replace with actual length

