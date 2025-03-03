from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class PlayingCardDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        # use ImageFolder function of torchvision -> It helps in managing labels for images by assuming images are named with labels
        self.data = ImageFolder(data_dir, transform=transform)
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes