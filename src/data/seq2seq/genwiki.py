from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class GenWikiDataset(Dataset):
    def __init__(self, data_dir: Path):
        pass

    def __getitem__(self, index) -> T_co:
        pass