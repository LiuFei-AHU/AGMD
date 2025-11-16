import numpy as np
import torch
from torch.utils.data import Dataset


class Neuro3DDataset(Dataset):
    def __init__(self, mris, labels, pets=None, aug=None, norm=False):
        super().__init__()
        self.mris = mris
        self.labels = labels
        self.pets = pets  # this can be none
        self.aug = aug   # this can be none. YOU CAN USE DATA AUGMENTATION COMPOSITION
        self.norm = norm

    def __getitem__(self, index):
        img = self.mris[index]
        img2 = self.pets[index] if self.pets is not None else None
        lbl = self.labels[index]
        if self.norm:
            img = img / np.max(img)
            img2 = img2 / np.max(img2) if img2 is not None else None
        if self.aug:
            for aug in self.aug:
                img = aug(img)
                img2 = aug(img2) if img2 is not None else None
        img = torch.as_tensor(img, dtype=torch.float32).unsqueeze(dim=0)  # dhw->cdhw
        img2 = torch.as_tensor(img2, dtype=torch.float32).unsqueeze(dim=0) if img2 is not None else None  # dhw->cdhw
        lbl = torch.as_tensor(lbl, dtype=torch.long)
        # the img2 may be None, because you can just use MRI for the student model
        if img2 is None:
            return img, lbl
        return img, img2, lbl

    def __len__(self):
        return len(self.labels)
