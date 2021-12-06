import numpy as np
import pystk

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms
from tournament import utils

DATASET_PATH = 'test.pkl'


class Team(IntEnum):
    RED = 0
    BLUE = 1


def to_image(x, proj, view): #puck coordinate
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

def puck_in_frame(instance):
    i = instance >> 24
    return 8 in i

def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        self.pickle = tournament.utils.load_recording(dataset_path)
        self.images = []
        self.labels = []
        for d in self.pickle:
            team1_images = d.get('team1_images')
            team1_projectile = d.get('team1_projectile')
            team2_images = d.get('team2_images')
            team2_projectile = d.get('team2_projectile')
            for i in len(team1_images):
                self.images.append(team1_images[i])
                proj = d.get('team1_state')[i].get('camera').get('projection')
                view = d.get('team1_state')[i].get('camera').get('view')
                in_frame = d.get('team1_projectile')[i]
                coord = to_image(ball_loc,proj,view)
                self.labels.append(tuple(in_frame,coord[0],coord[1]))
            for i in len(team2_images):
                self.images.append(team2_images[i])
                proj = d.get('team2_state')[i].get('camera').get('projection')
                view = d.get('team2_state')[i].get('camera').get('view')
                in_frame = d.get('team2_projectile')[i]
                coord = to_image(ball_loc,proj,view)
                self.labels.append(tuple(in_frame,coord[0],coord[1]))
        
        self.transform = transform(*self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = self.transform(*image)
        return image,label

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)