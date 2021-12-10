import numpy as np
import random
import pystk
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
from . import dense_transforms
from tournament import utils
from pickle import load

DATASET_PATH = 'test.pkl'

def to_image(x, proj, view): #puck coordinate
    p = proj @ view @ np.array(list(x) + [1])
    return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

def puck_in_frame(instance):
    i = instance >> 24
    return 8 in i

def load_recording(recording):
    while True:
        try:
            yield load(recording)
        except EOFError:
            break

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        self.images = []
        self.pucks = []
        self.locs = []
        self.data = []
        self.distances = []
        self.dist_max = 1.0
        from operator import sub 
        with(open(dataset_path, 'rb')) as f:
            for d in load_recording(f):
                team1_images = d.get('team1_images')
                team1_projectile = d.get('team1_projectile')
                team2_images = d.get('team2_images')
                team2_projectile = d.get('team2_projectile')
                ball_loc = d.get('soccer_state').get('ball').get('location')
                dist = 0.
                for i in range(0,len(team1_images)):
                    #team1
                    self.images.append(team1_images[i])
                    puck = float(d.get('team1_projectile')[i])
                    self.pucks.append(puck)
                    team1state = d.get('team1_state')
                    proj = team1state[i].get('camera').get('projection').T
                    view = team1state[i].get('camera').get('view').T
                    self.locs.append(self._to_image(ball_loc,proj,view))
                    kart_front = team1state[i].get('kart').get('front')
                    diff = [b - k for b,k in zip(ball_loc,kart_front)]
                    dist = float(np.linalg.norm(diff))* ((puck-0.5)*2.)
                    self.distances.append(dist)
                    #team2
                    self.images.append(team2_images[i])
                    puck = float(d.get('team2_projectile')[i])
                    self.pucks.append(puck)
                    team2state = d.get('team2_state')
                    proj = team2state[i].get('camera').get('projection').T
                    view = team2state[i].get('camera').get('view').T
                    self.locs.append(self._to_image(ball_loc,proj,view))
                    kart_front = team2state[i].get('kart').get('front')
                    diff = [b - k for b,k in zip(ball_loc,kart_front)]
                    dist = float(np.linalg.norm(diff))* ((puck-0.5)*2.)
                    self.distances.append(dist)
        self.transform = transform
        self.dist_max = np.abs(self.distances).max()
        print('max dist =',self.dist_max)
        self.dist_norm = np.linalg.norm(np.array(self.distances))

    def __len__(self):
        return len(self.images)

    def _to_image(self, x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1., 1.)

    def __getitem__(self, idx):
        img = self.images[idx]
        puck = self.pucks[idx]
        loc = self.locs[idx]
        dist = self.distances[idx]
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
            dist = dist / self.dist_max
        data = (img[0], torch.tensor(puck), torch.tensor(loc), torch.tensor(dist))
        return data

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)