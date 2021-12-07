import numpy as np
import random
import pystk
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
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

def get_values(team_number, data, idx, ball_loc):
    team = 'team%s' % team_number
    puck = data.get(team+'_projectile')[idx]
    teamstate = team+'_state'
    noise = random.uniform(-1e-10,1e-10)
    if(ball_loc[0] == ball_loc[1] == ball_loc[2] == 0.0):
        ball_loc[0] = ball_loc[0] + random.uniform(-1e-10,1e-10)
        ball_loc[1] = ball_loc[1] + random.uniform(-1e-10,1e-10)
        ball_loc[2] = ball_loc[2] + random.uniform(-1e-10,1e-10)
    proj = data.get(teamstate)[idx].get('camera').get('projection')
    view = data.get(teamstate)[idx].get('camera').get('view')
    loc = to_image(ball_loc,proj,view)
    return puck,loc

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor()):
        self.images = []
        self.pucks = []
        self.locs = []
        self.data = []
        with(open(dataset_path, 'rb')) as f:
            for d in load_recording(f):
                team1_images = d.get('team1_images')
                team1_projectile = d.get('team1_projectile')
                team2_images = d.get('team2_images')
                team2_projectile = d.get('team2_projectile')
                ball_loc = d.get('soccer_state').get('ball').get('location')
                for i in range(0,len(team1_images)):
                    #team1
                    self.images.append(team1_images[i])
                    puck = d.get('team1_projectile')[i]
                    self.pucks.append(float(puck))
                    proj = d.get('team1_state')[i].get('camera').get('projection').T
                    view = d.get('team1_state')[i].get('camera').get('view').T
                    self.locs.append(self._to_image(ball_loc,proj,view))
                    #team2
                    self.images.append(team2_images[i])
                    puck = d.get('team2_projectile')[i]
                    self.pucks.append(float(puck))
                    proj = d.get('team2_state')[i].get('camera').get('projection').T
                    view = d.get('team2_state')[i].get('camera').get('view').T
                    self.locs.append(self._to_image(ball_loc,proj,view))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def _to_image(self, x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def __getitem__(self, idx):
        img = self.images[idx]
        puck = self.pucks[idx]
        loc = self.locs[idx]
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        data = (img[0], torch.tensor(puck), torch.tensor(loc))
        return data

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)