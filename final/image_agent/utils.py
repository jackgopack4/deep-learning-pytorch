import numpy as np
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
            #print('loading recording')
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
        j = 0;
        with(open(dataset_path, 'rb')) as f:
            for d in load_recording(f):
                #print('called load_recording')
                j+=1
                #print('j=',j)
                if(j > 127): break
                team1_images = d.get('team1_images')
                team1_projectile = d.get('team1_projectile')
                team2_images = d.get('team2_images')
                team2_projectile = d.get('team2_projectile')
                ball_loc = d.get('soccer_state').get('ball').get('location')
                #print('team1_images length =',len(team1_images))
                for i in range(0,len(team1_images)):
                    self.images.append(team1_images[i])
                    #print(data'team1_image=',img)
                    proj = d.get('team1_state')[i].get('camera').get('projection')
                    view = d.get('team1_state')[i].get('camera').get('view')
                    self.pucks.append(int(d.get('team1_projectile')[i]))
                    self.locs.append(to_image(ball_loc,proj,view))
                    #self.labels.append((in_frame,coord[0],coord[1]))
                    #self.data.append((img, in_frame,coord))
                for i in range(0,len(team2_images)):
                    self.images.append(team2_images[i])
                    proj = d.get('team2_state')[i].get('camera').get('projection')
                    view = d.get('team2_state')[i].get('camera').get('view')
                    self.pucks.append(int(d.get('team2_projectile')[i]))
                    self.locs.append(to_image(ball_loc,proj,view))
                    #self.labels.append((in_frame,coord[0],coord[1]))
                    #self.data.append((img, in_frame,coord))
        #print(self.data)
        self.transform = transform
        #print(self.transform)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        puck = self.pucks[idx]
        loc = self.locs[idx]
        if self.transform:
            img = Image.fromarray(img)
            #print(img)
            img = self.transform(img)
        data = (img, np.array(puck), np.array(loc))
        return data

def load_data(dataset_path=DATASET_PATH, transform=dense_transforms.ToTensor(), num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path, transform=transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)