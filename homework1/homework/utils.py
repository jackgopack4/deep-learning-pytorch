from PIL import Image
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
      fileToOpen = dataset_path+"/labels.csv"
      self.dataset_path=dataset_path
      with open(fileToOpen, newline='') as csvFile:
        labelsReader = csv.reader(csvFile)
        self.labelsList=list(labelsReader)
        self.labelsList.pop(0)
      #raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
      #print(len(self.labelsList))
      return len(self.labelsList)
      #raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
      imageFileName=self.labelsList[idx][0]
      imageLabelName=self.labelsList[idx][1]
      I = Image.open(self.dataset_path+"/"+imageFileName)
      image_to_tensor = transforms.ToTensor()
      image_tensor = image_to_tensor(I)
      #print(image_tensor.shape)
      return (image_tensor,LABEL_NAMES.index(imageLabelName))
      #raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
