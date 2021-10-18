import torch

class ClassificationLoss(torch.nn.Module):
  def forward(self, input, target):
    return torch.nn.functional.cross_entropy(input, target)

class CNNClassifier(torch.nn.Module):
  class Block(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1):
      super().__init__()
      self.net = torch.nn.Sequential(
        torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
        torch.nn.ReLU(),
        torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
        torch.nn.ReLU()
      )
    def forward(self, x):
      return self.net(x)

  def __init__(self, layers=[32,64,128], n_input_channels=3):
    super().__init__()
    L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
         torch.nn.ReLU(),
         torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
    c = 32
    for l in layers:
      L.append(self.Block(c, l, stride=2))
      c = l
    self.network = torch.nn.Sequential(*L)
    self.classifier = torch.nn.Linear(c, 6)
    #raise NotImplementedError('CNNClassifier.__init__')

  def forward(self, x):
    # Compute the features
    z = self.network(x)
    # Global average pooling
    z = z.mean(dim=[2,3])
    # Classify
    return self.classifier(z)#[:,0]
    #raise NotImplementedError('CNNClassifier.forward')


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
