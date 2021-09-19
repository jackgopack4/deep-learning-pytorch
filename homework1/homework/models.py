import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
      loss = torch.nn.CrossEntropyLoss()
      return loss(input,target)  
      """
      @input:  torch.Tensor((B,C))
      @target: torch.Tensor((B,), dtype=torch.int64)
      @return:  torch.Tensor((,))
      """



class LinearClassifier(torch.nn.Module):
    def __init__(self, batch=128):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(3*64*64,64)
        self.linear2 = torch.nn.Linear(64,6)

    def forward(self, x):
        x=self.flatten(x)
        x=self.linear(x)
        x=self.linear2(x)
        return x
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
          torch.nn.Flatten(),
          torch.nn.Linear(3*64*64, 192),
          torch.nn.ReLU(),
          torch.nn.Linear(192,24),
          torch.nn.ReLU(),
          torch.nn.Linear(24,6)
        )

    def forward(self, x):
        return self.layers(x)
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
