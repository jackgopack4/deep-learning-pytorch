import torch
import torch.nn.functional as F
import torchvision
import numpy as np

class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size, stride=stride, padding=kernel_size//2, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Conv2d(n_output, n_output, kernel_size, padding=kernel_size//2,bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None
            # define residual connection
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output,kernel_size=1,stride=stride, bias=False),
                                                      torch.nn.BatchNorm2d(n_output))
        def forward(self, x):
            identity = x
            # add Residual connection
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity
    def __init__(self, layers=[32, 64, 128, 256], n_input_channels=3, n_output_channels=6, kernel_size=5):
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7, padding=3, stride=2, bias=False),
             torch.nn.BatchNorm2d(layers[0]),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]
        c = layers[0]
        for l in layers:
            L.append(self.Block(c, l, kernel_size, stride=2))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, n_output_channels)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        normalize=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])
        return self.classifier(self.network(normalize(x)).mean(dim=[2, 3]))

class FCN(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size, stride=1,residual=False):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size, stride=stride, padding=kernel_size//2, bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Conv2d(n_output, n_output, kernel_size, stride=1, padding=kernel_size//2,bias=False),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None
            # define residual connection
            if (stride != 1 or n_input != n_output) and residual:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output,kernel_size=1,stride=stride, bias=False),
                                                      torch.nn.BatchNorm2d(n_output))
        def forward(self, x):
            identity = x
            # add Residual connection
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity
    def __init__(self, layers=[64, 128, 256], n_input_channels=3, n_output_channels=5, kernel_size=7):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.Levels = []
        self.Upsamples = []        
        L = [torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7, padding=3, stride=2, bias=False),
             torch.nn.BatchNorm2d(layers[0]),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ]
        c = layers[0]
        L.append(self.Block(c,c,kernel_size=3))
        L.append(self.Block(c,c,kernel_size=3))
        self.Levels.append(torch.nn.Sequential(*L))
        self.Upsamples.append(torch.nn.Upsample(scale_factor=c//16, mode='bilinear'))
        for l in layers[1:]:
            L = []
            L.append(self.Block(c, l, kernel_size=3, stride=2,residual=True))
            L.append(self.Block(l, l, kernel_size=3))
            c = l
            self.Levels.append(torch.nn.Sequential(*L))
            self.Upsamples.append(torch.nn.Upsample(scale_factor=c//16, mode='bilinear'))
        self.conv1k = torch.nn.Conv2d(np.sum(layers), n_output_channels, kernel_size=1)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        
        normalize=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])
        # upsample levels array
        u = []
        for i in range(0,len(self.Levels)):
            x = self.Levels[i](normalize(x))
            u.append(self.Upsamples[i](x))
        combine_skips = torch.cat(u, dim=1)
        return self.conv1k(combine_skips)


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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
