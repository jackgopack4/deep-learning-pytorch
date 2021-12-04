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
                self.downsample = torch.nn.Sequential(
                                      torch.nn.Conv2d(n_input, n_output,kernel_size=1,stride=stride, bias=False),
                                      torch.nn.BatchNorm2d(n_output))
        def forward(self, x):
            identity = x
            # add Residual connection
            if self.downsample is not None:
                identity = self.downsample(x)
            return self.net(x) + identity
    def __init__(self, layers=[32, 64, 128], n_input_channels=3, n_output_channels=5, kernel_size=7):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        self.Levels = torch.nn.ModuleList()
        self.Upconvs = torch.nn.ModuleList()
        L = [torch.nn.Conv2d(n_input_channels, layers[0], kernel_size=7, stride=2, padding=3, bias=False),
             torch.nn.BatchNorm2d(layers[0]),
             torch.nn.ReLU(),
             torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1,dilation=1)
            ]
        c = layers[0]
        self.Levels.append(torch.nn.Sequential(*L))
        #print('added conv layer of ',n_input_channels,'to',layers[0])
        self.Upconvs.insert(0,torch.nn.Sequential(torch.nn.ConvTranspose2d(layers[0],n_input_channels,kernel_size=7,stride=2,padding=3,output_padding=1,bias=False),
                                                  torch.nn.BatchNorm2d(n_input_channels),
                                                  torch.nn.ReLU()))
        #print('added upconv layer of',layers[0],'to',n_output_channels)
        for l in layers[1:]:
            self.Levels.append(self.Block(c, l, kernel_size=3, stride=2,residual=True))
            #print('added conv layer of',c,'to',l)
            self.Upconvs.insert(0,torch.nn.Sequential(torch.nn.ConvTranspose2d(l,c,kernel_size=3,stride=2,padding=1,output_padding=1,bias=False),
                                   torch.nn.BatchNorm2d(c),
                                   torch.nn.ReLU()))
            #print('added upconv layer of',l,'to',c)
            c = l

        self.conv1k = torch.nn.ConvTranspose2d(6, n_output_channels, kernel_size=7,stride=1,padding=3,bias=False)
        #self.sigmoid=torch.nn.Sigmoid()

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
        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        normalize=torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])
        # upsample levels array
        u = []
        x = normalize(x)
        og_size=list(x.size())
        #print('og size',og_size)
        og_height=x.size(dim=2)
        og_width=x.size(dim=3)
        block = self.Levels[0]
        x1 = block(x)
        x1_height=x1.size(dim=2)
        x1_width=x1.size(dim=3)
        #print('size after first conv',x1.size(),'for og size',og_size)
        block = self.Levels[1]
        x2 = block(x1)
        #print('size after second conv',x2.size(),'for og size',og_size)
        block = self.Levels[2]
        x3 = block(x2)
        #print('size after third conv',x3.size(),'for og size',og_size)

        x3_u = self.Upconvs[0](x3)
        #print('size after first deconv',x3_u.size(),'for og size',og_size)

        x2_u = self.Upconvs[1](x3_u)[:,:,:x1_height,:x1_width]
        #print('size after second deconv',x2_u.size(),'for og size',og_size)
        x1_u = torch.cat([x1,x2_u],dim=1)
        #print('size after torch.cat',x1_u.size())
        x1_u = self.Upconvs[1](x1_u)
        #print('size after third deconv',x1_u.size(),'for og size',og_size)
        x1_u = self.Upconvs[2](x1_u)[:,:,:og_height,:og_width]
        x = torch.cat([x,x1_u],dim=1)
        #print('size after torch.cat',x.size())
        return self.conv1k(x)


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
