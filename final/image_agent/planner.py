import torch
import torch.nn.functional as F

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 32, 32]):
        super().__init__()

        conv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, 5, 2, 2), torch.nn.ReLU(True)]

        h, _conv = 3, []
        for c in channels:
            _conv += conv_block(c, h)
            h = c

        self._conv = torch.nn.Sequential(*_conv)
        self.classifier = torch.nn.Linear(h-7, 1)
        self.location = torch.nn.Conv2d(h, 1, 1)

    def forward(self, img):
        """
        Your code here
        Predict if puck in image and puck location in image coordinate, given the supertuxkart image
        @img: (B,3,300,400)
        return [(B,1),(B,2)]
        """
        #print('conv network:',self._conv)
        #print('linear classifier:',self.classifier)
        #print('conv classifier:',self.location)
        x = self._conv(img)
        #print('conv model',x)
        puck = self.classifier(x)
        puck = puck.mean(dim=[1,2,3])
        loc = spatial_argmax(self.location(x)[:, 0])
        return puck, loc
        # return self.classifier(x.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r