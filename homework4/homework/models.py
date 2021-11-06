import torch
import torch.nn.functional as F
from torch.autograd import Variable

DENSE_CLASS_DISTRIBUTION = [0.02929112, 0.0044619, 0.00411153]
INV_CLASS_DISTRIBUTION = [0.06807880165, 0.4469182071, 0.4850029912]

def convert_index_to_coordinates(index,width):
    y = index//width
    x = index - y*width
    return torch.tensor(x),torch.tensor(y)

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    m = torch.nn.MaxPool2d(kernel_size=max_pool_ks,stride=1,padding=max_pool_ks//2)
    output = m(heatmap[None,None])
    output = output.to(heatmap.device)
    output = torch.squeeze(output)
    new_heatmap = torch.where(heatmap>min_score,heatmap,torch.tensor(0.).to(output.device)).to(output.device)
    comparison = heatmap>=output
    comparison.to(output.device)
    res = torch.where(comparison,new_heatmap,torch.tensor(0.).to(output.device))
    res = res.to(output.device)
    nonzero = torch.nonzero(res,as_tuple=True)
    nums = res[nonzero]
    indices = range(0,len(nums))
    if len(nums) > max_det:
        values,indices = torch.topk(nums,k=max_det,dim=0,sorted=False)
    peaks = list(zip(nums[indices],nonzero[1][indices],nonzero[0][indices]))
    return peaks
# adapted from Focal Loss Lin et. al paper https://github.com/clcarwin/focal_loss_pytorch/
class FocalLoss(torch.nn.Module):
    
    def __init__(self, alpha=1., 
                 gamma=0., reduction='none'):
        torch.nn.Module.__init__(self)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        input = input.contiguous().view(input.size(0), input.size(1), -1)
        input = input.transpose(1,2)
        input = input.contiguous().view(-1, input.size(2)).squeeze()
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1,2)
        target = target.contiguous().view(-1, target.size(2)).squeeze()
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction,pos_weight=torch.tensor([.69,.95,.85]).to(target.device))
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
        
class Detector(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=3, kernel_size=3, use_skip=True):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)

        #raise NotImplementedError('Detector.__init__')

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d'%i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d'%i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z)
        #raise NotImplementedError('Detector.forward')

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = self.forward(image).to(device)
        model = model.squeeze(dim=0)
        res = []
        
        for i in range(0,3):
            peaks = extract_peak(model[i],max_det=30)
            channel_list = []
            for p in peaks: 
                channel_list.append([p[0].item(),p[1].item(),p[2].item(),0.,0.])
            res.append(channel_list)
        return res


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
