import torch
import torch.nn.functional as F

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
    
    m = torch.nn.MaxPool2d(kernel_size=max_pool_ks,stride=1,padding=max_pool_ks//2,return_indices=True)

    maxpools, indices = m(heatmap[None,None])
    flattened_indices = torch.flatten(indices)
    flattened_maxpools = torch.flatten(maxpools)
    flattened_length = torch.tensor(range(0,len(flattened_indices)))
    peaks = []
    width = len(heatmap[0])
    for i in range(0,len(flattened_indices)):
        if flattened_indices[i] == i: 
            if flattened_maxpools[i].item()>min_score:
                x,y = convert_index_to_coordinates(i,width)
                peaks.append([flattened_maxpools[i],x,y])

    if len(peaks) > max_det:
        peaks.sort(key=lambda x: x[0])
        remove_indices = len(peaks) - max_det
        for j in range(0,remove_indices):
            peaks.pop(0)
    """
    m = torch.nn.MaxPool2d(kernel_size=max_pool_ks,stride=1,padding=max_pool_ks//2)
    output = m(heatmap[None,None])
    output = torch.squeeze(output)
    heatmap = torch.where(heatmap>min_score,heatmap,torch.tensor(0.))
    print('new heatmap',heatmap)
    comparison = heatmap>=output
    res = torch.where(comparison,heatmap,torch.tensor(0.))
    print('res',res)
    nonzero = torch.nonzero(res,as_tuple=True)
    print('nonzero',nonzero)
    nums = res[nonzero]
    print('called nums with tuple')
    peaks = list(zip(nums,nonzero[0],nonzero[1]))
    peak_tensor=torch.tensor(peaks)

    if len(peak_tensor) > max_det:
        print('peak size',len(peak_tensor),'greater than max det',max_det)
        values,indices = torch.topk(nums,k=max_det,dim=0,sorted=False)
        print('called topk')
        print('indices type',indices.type())
        peak_tensor = peak_tensor[indices]
    return peak_tensor

class Detector(torch.nn.Module):
    def __init__(self):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        raise NotImplementedError('Detector.__init__')

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        raise NotImplementedError('Detector.forward')

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
        raise NotImplementedError('Detector.detect')


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
