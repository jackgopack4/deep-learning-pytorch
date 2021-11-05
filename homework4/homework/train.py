import torch
import numpy as np

from .models import Detector, save_model, FocalLoss
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torchvision

"""
def point_in_box(pred, lbl):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return (x0 <= px) & (px < x1) & (y0 <= py) & (py < y1)


def point_close(pred, lbl, d=5):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return ((x0 + x1 - 1) / 2 - px) ** 2 + ((y0 + y1 - 1) / 2 - py) ** 2 < d ** 2


def box_iou(pred, lbl, t=0.5):
    px, py, pw2, ph2 = pred[:, None, 0], pred[:, None, 1], pred[:, None, 2], pred[:, None, 3]
    px0, px1, py0, py1 = px - pw2, px + pw2, py - ph2, py + ph2
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    iou = (abs(torch.min(px1, x1) - torch.max(px0, x0)) * abs(torch.min(py1, y1) - torch.max(py0, y0))) / \
          (abs(torch.max(px1, x1) - torch.min(px0, x0)) * abs(torch.max(py1, y1) - torch.min(py0, y0)))
    return iou > t

class PR:
    def __init__(self, min_size=20, is_close=point_in_box):
        self.min_size = min_size
        self.total_det = 0
        self.det = []
        self.is_close = is_close

    def add(self, d, lbl):
        lbl = torch.as_tensor(lbl.astype(float), dtype=torch.float32).view(-1, 4)
        d = torch.as_tensor(d, dtype=torch.float32).view(-1, 5)
        all_pair_is_close = self.is_close(d[:, 1:], lbl)

        # Get the box size and filter out small objects
        sz = abs(lbl[:, 2]-lbl[:, 0]) * abs(lbl[:, 3]-lbl[:, 1])

        # If we have detections find all true positives and count of the rest as false positives
        if len(d):
            detection_used = torch.zeros(len(d))
            # For all large objects
            for i in range(len(lbl)):
                if sz[i] >= self.min_size:
                    # Find a true positive
                    s, j = (d[:, 0] - 1e10 * detection_used - 1e10 * ~all_pair_is_close[:, i]).max(dim=0)
                    if not detection_used[j] and all_pair_is_close[j, i]:
                        detection_used[j] = 1
                        self.det.append((float(s), 1))

            # Mark any detection with a close small ground truth as used (no not count false positives)
            detection_used += all_pair_is_close[:, sz < self.min_size].any(dim=1)

            # All other detections are false positives
            for s in d[detection_used == 0, 0]:
                self.det.append((float(s), 0))

        # Total number of detections, used to count false negatives
        self.total_det += int(torch.sum(sz >= self.min_size))


    @property
    def curve(self):
        true_pos, false_pos = 0, 0
        r = []
        for t, m in sorted(self.det, reverse=True):
            if m:
                true_pos += 1
            else:
                false_pos += 1
            prec = true_pos / (true_pos + false_pos)
            recall = true_pos / self.total_det
            r.append((prec, recall))
        return r

    @property
    def average_prec(self, n_samples=11):
        import numpy as np
        pr = np.array(self.curve, np.float32)
        return np.mean([np.max(pr[pr[:, 1] >= t, 0], initial=0) for t in np.linspace(0, 1, n_samples)])


class DetectorGrader():
    #Detector

    @Case(score=5)
    def test_format(self):
        #return value
        det = self.module.load_model().eval()
        for i, (img, *gts) in enumerate(self.module.utils.DetectionSuperTuxDataset('dense_data/valid', min_size=0)):
            d = det.detect(img)
            assert len(d) == 3, 'Return three lists of detections'
            assert len(d[0]) <= 30 and len(d[1]) <= 30 and len(d[2]) <= 30, 'Returned more than 30 detections per class'
            assert all(len(i) == 5 for c in d for i in c), 'Each detection should be a tuple (score, cx, cy, w/2, h/2)'
            if i > 10:
                break


class DetectionGrader():
    #Detection model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        det = self.module.load_model().eval().to(device)

        # Compute detections
        self.pr_box = [PR() for _ in range(3)]
        self.pr_dist = [PR(is_close=point_close) for _ in range(3)]
        self.pr_iou = [PR(is_close=box_iou) for _ in range(3)]
        for img, *gts in self.module.utils.DetectionSuperTuxDataset('dense_data/valid', min_size=0):
            with torch.no_grad():
                detections = det.detect(img.to(device))
                for i, gt in enumerate(gts):
                    self.pr_box[i].add(detections[i], gt)
                    self.pr_dist[i].add(detections[i], gt)
                    self.pr_iou[i].add(detections[i], gt)

    def test_box_ap0(self, min_val=0.5, max_val=0.75):
        #Average precision (inside box c=0)
        ap = self.pr_box[0].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    def test_box_ap1(self, min_val=0.25, max_val=0.45):
        #Average precision (inside box c=1)
        ap = self.pr_box[1].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    def test_box_ap2(self, min_val=0.6, max_val=0.85):
        #Average precision (inside box c=2)
        ap = self.pr_box[2].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    def test_dist_ap0(self, min_val=0.5, max_val=0.72):
        #Average precision (distance c=0)
        ap = self.pr_dist[0].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    def test_dist_ap1(self, min_val=0.25, max_val=0.45):
        #Average precision (distance c=1)
        ap = self.pr_dist[1].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    def test_dist_ap2(self, min_val=0.6, max_val=0.85):
        #Average precision (distance c=2)
        ap = self.pr_dist[2].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

    def test_iou_ap0(self, min_val=0.5):
        #Average precision (iou > 0.5  c=0) [extra credit]
        ap = self.pr_iou[0].average_prec
        return ap >= min_val, 'AP = %0.3f' % ap

    def test_iou_ap1(self, min_val=0.3):
        #Average precision (iou > 0.5  c=1) [extra credit]
        ap = self.pr_iou[1].average_prec
        return ap >= min_val, 'AP = %0.3f' % ap

    def test_iou_ap2(self, min_val=0.6):
        #Average precision (iou > 0.5  c=2) [extra credit]
        ap = self.pr_iou[2].average_prec
        return ap >= min_val, 'AP = %0.3f' % ap
"""



def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Detector()
    model = model.to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))



    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = FocalLoss().to(device)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    print(transform)
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4,transform=transform)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        print('now training epoch',epoch)
        for img, label, size in train_data:
            img, label, size = img.to(device), label.to(device), size.to(device)

            logit = model(img)
            logit = logit.to(device)
            loss_val = loss(logit, label)
            loss_val = loss_val.to(device)
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, logit, global_step)


            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        print('now evaluating epoch',epoch)
        for img, label, size in valid_data:
            img, label, size = img.to(device), label.to(device), size.to(device)
            logit = model(img)

        if valid_logger is not None:
            log(valid_logger, img, label, logit, global_step)
        save_model(model)
        

def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap()])')

    args = parser.parse_args()
    train(args)