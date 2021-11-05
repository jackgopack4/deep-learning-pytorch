import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torchvision

def average_prec(n_samples=11):
        import numpy as np
        pr = np.array(self.curve, np.float32)
        return np.mean([np.max(pr[pr[:, 1] >= t, 0], initial=0) for t in np.linspace(0, 1, n_samples)])

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


def test_box_ap0(min_val=0.5, max_val=0.75):
        """Average precision (inside box c=0)"""
        ap = self.pr_box[0].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

def test_box_ap1(min_val=0.25, max_val=0.45):
        """Average precision (inside box c=1)"""
        ap = self.pr_box[1].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

def test_box_ap2(min_val=0.6, max_val=0.85):
        """Average precision (inside box c=2)"""
        ap = self.pr_box[2].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

def test_dist_ap0(min_val=0.5, max_val=0.72):
        """Average precision (distance c=0)"""
        ap = self.pr_dist[0].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

def test_dist_ap1(min_val=0.25, max_val=0.45):
        """Average precision (distance c=1)"""
        ap = self.pr_dist[1].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

def test_dist_ap2(min_val=0.6, max_val=0.85):
        """Average precision (distance c=2)"""
        ap = self.pr_dist[2].average_prec
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val), 'AP = %0.3f' % ap

def test_iou_ap0(min_val=0.5):
        """Average precision (iou > 0.5  c=0) [extra credit]"""
        ap = self.pr_iou[0].average_prec
        return ap >= min_val, 'AP = %0.3f' % ap

def test_iou_ap1(min_val=0.3):
        """Average precision (iou > 0.5  c=1) [extra credit]"""
        ap = self.pr_iou[1].average_prec
        return ap >= min_val, 'AP = %0.3f' % ap

def test_iou_ap2(min_val=0.6):
        """Average precision (iou > 0.5  c=2) [extra credit]"""
        ap = self.pr_iou[2].average_prec
        return ap >= min_val, 'AP = %0.3f' % ap


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
    #w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)**(-args.gamma)
    loss = torch.nn.BCEWithLogitsLoss().to(device)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    print(transform)
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4,transform=transform)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        #conf = ConfusionMatrix()
        for img, label, size in train_data:
            img, label, size = img.to(device), label.to(device), size.to(device)
            #i, peaks, s = dense_transforms.ToHeatmap(img)
            #img_tensor = dense_transforms.ToTensor(img)
            #img_tensor, peaks = img_tensor.to(device), peaks.to(device).long()

            logit = model(img)
            loss_val = loss(logit, label)
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, logit, global_step)

            #conf.add(logit.argmax(1), label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        #val_conf = ConfusionMatrix()
        for img, label, size in valid_data:
            img, label, size = img.to(device), label.to(device), size.to(device)
            #i, peaks, s = dense_transforms.ToHeatmap(img)
            #img_tensor = dense_transforms.ToTensor(img)
            #img_tensor, peaks = img_tensor.to(device), peaks.to(device).long()
            logit = model(img)
            #val_conf.add(logit.argmax(1), label)

        if valid_logger is not None:
            log(valid_logger, img, label, logit, global_step)
        save_model(model)
        """        
        pr_box = [PR() for _ in range(3)]
        pr_dist = [PR(is_close=point_close) for _ in range(3)]
        pr_iou = [PR(is_close=box_iou) for _ in range(3)]
        for img, *gts in load_detection_data('dense_data/valid', min_size=0,transform=transform):
            with torch.no_grad():
                detections = model.detect(img.to(device))
                for i, gt in enumerate(gts):
                    pr_box[i].add(detections[i], gt)
                    pr_dist[i].add(detections[i], gt)
                    pr_iou[i].add(detections[i], gt)
        
        Average precision (inside box c=0)
        ap = average_prec(pr_box[0])
        print('Average precision (inside box c=0)',max(min(ap, max_val) - min_val, 0) / (max_val - min_val))

        print('Average precision (inside box c=1)',d.test_box_ap1)
        print('Average precision (inside box c=2)',d.test_box_ap2)
        print('Average precision (distance c=0)',d.test_dist_ap0)
        print('Average precision (distance c=1)',d.test_dist_ap1)
        print('Average precision (distance c=2)',d.test_dist_ap2)
        print('Average precision (iou > 0.5  c=0) [extra credit]',d.test_iou_ap0)
        print('Average precision (iou > 0.5  c=1) [extra credit]',d.test_iou_ap1)
        print('Average precision (iou > 0.5  c=2) [extra credit]',d.test_iou_ap2)"""
        

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