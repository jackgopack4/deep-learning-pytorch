import torch
import numpy as np

from .models import Detector, save_model, FocalLoss, DENSE_CLASS_DISTRIBUTION, INV_CLASS_DISTRIBUTION
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import torchvision

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
    
    loss = FocalLoss(gamma=args.gamma, alpha=args.alpha).to(device)
    #loss = torch.nn.BCEWithLogitsLoss().to(device)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    validation_transform=eval(args.valid_transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4, transform=validation_transform)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        loss_vals = []
        print('now training epoch',epoch)
        for img, label, size in train_data:
            img, label, size = img.to(device), label.to(device), size.to(device)
            logit = model(img)
            logit = logit.to(device)
            loss_val = loss(logit, label)
            loss_val = loss_val.to(device)
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, logit, global_step)
                train_logger.add_scalar('loss', loss_val, global_step)
                loss_vals.append(loss_val)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_loss = sum(loss_vals) / len(loss_vals)
        print('avg loss for epoch',epoch,'=',avg_loss.item())
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
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-a', '--alpha', type=float, default=1, help="frequency weight")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap()])')
    parser.add_argument('-v', '--valid_transform',
                        default='Compose([ToTensor(), ToHeatmap()])')
    args = parser.parse_args()
    train(args)