import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
import torchvision


def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = FCN()
    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'fcn.th')))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=10,cooldown=5)
    loss = torch.nn.CrossEntropyLoss()

    transform = torchvision.transforms.Compose([
      torchvision.transforms.ColorJitter(brightness=1,contrast=1,saturation=1,hue=.25),
      torchvision.transforms.RandomHorizontalFlip(),
      #torchvision.transforms.RandomResizedCrop(64)
    ])

    train_data = load_dense_data('dense_data/train')
    valid_data = load_dense_data('dense_data/valid')

    max_viou = 0.0
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        # do training
        c_train = ConfusionMatrix()
        for img, labels in train_data:
            img, labels = img.to(device), labels.to(device)
            transformed_img=transform(img)
            logit = model(transformed_img).to(device)
            loss_val = loss(logit, labels)
            c_train.add(logit.argmax(1),labels)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        print( 'epoch = ', epoch, 'optimizer_lr', optimizer.param_groups[0]['lr'])
        train_iou = c_train.iou().detach().cpu().numpy()
        if train_logger:
            train_logger.add_scalar('iou_accuracy', train_iou, global_step)


        model.eval()
        # do evaluation
        c_valid = ConfusionMatrix()
        for img, labels in valid_data:
            img, labels = img.to(device), labels.to(device)
            c_valid.add(model(img).argmax(1), labels)
        valid_iou = c_valid.iou().detach().cpu().numpy()
        scheduler.step(valid_iou)

        if valid_logger:
            valid_logger.add_scalar('iou_accuracy', valid_iou, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, train_iou, valid_iou))
        if(valid_iou > max_viou): 
          max_viou = valid_iou
          save_model(model)
          print('saving model with viou',max_viou)
    #save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)
