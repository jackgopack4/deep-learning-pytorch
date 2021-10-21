from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES, accuracy
import torch
import torchvision
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNNClassifier()
    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=5)
    loss = torch.nn.CrossEntropyLoss()

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    transform = torchvision.transforms.Compose([
      torchvision.transforms.ColorJitter(),
      torchvision.transforms.RandomHorizontalFlip,
      torchvision.transforms.RandomResizedCrop(64)
    ])
    
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        acc_vals = []
        transform(train_data[0][:])
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            acc_val = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_acc = sum(acc_vals) / len(acc_vals)
        print( 'epoch = ', epoch, 'optimizer_lr', optimizer.param_groups[0]['lr'])
        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            acc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        avg_vacc = sum(acc_vals) / len(acc_vals)
        scheduler.step(avg_vacc)

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
        save_model(model)
    save_model(model)


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
