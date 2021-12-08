from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
from torchvision.transforms import ToTensor
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW4 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))
    
    puck_loss = torch.nn.BCEWithLogitsLoss()
    loc_loss = torch.nn.L1Loss()
    dist_loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.25)
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    train_data = load_data('train.pkl', transform=transform, num_workers=args.num_workers)
    valid_data = load_data('valid.pkl', transform=dense_transforms.ToTensor(),num_workers=args.num_workers)
    global_step = 0
    lowest_loss_val = 2
    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        puck_losses = []
        loc_losses = []
        dist_losses = []
        for data in train_data:
            img = data[0].to(device)
            puck = data[1].to(device)
            loc = data[2].to(device)
            dist = data[3].to(device)
            pred_puck, pred_loc = model(img)
            puck_loss_val = puck_loss(pred_puck[:,0], puck)
            loc_loss_val = loc_loss(pred_loc, loc)
            dist_loss_val = dist_loss(pred_puck[:,1],dist)
            loss_val = puck_loss_val + dist_loss_val*args.loss_weight*2.5 + loc_loss_val * args.loss_weight

            if train_logger is not None:
                train_logger.add_scalar('puck_loss', puck_loss_val, global_step)
                train_logger.add_scalar('loc_loss', loc_loss_val, global_step)
                train_logger.add_scalar('dist_loss',dist_loss_val,global_step)
                train_logger.add_scalar('loss', loss_val, global_step)
                if global_step % 20 == 0:
                    log(train_logger, img, loc, pred_loc, puck, dist, pred_puck, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            
            losses.append(loss_val.detach().cpu().numpy())
            puck_losses.append(puck_loss_val.detach().cpu().numpy())
            loc_losses.append(loc_loss_val.detach().cpu().numpy())
            dist_losses.append(dist_loss_val.detach().cpu().numpy())
        #training losses
        avg_loss = np.mean(losses)
        avg_puck_loss = np.mean(puck_losses)
        avg_loc_loss = np.mean(loc_losses)
        avg_dist_loss = np.mean(dist_losses)
        print('epoch %-3d training: \t puck_loss = %0.3f \t dist_loss = %0.3f \t loc_loss = %0.3f \t total_loss = %0.3f' % (epoch, avg_puck_loss, avg_dist_loss, avg_loc_loss, avg_loss))
        
        model.eval()
        losses = []
        puck_losses = []
        loc_losses = []
        dist_losses = []
        for data in valid_data:
            img  = data[0].to(device)
            puck = data[1].to(device)
            loc  = data[2].to(device)
            dist = data[3].to(device)
            pred_puck, pred_loc = model(img)
            
            puck_loss_val = puck_loss(pred_puck[:,0], puck)
            loc_loss_val = loc_loss(pred_loc, loc)
            dist_loss_val = dist_loss(pred_puck[:,1], dist)
            loss_val = puck_loss_val + dist_loss_val*args.loss_weight*2.5 + loc_loss_val * args.loss_weight
            
            if valid_logger is not None:
                valid_logger.add_scalar('puck_loss', puck_loss_val, global_step)
                valid_logger.add_scalar('loc_loss', loc_loss_val, global_step)
                valid_logger.add_scalar('dist_loss', dist_loss_val, global_step)
                valid_logger.add_scalar('loss', loss_val, global_step)
                if global_step % 20 == 0:
                    log(valid_logger, img, loc, pred_loc, puck, dist, pred_puck, global_step)
            
            global_step += 1
            losses.append(loss_val.detach().cpu().numpy())
            puck_losses.append(puck_loss_val.detach().cpu().numpy())
            loc_losses.append(loc_loss_val.detach().cpu().numpy())
            dist_losses.append(dist_loss_val.detach().cpu().numpy())

        avg_loss = np.mean(losses)
        avg_puck_loss = np.mean(puck_losses)
        avg_loc_loss = np.mean(loc_losses)
        avg_dist_loss = np.mean(dist_losses)
        print('epoch %-3d validation: \t puck_loss = %0.3f \t dist_loss = %0.3f \t loc_loss = %0.3f \t total_loss = %0.3f' % (epoch, avg_puck_loss, avg_dist_loss, avg_loc_loss, avg_loss))
        
        scheduler.step()
        if avg_loc_loss < lowest_loss_val:
            save_model(model)
            lowest_loss_val = avg_loc_loss

    #save_model(model)

def log(logger, img, loc, pred_loc, puck, dist, pred_puck, global_step):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.annotate('Puck: %s distance: %0.3f' % (puck[0].cpu().detach().numpy(),dist[0].cpu().detach().numpy()),(25.,25.),color='green')
    ax.annotate('Predicted: %0.3f distance: %0.3f' % (pred_puck[0][0].cpu().detach().numpy(),pred_puck[0][1].cpu().detach().numpy()), (25.,50),color='red')
    ax.add_artist(plt.Circle(WH2*(loc[0].cpu().detach().numpy()+1), 10, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred_loc[0].cpu().detach().numpy()+1), 10, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), ToTensor()])')
    parser.add_argument('-l', '--loss_weight', type=float, default=0.01)
    parser.add_argument('-s', '--step_size',type=int,default=50)

    args = parser.parse_args()
    train(args)