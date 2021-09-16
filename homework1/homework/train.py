from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
# adapted heavily from PyTorch "Training a Classifier" tutorial
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def train(args):
  model = model_factory[args.model]()
  trainloader = load_data('data/train')
  testloader = load_data('data/valid')
  loss = ClassificationLoss()
  optimizer = torch.optim.SGD(model.parameters(),lr=.006)
  break_point=.1
  if(args.model=='linear'): break_point=.2
  scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
  for epoch in range(30):
    size_train = len(trainloader.dataset)
    #train loop
    for batch, (X_train, y_train) in enumerate(trainloader):
      pred_train = model(X_train)
      loss_train = loss(pred_train, y_train)
      optimizer.zero_grad()
      loss_train.backward()
      optimizer.step()
      if batch % 32 == 0:
        loss_train, current = loss_train.item(), batch * len(X_train)
        print(f"loss: {loss_train:>7f}  [{current:>5d}/{size_train:>5d}]")
    scheduler1.step()
    #test loop
    size_test = len(testloader.dataset)
    num_batches = len(testloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
      for X, y in testloader:
        pred = model(X)
        test_loss += loss(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size_test
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  save_model(model)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
