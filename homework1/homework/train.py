from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
# adapted heavily from PyTorch "Training a Classifier" tutorial
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

def train(args):
  model = model_factory[args.model]()
  trainloader = load_data('data/train')
  loss = ClassificationLoss()
  optimizer = torch.optim.SGD(model.parameters(),lr=.0025)
  break_point=.1
  if(args.model=='linear'): break_point=.2
  #print(args.model)
  for epoch in range(25):
    #testloader = load_data('data/valid')
    #classifier=locals()[model]()
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data
      optimizer.zero_grad()
      #print(len(inputs))
      #print(len(labels))
      outputs=model(inputs)
      l = loss(outputs,labels)
      l.backward()
      optimizer.step()
    print(l.item())
    
    if(l.item()<break_point): 
      break
    """
    Your code here

    """
    #raise NotImplementedError('train')

  save_model(model)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
