%matplotlib inline

import math
import os
import time

import matplotlib as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet34

from utils.helpers import AverageMeter

# preparations
data_path = '~/play-pytorch/data/dogscats/train/' 
pics = os.listdir(data_path + 'dogs')
model = resnet34(True)

# loading the data
train_data = datasets.ImageFolder(
    data_path,
    transforms.Compose(
        [
            transforms.RandomResizedCrop(256),
            transforms.ToTensor()
        ]
    )
)
train_loader = DataLoader(
    train_data,
    batch_size=256,
    num_workers=4,
    pin_memory=True
)
optimizer = torch.optim.SGD(
    model.parameters(),
    0.1,
    momentum=0,
    weight_decay=0
)
criterion = nn.CrossEntropyLoss()
cuda = torch.cuda.is_available()


# actually training the model
batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
model.train()

end = time.time()
for i, (input, target) in enumerate(train_loader):
    # measure data loading time
    logging.info('data loaded')
    if cuda:
        input, target = input.cuda(async=True), target.cuda(async=True)

    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    # compute output
    output = model(input_var)
    #topk = (1,5) if labels >= 100 else (1,) # TO FIX
    # For nets that have multiple outputs such as Inception
    if isinstance(output, tuple):
        loss = sum((criterion(o, target_var) for o in output))
        # print (output)
        for o in output:
            prec1 = accuracy(o.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0)*len(output))
    else:
        loss = criterion(output, target_var)
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        losses.update(loss.data[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
