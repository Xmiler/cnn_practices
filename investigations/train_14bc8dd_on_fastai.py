import random
from functools import partial

import numpy as np

import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torch.optim import SGD
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10

from fastai.basic_train import Learner, DataBunch, LearnerCallback
from fastai.metrics import accuracy

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0xDEADFACE)
np.random.seed(0xDEADFACE)
torch.manual_seed(0xDEADFACE)

device_id = 0
# ---

BATCH_SIZE = 128

# model
model = resnet50(pretrained=False, num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.avgpool = nn.AdaptiveAvgPool2d(1)
model.to(device_id)
criterion = nn.CrossEntropyLoss()

# data
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CIFAR10(root='./data', train=True, transform=transform_train)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

valset = CIFAR10(root='./data', train=False, transform=transform_test)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

train_subset = Subset(CIFAR10(root='./data', train=True, transform=transform_test),
                      indices=np.random.permutation(np.arange(len(valset))))
train_eval_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
class MakeRandomizerConsistentOnTrainBegin(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
    def on_train_begin(self, **kwargs):
        for _ in train_eval_loader:
            pass
        for _ in val_loader:
            pass
class MakeRandomizerConsistentOnEpochEnd(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
    def on_epoch_end(self, **kwargs):
        for _ in train_eval_loader:
            pass

# ---
databunch = DataBunch(train_loader, val_loader)

opt_func = partial(SGD, lr=0.1, momentum=0.9, weight_decay=5e-4)

learner = Learner(data=databunch, model=model, opt_func=opt_func, loss_func=criterion,
                  metrics=[accuracy], true_wd=False)

learner.unfreeze()


# ---
callback_on_train_begin = MakeRandomizerConsistentOnTrainBegin(learner)
callback_on_epoch_end = MakeRandomizerConsistentOnEpochEnd(learner)
learner.fit(epochs=150, lr=0.1, wd=5e-4, callbacks=[callback_on_train_begin, callback_on_epoch_end])
learner.fit(epochs=100, lr=0.01, wd=5e-4, callbacks=[callback_on_epoch_end])
learner.fit(epochs=100, lr=0.001, wd=5e-4, callbacks=[callback_on_epoch_end])
