import os
from datetime import datetime
import random

import numpy as np

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
from tensorboardX import SummaryWriter
from ignite.engine import Events
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.handlers import ModelCheckpoint

from utils.optim import AdamW, SGDW
from utils.scheduler import DefaultSchedulerFastAI, Linear


def print_with_time(string):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time + ' - ' + string)


print(' ================= Initialization ================= ')
EXPERIMENT_NAME = 'cifar_resnet50'
# --->>> Service parameters
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0xDEADFACE)
np.random.seed(0xDEADFACE)
torch.manual_seed(0xDEADFACE)

output_path = './snapshots/'
writer = SummaryWriter(os.path.join(output_path, 'tensorboard', EXPERIMENT_NAME))
log_interval = 50
device = "cuda"

# --->>> Training parameters
BATCH_SIZE = 128
MAX_EPOCHS = 350
BASE_LR = 0.01
WD = 5e-3

# model
model = resnet50(pretrained=False, num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.avgpool = nn.AdaptiveAvgPool2d(1)
model.to(device=device)

optimizer = AdamW(model.parameters(), lr=BASE_LR, betas=(0.95, 0.99), weight_decay=WD)

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


# --->>> Callbacks
def log_loss_during_training(engine):
    iteration_on_epoch = (engine.state.iteration - 1) % len(train_loader) + 1
    if iteration_on_epoch % log_interval == 0:
        print_with_time("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}"
                        .format(engine.state.epoch, iteration_on_epoch,
                                len(train_loader), engine.state.output))
        writer.add_scalars('during_training', {'loss': engine.state.output},
                           global_step=engine.state.iteration)


metrics = {'avg_loss': Loss(criterion), 'avg_accuracy': Accuracy()}
evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


def compute_and_log_metrics_on_train(engine):
    epoch = engine.state.epoch
    print_with_time('Compute metrics ..')
    metrics = evaluator.run(train_eval_loader).metrics
    print_with_time("Training Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f}"
                    .format(engine.state.epoch, metrics['avg_loss'], metrics['avg_accuracy']))
    writer.add_scalars('loss', {'train': metrics['avg_loss']}, global_step=epoch)
    writer.add_scalars('accuracy', {'train': metrics['avg_accuracy']}, global_step=epoch)


def compute_and_log_metrics_on_val(engine):
    epoch = engine.state.epoch
    metrics = evaluator.run(val_loader).metrics
    print_with_time("Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f}"
                    .format(engine.state.epoch, metrics['avg_loss'], metrics['avg_accuracy']))
    writer.add_scalars('loss', {'validation': metrics['avg_loss']}, global_step=epoch)
    writer.add_scalars('accuracy', {'validation': metrics['avg_accuracy']}, global_step=epoch)


# --->>> Trainer
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

# attach callbacks
trainer.add_event_handler(Events.STARTED, compute_and_log_metrics_on_train)
trainer.add_event_handler(Events.STARTED, compute_and_log_metrics_on_val)

trainer.add_event_handler(Events.ITERATION_STARTED, Linear(optimizer, BASE_LR, MAX_EPOCHS, len(train_loader), writer))
trainer.add_event_handler(Events.ITERATION_STARTED, log_loss_during_training)

trainer.add_event_handler(Events.EPOCH_COMPLETED, compute_and_log_metrics_on_train)
trainer.add_event_handler(Events.EPOCH_COMPLETED, compute_and_log_metrics_on_val)

trainer.add_event_handler(Events.COMPLETED,
                          ModelCheckpoint(output_path, 'iteration', save_interval=1),
                          {"model": model, "optimizer": optimizer})

print(' ================= Training! ================= ')
trainer.run(train_loader, max_epochs=MAX_EPOCHS)
