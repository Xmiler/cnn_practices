import os
from datetime import datetime

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.optim import SGD
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
from ignite.engine import Events
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter


def print_with_time(string):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(time + ' - ' + string)


print(' ================= Initialization ================= ')
EXPERIMENT_NAME = 'cifar_resnet50'

# --->>> Service parameters
np.random.seed(0xDEADFACE)
output_path = './snapshots/'
writer = SummaryWriter(os.path.join(output_path, 'tensorboard', EXPERIMENT_NAME))
log_interval = 50
device = "cuda"


# --->>> Training parameters
BATCH_SIZE = 128
MAX_EPOCHS = 15


def adjust_learning_rate(optimizer, epoch):
    if epoch < 5:
        lr = 0.1
    elif epoch < 10:
        lr = 0.01
    elif epoch < MAX_EPOCHS:
        lr = 0.01
    else:
        assert False
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# model
model = resnet50(pretrained=False, num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.avgpool = nn.AdaptiveAvgPool2d(1)
model.to(device=device)

optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

criterion = nn.CrossEntropyLoss()

# data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CIFAR10(root='./data', train=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

valset = CIFAR10(root='./data', train=False, transform=transform)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

rnd_indcs = np.random.permutation(np.arange(len(valset)))
train_subset = Subset(trainset, indices=rnd_indcs)
train_eval_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# --->>> Callbacks
def update_lr_scheduler(engine):
    lr = adjust_learning_rate(optimizer, engine.state.epoch)
    print_with_time("Learning rate: {}".format(lr))
    writer.add_scalar('lr', lr, global_step=engine.state.epoch)


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
trainer.add_event_handler(Events.EPOCH_STARTED, update_lr_scheduler)

trainer.add_event_handler(Events.ITERATION_STARTED, log_loss_during_training)

trainer.add_event_handler(Events.EPOCH_STARTED, compute_and_log_metrics_on_train)
trainer.add_event_handler(Events.EPOCH_STARTED, compute_and_log_metrics_on_val)

trainer.add_event_handler(Events.COMPLETED, compute_and_log_metrics_on_train)
trainer.add_event_handler(Events.COMPLETED, compute_and_log_metrics_on_val)
trainer.add_event_handler(Events.COMPLETED,
                          ModelCheckpoint('snapshots/', 'iteration', save_interval=1),
                          {"model": model, "optimizer": optimizer})

print(' ================= Training! ================= ')
trainer.run(train_loader, max_epochs=MAX_EPOCHS)
