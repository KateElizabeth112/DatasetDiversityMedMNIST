import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Subset
import torchvision.transforms as transforms
from medmnist import INFO
from evaluatorLocal import Evaluator
from models import ResNet18, ResNet50
from torchvision.models import resnet18, resnet50
from tqdm import trange


def runTraining(train_dataset, val_dataset, test_dataset, data_flag, output_root, num_epochs, batch_size, size, model_flag, resize):
    lr = 0.001
    gamma = 0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3
    n_classes = len(info['label'])

    device = torch.device('cpu')

    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

    print('==> Building and training model for {} epochs...'.format(num_epochs))

    if model_flag == 'resnet18':
        model = resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels,
                                                                                          num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels,
                                                                                          num_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)

    train_evaluator = Evaluator(data_flag, 'train', size=size)
    val_evaluator = Evaluator(data_flag, 'val', size=size)
    test_evaluator = Evaluator(data_flag, 'test', size=size)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if num_epochs == 0:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_' + log for log in logs]
    val_logs = ['val_' + log for log in logs]
    test_logs = ['test_' + log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)
    train_loss = []

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        train_loss.append(train(model, train_loader, task, criterion, optimizer, device))

        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device)

        scheduler.step()

        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)

    metrics = {"train_AUC": train_metrics[1],
               "train_acc": train_metrics[2],
               "val_AUC": val_metrics[1],
               "val_acc": val_metrics[2],
               "test_AUC": test_metrics[1],
               "test_acc": test_metrics[2]}

    return metrics


def train(model, train_loader, task, criterion, optimizer, device):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        total_loss.append(loss.item())

        if iteration % 10 == 0:
            print('train_loss {0:.3f}, iter: {1}'.format(loss.item(), iteration))
        iteration += 1

        loss.backward()
        optimizer.step()

    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, criterion, device, run=None, save_folder=None):
    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)
    y_targets = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
            y_targets = torch.cat((y_targets, targets))

        y_score = y_score.detach().cpu().numpy()
        y_targets = y_targets.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, y_targets)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default='chestmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--size',
                        default=28,
                        help='the image size of the dataset, 28 or 64 or 128 or 224, default=28',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=50,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--root',
                        default="/Users/katecevora/Documents/PhD/data/MedMNIST",
                        help='Root directory of dataset',
                        type=str)

    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    size = args.size
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = args.download
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = args.as_rgb
    #model_path = args.model_path
    #run = args.run
    root = args.root
    subset_idx = np.arange(0, 1000)

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[.5], std=[.5])])

    train_dataset = Subset(
        DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb, size=size, root=root),
        subset_idx)

    val_dataset = Subset(DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb, size=size, root=root), np.arange(0,200))
    test_dataset = Subset(DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb, size=size, root=root), np.arange(0, 200))

    metrics = runTraining(train_dataset,
                          val_dataset,
                          test_dataset,
                          data_flag,
                          output_root,
                          num_epochs,
                          batch_size,
                          size,
                          model_flag,
                          resize)
