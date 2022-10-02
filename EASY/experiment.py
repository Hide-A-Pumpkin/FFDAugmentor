import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import os
from tqdm import tqdm as tqdm
import time
import random
from torchvision.models import resnet18,resnet50,resnet152
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.io import read_image
# from models.my_model import MyNet, train, evaluate, evaluate_test
from torchvision import datasets, transforms

class Dataset():
    def __init__(self, data, targets, transforms = [], batch_size=64, shuffle = True, device=torch.device('cuda')):
        if torch.is_tensor(data):
            self.length = data.shape[0]
            self.data = data.to(device)
        else:
            self.length = len(data)
        self.targets = torch.tensor(targets).to(device)
        assert(self.length == targets.shape[0])
        self.batch_size = batch_size
        self.transforms = transforms
        self.permutation = torch.arange(self.length)
        self.n_batches = self.length // self.batch_size + (0 if self.length % self.batch_size == 0 else 1)
        self.shuffle = shuffle
        self.dataset = list(data)
        self.label = list(targets)
    def __iter__(self):
        if self.shuffle:
            self.permutation = torch.randperm(self.length)
        for i in range(self.n_batches):
            if torch.is_tensor(self.data):
                yield self.transforms(self.data[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]), self.targets[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]
            else:
                yield torch.stack([self.transforms(self.data[x]) for x in self.permutation[i * self.batch_size : (i+1) * self.batch_size]]), self.targets[self.permutation[i * self.batch_size : (i+1) * self.batch_size]]
    def __len__(self):
        return self.n_batches
    def __getitem__(self, index):
        return self.dataset[index],self.label[index]

# transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def data_label(dataset):
    datas = []
    targets = []
    for data, traget in dataset:
        datas.append(data)
        targets.append(traget)

    return datas, targets

def iterator(data, target, transforms, forcecpu = False, shuffle = True, use_hd = False):
    return Dataset(data, target, transforms, shuffle = shuffle)

def img_preprocesser(sample):
    return 1 - sample / 255

n_shots=[1]

module_path = os.path.dirname(os.getcwd())
home_path = module_path

#data augmentation
transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.RandomErasing(p = 0.5),
        ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_data_path = '/root/oracle_fs/img/oracle_200_3_shot/FFD_train_11_block5_30_3_shot'
test_data_path =  '/root/oracle_fs/img/oracle_200_3_shot/test'
batch_size=64

dataset_transform = T.Compose([T.ToTensor()])
# print(train_data_path)
train_dataset = ImageFolder(train_data_path, transform=dataset_transform)
test_dataset = ImageFolder(test_data_path, transform=dataset_transform)
train, train_targets = data_label(train_dataset)
# train = train[0]
test, test_targets = data_label(test_dataset)
# test = test[0]
train_targets, test_targets = np.array(train_targets), np.array(test_targets)
if n_shots[0] == 1:
    norm = transforms.Normalize((0.8408, 0.8408, 0.8408), (0.3219, 0.3219, 0.3219))
if n_shots[0] == 3:
    norm = transforms.Normalize((0.8399, 0.8399, 0.8399), (0.3229, 0.3229, 0.3229))
if n_shots[0] == 5:
    norm = transforms.Normalize((0.8388, 0.8388, 0.8388), (0.3241, 0.3241, 0.3241))

train_transforms = torch.nn.Sequential(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), norm)
all_transforms = torch.nn.Sequential(norm)



train_loader = iterator(train, train_targets, transforms = transform_train, forcecpu = True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = iterator(test, test_targets, transforms = transform_test, forcecpu = True, shuffle = False)        
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# choose device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# classifier = resnet18(pretrained = True)
print(device)
print(torch.cuda.get_device_name(3))


module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

class BasicBlockRN12(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlockRN12, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope = 0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope = 0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.dropout(out, p=0.1, training=self.training, inplace=True)
        return out
    
class ResNet12(nn.Module):
    def __init__(self, feature_maps, input_shape, num_classes):
        super(ResNet12, self).__init__()        
        layers = []
        layers.append(BasicBlockRN12(input_shape[0], feature_maps))
        layers.append(BasicBlockRN12(feature_maps, int(2.5 * feature_maps)))
        layers.append(BasicBlockRN12(int(2.5 * feature_maps), 5 * feature_maps))
        layers.append(BasicBlockRN12(5 * feature_maps, 10 * feature_maps))        
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(10 * feature_maps, num_classes)
        self.linear_rot = nn.Linear(10 * feature_maps, 4)
        self.mp = nn.MaxPool2d((2,2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, index_mixup = None, lam = -1):
        mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = self.mp(F.leaky_relu(out, negative_slope = 0.1))
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.dropout(out, p=0.1, training=self.training, inplace=True)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_maps, input_shape, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = feature_maps
        self.length = len(num_blocks)
        self.conv1 = nn.Conv2d(input_shape[0], feature_maps, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_maps)
        layers = []
        for i, nb in enumerate(num_blocks):
            layers.append(self._make_layer(block, (2 ** i) * feature_maps, nb, stride = 1 if i == 0 else 2))
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear((2 ** (len(num_blocks) - 1)) * feature_maps, num_classes)
        self.linear_rot = nn.Linear((2 ** (len(num_blocks) - 1)) * feature_maps, 4)
        self.depth = len(num_blocks)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            if i < len(strides) - 1:
                layers.append(nn.ReLU())
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, index_mixup = None, lam = -1):
        mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
            out = F.relu(out)
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        out = self.linear(features)
        return out

def ResNet18(feature_maps, input_shape, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], feature_maps, input_shape, num_classes)

def ResNet20(feature_maps, input_shape, num_classes):
    return ResNet(BasicBlock, [3, 3, 3], feature_maps, input_shape, num_classes)

class BasicBlockWRN(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super(BasicBlockWRN, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, feature_maps, input_shape, depth = 28, widen_factor = 10, num_classes = 200, drop_rate =0.1):
        super(WideResNet, self).__init__()
        nChannels = [feature_maps, feature_maps*widen_factor, 2 * feature_maps*widen_factor, 4 * feature_maps*widen_factor]
        n = (depth - 4) / 6
        self.conv1 = nn.Conv2d(input_shape[0], nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(NetworkBlock(n, nChannels[0], nChannels[1], BasicBlockWRN, 1, drop_rate))
        self.blocks.append(NetworkBlock(n, nChannels[1], nChannels[2], BasicBlockWRN, 2, drop_rate))
        self.blocks.append(NetworkBlock(n, nChannels[2], nChannels[3], BasicBlockWRN, 2, drop_rate))
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.linear = nn.Linear(nChannels[3], int(num_classes))
        # self.rotations = rotations
        self.linear_rot = nn.Linear(nChannels[3], 4)

    def forward(self, x, index_mixup = None, lam = -1):
        # mixup_layer = -1
        if lam != -1:
            mixup_layer = random.randint(0, 3)
        else:
            mixup_layer = -1
        out = x
        if mixup_layer == 0:
            out = lam * out + (1 - lam) * out[index_mixup]
        out = self.conv1(out)
        for i in range(len(self.blocks)):
            out = self.blocks[i](out)
            if mixup_layer == i + 1:
                out = lam * out + (1 - lam) * out[index_mixup]
        out = torch.relu(self.bn(out))
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        features = out
        out = self.linear(features)
        return out

def train(model, device, train_dataloader, optimizer,scheduler, epoch, loss_fn):
    model.train()
    total_loss = 0.
    correct = 0.
    total_len = len(train_dataloader.dataset)

    for idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        preds = model(data) # batch_size * 10
        # print('pred',preds)
        loss = loss_fn(preds, target)

        # print('loss',loss)
        total_loss += loss_fn(preds, target).item()
        pred = preds.argmax(dim = 1)
        correct += pred.eq(target.view_as(pred)).sum().item()
        # loss = F.nll_loss(preds,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if idx % 1000 == 0:
            print("Loss: {}".format(loss.item()))

    total_loss = total_loss / total_len
    acc = correct/total_len
    return total_loss, acc

def evaluate(model, device, valid_dataloader,loss_fn, flag):
    model.eval()
    total_loss = 0.
    correct = 0.
    total_len = len(valid_dataloader.dataset)
    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) # batch_size * 1
            total_loss += loss_fn(output, target).item()
            pred = output.argmax(dim = 1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    total_loss = total_loss / total_len
    acc = correct/total_len
    if flag == 1:
        f=open('why.txt','a')
        f.write("Accuracy on test set:{}\n".format(acc))
        f.close()
        print("Accuracy on test set:{}".format(acc)) 
    else:
        f=open('why.txt','a')
        f.write("valid loss:{}, Accuracy:{}".format(total_loss, acc))
        f.close()
        print("valid loss:{}, Accuracy:{}".format(total_loss, acc)) 
    return total_loss, acc

# hyper parameter
lr = 0.04
momentum = 0.9
weight_decay = 4e-4
num_features = 3
hidden_size = 1024
output_size = 200
loss_fn = nn.CrossEntropyLoss()

model_save_path = "output/baseline_res18.pth"

model = WideResNet(16, [3, 50, 50], num_classes = output_size)
# model = resnet152(pretrained = False)
# model.fc = torch.nn.Linear(2048, 200)
model = model.to(device)
#选择一个optimizer
#SGD/ADAM
# optimizer2 = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
optimizer2 = torch.optim.SGD(model.parameters(), lr = lr, momentum=momentum, weight_decay=weight_decay)
#cyclicLR
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=0.01, max_lr=0.04)

starttime = time.time()
num_epochs = 100
total_loss = []
train_loss= []
acc = []
train_acc =[]

train_loss1, train_acc1 = train(model, device, train_dataloader,optimizer2, scheduler, 0, loss_fn)

for epoch in range(1,num_epochs):
    f=open('why.txt','a')
    f.write("Epoch:%d \n"%epoch)
    f.close()
    print("Epoch:%d"%epoch,end=' ')
    train_loss1, train_acc1 = train(model, device, train_dataloader, optimizer2, scheduler, epoch, loss_fn)
    
    if epoch> num_epochs-90:
        total_loss_0, acc_0 = evaluate(model, device, test_dataloader,loss_fn, 0)
        # torch.save(model.state_dict(),model_save_path)
        total_loss.append(total_loss_0)
        acc.append(acc_0)

torch.save(model.state_dict(),model_save_path)
evaluate(model, device, test_dataloader,loss_fn, 1)

endtime = time.time()
dtime = endtime - starttime
print("Finish! run time: %.8s s" % dtime)
print('Highest test result:', max(acc))

f=open('why.txt','a')
f.write("Finish! run time: %.8s s \n" % dtime)
f.write('Highest test result: %s\n'%( str(max(acc))))
f.close()