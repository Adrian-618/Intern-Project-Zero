import pickle

from sklearn.utils import shuffle
import torch
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
import numpy as np
import sys

def dataloader(dataset="cifar", batch_size_train=64, batch_size_test=1000, split_dataset=1):
    # split_dataset is used to mark the usage of the dataset, 0 == testdata, 1 == shadow_train, 2 == shadow_out, 3 = target_train, 4 = target_out

    if dataset == "cifar":

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    elif dataset == "mnist":

        transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root="../data", train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False)

    total_size = len(trainset)

    if split_dataset == 0:
        return testloader

    else:
        split_size=int(0.25*total_size)
        #first param is data set to be saperated, the second is list stating how many sets we want it to be.
        torch.manual_seed(42)
        shadow_set,target_set, shadow_out, target_out=random_split(trainset,[split_size,split_size,split_size,split_size])
        shadow_set_loader = DataLoader(shadow_set, shuffle=True, batch_size=batch_size_train)
        shadow_out_loader = DataLoader(shadow_out, shuffle=False, batch_size=batch_size_train)
        target_set_loader = DataLoader(target_set, shuffle=True, batch_size=batch_size_train)
        target_out_loader = DataLoader(target_out, shuffle=False, batch_size=batch_size_train)
        return shadow_set_loader, shadow_out_loader,target_set_loader, target_out_loader

def getData(dataset="cifar", batch_size_train=64, batch_size_test=1000):
    dataPath = '../data/CIFAR_pre'
    targetTrain, targetTrainLabel  = load_data(dataPath + '/targetTrain.npz')
    targetTest,  targetTestLabel   = load_data(dataPath + '/targetTest.npz')
    shadowTrainRaw, shadowTrainLabel  = load_data(dataPath + '/shadowTrain.npz')
    shadowTestRaw,  shadowTestLabel   = load_data(dataPath + '/shadowTest.npz')
    targetTrainData = TensorDataset(torch.tensor(targetTrain),torch.tensor(targetTrainLabel).type(torch.LongTensor))
    targetTestData = TensorDataset(torch.tensor(targetTest),torch.tensor(targetTestLabel).type(torch.LongTensor))
    shadowTrainData = TensorDataset(torch.tensor(shadowTrainRaw),torch.tensor(shadowTrainLabel).type(torch.LongTensor))
    shadowTestData = TensorDataset(torch.tensor(shadowTestRaw),torch.tensor(shadowTestLabel).type(torch.LongTensor))
    targetTrainLoader = DataLoader(dataset = targetTrainData, batch_size = 64, shuffle = True)
    shadowTrainLoader = DataLoader(dataset = shadowTrainData, batch_size = 64, shuffle = True)
    targetTestLoader = DataLoader(dataset = targetTestData, batch_size = 64, shuffle = False)
    shadowTestLoader = DataLoader(dataset = shadowTestData, batch_size = 64, shuffle = False)
    return targetTrainLoader, targetTestLoader, shadowTrainLoader, shadowTestLoader

def load_data(data_name):
	with np.load( data_name) as f:
		train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
	return train_x, train_y

def clipDataTopX(dataToClip, top=3):
	res = [ sorted(s, reverse=True)[0:top] for s in dataToClip ]
	return np.array(res)

def generateAttackData(shadow_data=[],shadow_label=[],target_data=[],target_label=[], is_read = True, topX=3 ):
    if is_read == True:
        attackerModelDataPath = '../data/attackerModelData'
        targetX, targetY = load_data(attackerModelDataPath + '/targetModelData.npz')
        shadowX, shadowY = load_data(attackerModelDataPath + '/shadowModelData.npz')
    else:
        targetX = target_data
        targetY = target_label
        shadowX = shadow_data
        shadowY = shadow_label
	
    targetX = clipDataTopX(targetX,top=topX)
    print(targetX)
    shadowX = clipDataTopX(shadowX,top=topX)
    targetY= targetY.astype('float32')
    shadowY= shadowY.astype('float32')
    train_dataset = TensorDataset(torch.tensor(targetX),torch.tensor(targetY).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(shadowX), torch.tensor(shadowY).unsqueeze(1))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
    )
    test_loader=DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
    )
    return train_loader, test_loader
