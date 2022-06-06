import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import argparse
from Attacker import Backdoor_Atk
from model import BadNet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist') # mnist or cifar10
parser.add_argument('--trigger_label', type=int, default=0, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--poi_portion', type=float, default=0.1)

def main():
    arg = parser.parse_args()
    trigger_label = arg.trigger_label
    poi_portion = arg.poi_portion
    dataset = arg.dataset
    batch_size = 64

    if dataset == 'mnist':
        train_data = datasets.MNIST(root = '../data/', train = True, 
                                    transform = transforms.ToTensor(), download = True)
        test_data = datasets.MNIST(root = '../data/', train = False, 
                                    transform = transforms.ToTensor(), download = True)
    elif dataset == 'cifar10':
        train_data = datasets.CIFAR10(root = '../data/', train = True, 
                                    transform = transforms.ToTensor(), download = True)
        test_data = datasets.CIFAR10(root = '../data/', train = False, 
                                    transform = transforms.ToTensor(), download = True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_data_adv = Backdoor_Atk(train_data, trigger_label, portion=poi_portion, mode="train", device=device, dataname=dataset)
    test_data_ori = Backdoor_Atk(test_data,  trigger_label, portion=0,    mode="test",  device=device, dataname=dataset)
    test_data_tri = Backdoor_Atk(test_data,  trigger_label, portion=1,    mode="test",  device=device, dataname=dataset)

    train_data_loader       = DataLoader(dataset=train_data_adv,    batch_size=batch_size, shuffle=True)
    test_data_ori_loader    = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True)
    test_data_tri_loader    = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True) 

    model = BadNet(train_data_loader.dataset.channels , train_data_loader.dataset.class_num).to(device)
    #cost = torch.nn.CrossEntropyLoss()
    cost = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.01)

    epochs = 25
    print('------ Train start ---------')
    for epoch in range(epochs) :
        # train
        sum_loss = 0.0
        train_acc = 0
        for data in train_data_loader:
            inputs, lables = data
            inputs, lables = Variable(inputs).cuda(), Variable(lables).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cost(outputs, lables)
            loss.backward()
            optimizer.step()
            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data

        acc_train = eval(model, train_data_loader, batch_size=batch_size, mode='backdoor')
        acc_test_ori = eval(model, test_data_ori_loader, batch_size=batch_size, mode='backdoor')
        acc_test_tri = eval(model, test_data_tri_loader, batch_size=batch_size, mode='backdoor')

        print("Epoch: %d of %d, Loss:%.03f,  Train acc: %.4f, Test_ori acc: %.4f, Test_tri acc: %.4f\n"\
            % ( epoch + 1, epochs, sum_loss / len(train_data_loader), acc_train, acc_test_ori, acc_test_tri))

        torch.save(model.state_dict(), f'/model'+ dataset + '.pth')


def eval(model, data_loader, batch_size=64, mode='backdoor'):
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        
        batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    return accuracy_score(y_true.cpu(), y_predict.cpu())

def print_model_perform(model, data_loader):
    model.eval() # switch to eval mode
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        
        batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    try:
        target_names_idx = set.union(set(np.array(y_true.cpu())), set(np.array(y_predict.cpu())))
        target_names = [data_loader.dataset.classes[i] for i in target_names_idx]
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=target_names))
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()