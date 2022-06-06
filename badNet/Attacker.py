import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class Backdoor_Atk(Dataset):

    def __init__(self, dataset, trigger_label, portion=0.1, mode="train", device=torch.device("cuda"), dataname="mnist"):
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.data, self.targets = self.add_trigger(dataset.data, dataset.targets, trigger_label, portion, dataname)
        self.channels, self.width, self.height = self.__shape_info__()

        if portion == 1 :
            for i in range(9):
                plt.subplot(3,3,i+1)
                plt.imshow(self.data[i].reshape(self.width,self.height,self.channels)/255)
                plt.axis('off')
            plt.show()

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1
        label = torch.Tensor(label) 

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def add_trigger(self, data, targets, trigger_label, portion, dataname):

        if dataname == "mnist":
            #print(data.shape)
            data_res = np.array(data.reshape(len(data),1,28,28)) # reshape the data to be fed into the network
        elif dataname == "cifar10":
            data_res = np.array(data.reshape(len(data),3,32,32))

        new_data = copy.deepcopy(data_res)
        new_targets = np.array(copy.deepcopy(targets))
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        _, width, height = new_data.shape[1:]

        new_targets[perm] = trigger_label

        new_data[perm, :, width-3, height-3] = 255
        new_data[perm, :, width-4, height-2] = 255
        new_data[perm, :, width-2, height-4] = 255
        new_data[perm, :, width-2, height-2] = 255

        print("Adversarial images generated: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(perm), len(new_data)-len(perm), portion))
        return torch.Tensor(new_data), new_targets