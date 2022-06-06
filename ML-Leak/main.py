from model import ConvNet, BadNet, MlleaksMLP, init_weights
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import datasets
from train import train,  get_attack_data, eval_model
from data import dataloader, generateAttackData, getData
import os
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar', help='dataset choice,"cifar" or "mnist"')
parser.add_argument('--batch_size', default=64, type=int, help='batch size for training.')
parser.add_argument('--epoch', default=50, type=int, help='Number of epochs for shadow and target model.')
parser.add_argument('--attack_epoch', default=50, type=int, help='Number of epochs for attack model.')
parser.add_argument('--only_eval', default=False, type=bool, help='If true, only evaluate trained loaded models.')
parser.add_argument('--only_eval_attacker', default=False, type=bool, help='If true, only evaluate trained attacker model.')
parser.add_argument('--save_new_models', default=False, type=bool, help='If true, trained models will be saved.')
parser.add_argument('--save_new_models_attacker', default=False, type=bool, help='If true, trained attacker models will be saved.')
args = parser.parse_args()

def main():
    dataset = args.dataset
    shadow_path, target_path, attack_path = "./models/shadow_" + str(dataset) + ".pth", \
                                            "./models/target_" + str(dataset) + ".pth", \
                                            "./models/attack_" + str(dataset) + ".pth"

    # Cifar has rgb images(3 channels) and mnist is grayscale(1 channel)
    if dataset == "cifar":
        input_size = 3
    elif dataset == "mnist":
        input_size = 1

    n_epochs = args.epoch
    attack_epochs = args.attack_epoch
    batch_size = args.batch_size
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Note: getData involves preprocessed data for best performance, code not included. Refer to original ML-Leak for processing dataset
    shadow_train_loader, shadow_out_loader, target_train_loader, target_out_loader = getData(dataset=dataset, batch_size_train=batch_size, batch_size_test=1000)
    #shadow_train_loader, shadow_out_loader, target_train_loader, target_out_loader = dataloader(dataset=dataset, batch_size_train=batch_size, batch_size_test=1000)

    testloader = dataloader(dataset=dataset, batch_size_train=batch_size, batch_size_test=1000,
                            split_dataset=0)

    # badNet
    shadow_net = ConvNet(input_size=input_size).to(device)
    target_net = ConvNet(input_size=input_size).to(device)

    # Simple initialization of model weights
    shadow_net.apply(init_weights)
    target_net.apply(init_weights)

    target_loss = nn.CrossEntropyLoss()
    shadow_loss = nn.CrossEntropyLoss()
    target_optim = optim.Adam(target_net.parameters(), lr=0.001)
    #target_optim = optim.SGD(target_net.parameters(), lr=0.001)
    shadow_optim = optim.Adam(shadow_net.parameters(), lr=0.001)
    #shadow_optim = optim.SGD(shadow_net.parameters(), 0.001)

    # attack net is a binary classifier to determine membership
    attack_net = MlleaksMLP().to(device)
    # Binary cross entropy
    attack_loss = nn.BCELoss()
    attack_optim = optim.Adam(attack_net.parameters(), lr=0.01)
    #attack_optim = optim.SGD(attack_net.parameters(), lr=0.1)

    if os.path.exists(shadow_path):
        print("Load shadow model")
        shadow_net.load_state_dict(torch.load(shadow_path))
    # Training of shadow model on shadow training set
    if not args.only_eval:
        print("start training shadow model: ")
        train(shadow_net, n_epochs, shadow_train_loader, shadow_out_loader, shadow_loss, shadow_optim, verbose=False)

        if args.save_new_models:
            if not os.path.exists("./models"):
                os.mkdir("./models")  # Create the folder models if it doesn't exist
            # Save model after each epoch if argument is true
            torch.save(shadow_net.state_dict(), "./models/shadow_" + str(dataset) + ".pth")

    if os.path.exists(target_path):
        print("Load target model")
        target_net.load_state_dict(torch.load(target_path))
    # Train of target model on the target training set
    if not args.only_eval:
        print("start training target model: ")
        train(target_net, n_epochs, target_train_loader,target_out_loader, target_loss, target_optim, verbose=False)

        if args.save_new_models:
            # Save model after each epoch
            if not os.path.exists("./models"):
                os.mkdir("./models")  # Create the folder models if it doesn't exist
            torch.save(target_net.state_dict(), target_path)

    if os.path.exists(attack_path):
        print("Load attack model")
        attack_net.load_state_dict(torch.load(attack_path))
    # Training of attack model based on shadow net posteriors on shadow train and out datasets.
    if not args.only_eval_attacker:
        print("start training attacker model")
        shadow_data, shadow_label = get_attack_data(shadow_net, shadow_train_loader, shadow_out_loader)
        target_data, target_label = get_attack_data(target_net, target_train_loader, target_out_loader)
        attack_train_loader, attack_test_loader = generateAttackData(shadow_data, shadow_label, target_data, target_label, False, 3)
        train(attack_net, attack_epochs, attack_train_loader, attack_test_loader, attack_loss, attack_optim, attack = True, verbose=False)

        if args.save_new_models_attacker:
            if not os.path.exists("./models"):
                os.mkdir("./models")  # Create the folder models if it doesn't exist
            # Save model after each epoch
            torch.save(attack_net.state_dict(), attack_path)

    # Only evaluated pretrained loaded models when only_eval argument is True
    if args.only_eval:
        print("Classification Report Shadow Net:")
        eval_model(shadow_net, testloader, report=True)
        print("Classification Report Target Net:")
        eval_model(target_net, testloader, report=True)
        print("Report of Attack Net")
        eval_model(attack_net, attack_test_loader, report=True, attacker=True)

if __name__ == '__main__':
    main()
