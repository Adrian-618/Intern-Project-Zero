from time import process_time_ns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

# Contains functions for model training and evaluation and for training and evaluating the attacker model

def train(model, data_loader, criterion, optimizer, verbose=True):
    """
    Function for model training step
    """
    running_loss = 0
    model.train()
    for step,  (batch_img, batch_label)in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()  # Set gradients to zero
        output = model(batch_img.cuda())  # Forward pass
        loss = criterion(output, batch_label.cuda())
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss
        # Print loss for each minibatch
        if verbose:
            print("[%d/%d] loss = %f" % (step, len(data_loader), loss.item()))
    return running_loss


# Trains attack model, which can classify a data sample as a member or not of the training set, by using shadow
# model's posterior probabilities for sample class predictions, as its feature vector.
def get_attack_data(model, train_data, test_data):
    data = []
    label = []    
    model.eval()
    for step, (train_img, _) in enumerate(train_data):
        train_posteriors = F.softmax(model(train_img.cuda()), dim=1)
        data.append(train_posteriors.cpu().detach().numpy())
        label.append(np.ones(len(train_posteriors)))

    for step, (test_img, _) in enumerate(test_data):
        test_posteriors = F.softmax(model(test_img.cuda()), dim=1)
        # print(test_posteriors)
        data.append(test_posteriors.cpu().detach().numpy())
        label.append(np.zeros(len(test_posteriors)))
    data = np.vstack(data)
    label = np.concatenate(label)
    return data, label

def eval_model(model, test_loader, report=True, attacker = False):
    """
    Simple evaluation with the addition of a classification report with precision and recall
    """
    total = 0
    correct = 0
    gt = []
    preds = []
    with torch.no_grad():  # Disable gradient calculation
        model.eval()
        for step, (batch_img, batch_label) in enumerate(tqdm(test_loader)):

            output = model(batch_img.cuda())
            # print(output)

            if attacker == True:
                predicted = torch.where(output>0.5, 1, 0)
                # print(predicted)
            else:
                predicted = torch.argmax(output, dim=1)
            # print(predicted)
            preds.append(predicted)

            gt.append(batch_label)

            total += batch_img.size(0)
            correct += torch.sum(batch_label.cuda() == predicted)

    accuracy = 100 * (correct/total)
    # print(correct)
    # print(total)
    if report:
        gt = torch.cat(gt, 0)
        preds = torch.cat(preds, 0)
        print(classification_report(gt.cpu(), preds.cpu()))

    return accuracy