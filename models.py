import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
import torchnet.meter as meter


class Baseline(nn.Module):
    """
    Baseline network
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)

        return x


def train(net, optimizer, loss_function, data_loader, val_loader, epoch, device):
    net.to(device)
    save_epoch = epoch // 10 if epoch > 20 else 1
    train_loss = []
    val_accuracies = []
    for e in tqdm(range(1, epoch + 1)):
        net.train()
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = net(data)
            loss = loss_function(out, label.long())

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        val_acc = val(net, val_loader, ignored_labels=[0], device=device)
        val_accuracies.append(val_acc)
        metric = abs(val_acc)

        if e % save_epoch == 0:
            save_model(net, 'BaseLine_run' + str(e) + '_' + str(metric), 'checkpionts/')
    return train_loss, val_accuracies


def val(net, data_loader, ignored_labels, device):
    accuracy, total = 0., 0.
    net.eval()
    for batch_idx, (data, target) in enumerate(data_loader):
        # Load the data into the GPU if required
        data, target = data.to(device), target.to(device)
        output = net(data)

        _, output = torch.max(output, dim=1)
        # print(output)

        for out, pred in zip(output.view(-1), target.view(-1)):
            if out.item() in ignored_labels:
                continue
            else:
                accuracy += out.item() == pred.item()
                total += 1

    net.train()
    return accuracy / total


def test(net, data_loader, device):
    net.to(device)
    net.eval()
    pred_labels = []
    for batch_idx, (data, _) in enumerate(data_loader):
        with torch.no_grad():
            data = data.to(device)
            out = net(data)
            _, output = torch.max(out, dim=1)
            # print(output)
            pred_labels.append(output)
    return pred_labels


def save_model(model, model_name, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), save_dir + model_name + '.pth')
    else:
        print('Model is error')
