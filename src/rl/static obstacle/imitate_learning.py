import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from net_Lin import *
from tqdm import tqdm


class My_dataset(Dataset):
    def __init__(self, path, scan_num):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # path = [scan_path, target_path, action_path]
        self.scan_num = scan_num
        self.scan = [path[0] + x for x in os.listdir(r"" + path[0])]
        self.target = [path[1] + x for x in os.listdir(r"" + path[1])]
        self.action_path = [path[2] + x for x in os.listdir(r"" + path[2])]

    def __getitem__(self, index):
        scan = np.load(self.scan[index])
        re_data = []
        for _ in range(self.scan_num):
            re_data.append(scan[_ * (len(scan) // self.scan_num)])
        scan = self.data_to_tensor(re_data)

        target = np.load(self.target[index])
        target = self.np_to_tensor(target).unsqueeze(0)

        state = torch.cat((target, scan), 0)

        action = np.load(self.action_path[index])
        action[0][0], action[0][1] = action[0][1], action[0][0]
        action = self.np_to_tensor(action)
        return state, action.squeeze(0)

    def __len__(self):
        return len(self.action_path)

    def np_to_tensor(self, data):
        return torch.from_numpy(data).float().to(self.device)

    def data_to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float).to(self.device)


class imitate_learning():
    def __init__(self, path, scan_num):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.my = My_dataset(path, scan_num)
        self.train_loader = DataLoader(dataset=self.my, batch_size=8, shuffle=True)

        self.actor = Actor_net(scan_num + 1, 2).to(self.device)
        # self.actor.load_state_dict(torch.load(model_path))

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.epoch = 500

    def train(self):
        total_loss = 0.0
        for batch, (data, label) in enumerate(self.train_loader):
            action, log_prob = self.actor.sample(data)
            loss = self.loss(action, label.float().to(self.device))
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def save_(self):
        torch.save(self.actor.state_dict(), './Expert_data/model_100.pth')

    def step(self):
        for epoch in range(self.epoch):
            print('epoch', epoch)
            loss = self.train()
            print('avg loss:', loss)
        self.save_()


if __name__ == '__main__':
    scan_path = './Expert_data/scan/'
    target_path = './Expert_data/target/'
    action_path = './Expert_data/action/'
    scan_num = 100
    path = [scan_path, target_path, action_path]
    im = imitate_learning(path, scan_num)
    im.step()
