#!/home/a/anaconda3/envs/torch/bin/python3
import torch
from collections import deque
from random import sample
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=5)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=9, out_channels=18, kernel_size=5)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=18, out_channels=18, kernel_size=5)

        self.fc1 = nn.Linear(5832, 648)
        self.fc2 = nn.Linear(648, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 3)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool_1(out)
        #
        out = F.relu(self.conv2(out))
        out = self.pool_2(out)

        out = F.relu(self.conv3(out))

        out = torch.flatten(out, 1)
        #
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)

        return out


class critic_Linear(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(critic_Linear, self).__init__()
        self.h1 = nn.Linear(state_dim + action_dim, 512)
        self.h2 = nn.Linear(512, 256)
        self.h3 = nn.Linear(256, 64)
        self.h4 = nn.Linear(64, action_dim)

        self.apply(weights_init_)

    def forward(self, x, y):
        # x_y_z = torch.cat((x, y, z), 1)
        x_y_z = torch.cat((x, y), 1)
        out = F.relu(self.h1(x_y_z))
        out = F.relu(self.h2(out))
        out = F.relu(self.h3(out))
        out = self.h4(out)
        return out


class Actor_Linear(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_Linear, self).__init__()
        self.h1 = nn.Linear(state_dim, 512)
        self.h2 = nn.Linear(512, 256)
        self.h3 = nn.Linear(256, 64)

        self.mean = nn.Linear(64, action_dim)
        self.std = nn.Linear(64, action_dim)

        self.apply(weights_init_)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, epsilon=1e-6):
        # action, _ = a.actor(state)
        out = F.relu(self.h1(x))
        # if x.shape[0] == 1:
        #     print(x)
        #     print(out)
        out = F.relu(self.h2(out))
        out = F.relu(self.h3(out))

        mean = self.mean(out)
        log_std = self.std(out)

        std = F.softplus(log_std)

        dist = Normal(mean, std)

        normal_sample = dist.rsample()  # 在标准化正态分布上采样
        log_prob_ = dist.log_prob(normal_sample)  # 计算该值的标准正太分布上的概率

        # action = torch.tanh(normal_sample)  # 对数值进行tanh

        action = torch.zeros(normal_sample.shape[0], 2).to(self.device)
        action[:, 0] = torch.tanh(mean[:, 0])
        action[:, 1] = torch.sigmoid(mean[:, 1])
        # action[:, 1] = torch.tanh(mean[:, 1])

        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob_ - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)  # 为了提升目标对应的概率值

        action[:, 0] = action[:, 0] * 2.5
        action[:, 1] = action[:, 1] * 0.25

        return action, log_prob, mean, std


class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_net, self).__init__()
        # self.cnn = cnn()
        self.Linear = critic_Linear(state_dim=state_dim, action_dim=action_dim)

    def forward(self, state: tuple, action):
        # out_cnn = self.cnn(state[0])
        # out = self.Linear(state[1], out_cnn, action)
        out = self.Linear(state, action)
        return out


class Actor_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor_net, self).__init__()
        # self.cnn = cnn()
        self.Linear = Actor_Linear(state_dim=state_dim, action_dim=action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state: tuple):
        # out_cnn = self.cnn(state[0])
        # mean, log_std = self.Linear(state[1], out_cnn)
        action, log_std, mean, std = self.Linear(state)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return action, log_std, mean, std


class Replay_Buffers():
    def __init__(self, batch_size):
        self.buffer_size = 100000
        self.buffer = deque([], maxlen=self.buffer_size)
        self.batch = batch_size

    def write_Buffers(self, state, next_state, reward, action, done):
        once = {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'done': done, }
        self.buffer.append(once)
        if len(self.buffer) > self.batch:
            return sample(self.buffer, self.batch)
