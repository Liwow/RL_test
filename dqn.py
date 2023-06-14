import gym
import collections
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
import rl_utils

# 抛弃版本

learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()  # 继承nn.Module
        self.fc1 = nn.Linear(4, 128)  # 4维state
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # 当前state的最大q-{a为0时的q，a为1的q}

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer, max_q_value_list):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)  # 状态动作价值，当前s执行一个action后预估的q

        # max_a_prime = torch.argmax(q(s_prime), dim=1).reshape(-1, 1)  # ddqn
        # max_q_prime = q_target(s_prime).gather(1, max_a_prime)  # ddqn
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # dqn计算 下一状态的q最大

        target = r + gamma * max_q_prime * done_mask  # 这里会有高估的问题，用自己的估算来当作目标值
        max_q_value_list.extend(target.tolist())
        loss = F.smooth_l1_loss(q_a, target)

        # q_a与target。q_a是q网络预估执行已知a后的q（a确定）,
        # target是目标q网络预估下一状态的最大可能q（a未确定，greedy选择使q最大的a）
        # ddqn使用 q网络的下一状态最大q对应的a，当作预估网络下一状态的a，作为预估值（即仅max_q_prime改变）

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def plot(max_q_value_list, return_list, env_name):
    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format(env_name))
    plt.show()


def main():
    env = gym.make('CartPole-v0')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    onesum = 0.0
    rewards = []
    max_q_value_list = []
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False

        while not done:  # 一局game
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime
            onesum += r
            if done:
                rewards.append(onesum)
                score += onesum
                onesum = 0
                break

        if memory.size() > 2000:
             train(q, q_target, memory, optimizer, max_q_value_list)

        if n_epi % print_interval == 0 and n_epi != 0:  # 20局输出一下
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0
    # plot(max_q_value_list, rewards, 'CartPole-v1')
    env.close()


if __name__ == '__main__':
    main()
