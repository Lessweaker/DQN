import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import copy
# 环境参数
GRID_SIZE = 10
NUM_AGENTS = 3
ACTIONS = ['up', 'down', 'left', 'right']
MAX_STEPS = GRID_SIZE * 10  # 最大步数设为网格边长的10倍

# DQN参数
BATCH_SIZE = 64
BUFFER_CAPACITY = 100000
LR = 0.001
GAMMA = 0.99
TAU = 0.005
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.992


class MultiAgentSearchEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.num_agents = NUM_AGENTS
        self.max_steps = MAX_STEPS  # 新增最大步数参数
        self.goal = np.array([2,3])
        self.reset()
    
    def reset(self):
        self.explored = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        # self.agents_pos = np.array([[np.random.randint(self.grid_size),
                                #    np.random.randint(self.grid_size)] for _ in range(self.num_agents)])
        self.agents_pos = np.array([np.array([2,9]),np.array([7,7]),np.array([8,3])])
        self.current_step = 0  # 初始化步数计数器
        for pos in self.agents_pos:
            self.explored[tuple(pos)] = True
        return self._get_obs()
    
    def _get_obs(self):
        obs = []
        for i in range(self.num_agents):
            other_pos = []
            for j in range(self.num_agents):
                if j != i:
                    other_pos.extend(self.agents_pos[j])
            obs.append({
                'explored': self.explored.copy(),
                'position': self.agents_pos[i],
                'others': np.array(other_pos)
            })
        return obs
    
    def step(self, actions):
        self.current_step += 1  # 更新当前步数
        new_positions = np.copy(self.agents_pos)
        rewards = np.zeros(self.num_agents)
        done = False
        
        # 移动所有智能体
        for i in range(self.num_agents):
            action = actions[i]
            x, y = self.agents_pos[i]
            if action == 0:    # 上
                new_pos = (max(0, x-1), y)
            elif action == 1:  # 下
                new_pos = (min(self.grid_size-1, x+1), y)
            elif action == 2:  # 左
                new_pos = (x, max(0, y-1))
            elif action == 3:  # 右
                new_pos = (x, min(self.grid_size-1, y+1))
            new_positions[i] = new_pos
        
        # # 检查碰撞并更新位置
        # for i in range(self.num_agents):
        #     collision = False
        #     for j in range(i+1, self.num_agents):
        #         if np.array_equal(new_positions[i], new_positions[j]):
        #             collision = True
        #             break
        #     if collision:
        #         rewards[i] = -2
        #     else:
        #         self.agents_pos[i] = new_positions[i]
        #         if not self.explored[tuple(new_positions[i])]:
        #             rewards[i] =2
        #             self.explored[tuple(new_positions[i])] = True
        #         else:
        #             # rewards[i] -= 0.1
        #             rewards[i] -= 0.5
            
        # done = np.all(self.explored) or (self.current_step >= self.max_steps)
        # 检查碰撞并更新位置
        for i in range(self.num_agents):
            collision = False
            for j in range(i+1, self.num_agents):
                if np.array_equal(new_positions[i], new_positions[j]):
                    collision = True
                    break
            if collision:
                rewards[i] -= 10            
            else:
                self.agents_pos[i] = new_positions[i]
                if not self.explored[tuple(new_positions[i])]:
                    self.explored[tuple(new_positions[i])] = True
                    rewards[i] += 0.5
                # else:
                #     rewards[i] -= 0.2
                    # rewards[i] -= 0.5
            if self.explored[tuple(self.goal)]:
                rewards[i] = 20
                done = True
                break
            else:
                rewards[i] -= 0.3
                # print(i)
                # print(new_positions[i])
                # print(tuple(new_positions[i]))
                
        # print(self.explored[tuple(self.goal)])
        # print(done)
        done = self.explored[tuple(self.goal)] or (self.current_step >= self.max_steps)
        # done = np.array_equal(np.array(new_positions[0]), np.array([0,0]))  or (self.current_step >= self.max_steps)
        # 在step函数的奖励计算部分添加时间惩罚
        # if done and (self.current_step >= self.max_steps):
        #     # 未在限制步数内完成探索的惩罚
        #     rewards -= 0.5 * (self.max_steps - self.current_step)/self.max_steps
        return self._get_obs(), rewards, done, {}
    # 可视化函数
    def visualize_trajectory(self, explored, trajectories):
        # print(trajectories[0])
        # plt.figure(figsize=(8, 8))
        # plt.imshow(explored, cmap='Greys', alpha=0.3)
        colors = ['r', 'b', 'g']
        for i in range(NUM_AGENTS):
            trajectory = np.array(trajectories[i])
            # plt.plot(trajectory[:, 1], trajectory[:, 0], marker='o', color=colors[i], label=f'Agent {i+1}')
            plt.plot(trajectory[:, 1], trajectory[:, 0], color=colors[i], label=f'Agent {i+1}')
            # plt.scatter(trajectory[0, 1], trajectory[0, 0], color='green', marker='s', s=100, label='Start' if i == 0 else "")
            plt.scatter(trajectory[-1, 1], trajectory[-1, 0], color='black', marker='*', s=200, label='End' if i == 0 else "")
        plt.scatter(self.goal[1],self.goal[0],color='red', marker='s',label='goal')
        plt.legend()
        plt.title("Multi-Agent Search Trajectories")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(0,10)
        plt.ylim(0,10)
        plt.grid(False)
class DQN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 输入通道数为 input_channels
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输入通道数为 32
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * GRID_SIZE * GRID_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        x = self.conv(x)  # 输入形状为 (batch_size, input_channels, GRID_SIZE, GRID_SIZE)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_dim, action_dim).to(self.device)
        self.target_net = DQN(input_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_CAPACITY)
        self.action_dim = action_dim
        self.loss_history = []  # 新增loss记录列表
    
    def select_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = self._process_state(state)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def _process_state(self, state):
        explored = state['explored']  # 形状为 (GRID_SIZE, GRID_SIZE)
        explored_tensor = torch.FloatTensor(explored).unsqueeze(0).unsqueeze(0).to(self.device)  # 形状为 (1, 1, GRID_SIZE, GRID_SIZE)
        return explored_tensor
    
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0
        
        transitions = self.buffer.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
        
        states = [self._process_state(s).squeeze(0) for s in batch[0]]
        states = torch.stack(states)
        actions = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = [self._process_state(ns).squeeze(0) for ns in batch[3]]
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * GAMMA * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        loss_value = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 软更新目标网络
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(TAU*policy_param.data + (1-TAU)*target_param.data)

        return loss_value


# 滑动平均滑动窗函数
def moving_average(data, window_size):
    """计算滑动平均"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# 训练过程
env = MultiAgentSearchEnv()
input_channels = 1  # 输入数据的通道数为 1
output_dim = len(ACTIONS)  # 输出维度为动作空间的大小
agent = DQNAgent(input_channels, output_dim)

epsilon = EPSILON_START

episode_rewards = []
EPISODE = 500

best_coverage = 0.0
best_trajectory = None
best_explored = None
best_reward = 0
episode_losses = []  # 新增episode级loss记录

# 记录轨迹
trajectories = [[] for _ in range(NUM_AGENTS)]

for episode in range(EPISODE):
    state = env.reset()
    done = False
    total_loss = 0.0
    total_rewards = np.zeros(NUM_AGENTS)
    for i in range(NUM_AGENTS):
        trajectories[i] = [env.agents_pos[i].copy()]
    # 重置轨迹
    
    while not done:
        actions = []
        for i in range(NUM_AGENTS):
            action = agent.select_action(state[i], epsilon)
            actions.append(action)
        
        next_state, rewards, done, _ = env.step(actions)
        
        for i in range(NUM_AGENTS):
            agent.buffer.push(state[i], actions[i], rewards[i], next_state[i], done)
            trajectories[i].append(env.agents_pos[i].copy())
        state = next_state
        total_rewards += rewards
        loss_value = agent.update()
        total_loss += loss_value  # 累加每一步的损失值
    
    # 记录历史最佳覆盖率和路径
    coverage = np.sum(env.explored) / (GRID_SIZE**2)
    if sum(total_rewards) > best_reward:
        best_reward = sum(total_rewards)
        best_trajectory = copy.deepcopy(trajectories)  # 更新最佳路径
        best_explored = env.explored.copy()

    epsilon = max(EPSILON_END, epsilon*EPSILON_DECAY)
    
    episode_rewards.append(sum(total_rewards))
    episode_losses.append(total_loss / NUM_AGENTS)  # 平均每个智能体的损失值

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {sum(total_rewards):.1f}, Epsilon: {epsilon:.3f}")
        print(len(agent.buffer))

print("Training completed!")

# 可视化最终轨迹
env.visualize_trajectory(env.explored, trajectories)
plt.show()
# 可视化历史最佳轨迹
if best_trajectory:
    print(f"Best coverage: {best_coverage:.2f}")
    env.visualize_trajectory(best_explored, best_trajectory)
    plt.savefig('best_trajectory.svg')  # 保存图像
    plt.show()


window_size = 20  # 滑动平均窗口大小
smoothed_rewards = moving_average(episode_rewards, window_size)
smoothed_losses = moving_average(episode_losses, window_size)  # 对损失进行滑动平均

# 绘制原始奖励曲线、滑动平均奖励曲线和损失曲线
plt.figure(figsize=(14, 6))

# 绘制奖励曲线
plt.subplot(1, 2, 1)
plt.plot(np.linspace(1, EPISODE, EPISODE), episode_rewards, label="Episode Rewards", alpha=0.6)
plt.plot(np.linspace(window_size, EPISODE, EPISODE - window_size + 1), smoothed_rewards, label=f"Smoothed Rewards (Window={window_size})", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Rewards with Moving Average")
plt.legend()
plt.grid()

# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(np.linspace(1, EPISODE, EPISODE), episode_losses, label="Loss per Episode", alpha=0.6)
plt.plot(np.linspace(window_size, EPISODE, EPISODE - window_size + 1), smoothed_losses, label=f"Smoothed Losses (Window={window_size})", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Training Loss with Moving Average")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
