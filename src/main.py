# imports
import math
import time
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import game

# Replay Memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# The model itself
class DQN(nn.Module):

    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 4, 3, padding=1)
        self.linear1 = nn.LazyLinear(128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

device = torch.device("cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.20
EPS_END = 0.20
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = game.Environment("levels/the_tower.txt")

# left still right left-jump still-jump right-jump
# 2    0     1     6         4          5
n_actions = 6

state = env.get_state()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)

model_filepath = input("Filepath to Saved Model(within saved_models directory): ")
model_filepath = "saved_models/" + model_filepath
if not model_filepath == "saved_models/":
    policy_net.load_state_dict(torch.load(model_filepath))

backup_filepath = input("Filepath for Backups every 10 generations: ")
backup_filepath = "saved_models/" + backup_filepath

target_net.load_state_dict(policy_net.state_dict())
policy_net.forward(state)
target_net.forward(state)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
is_random_action = False

def select_action(state, is_testing):
    global steps_done
    global is_random_action
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START-EPS_END) * \
        math.exp(-1*steps_done/EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or is_testing:
        with torch.no_grad():
            is_random_action = False
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        is_random_action = True
        return torch.randint(0, 6, (1, 1), device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

num_episodes = (int)(input("How many episodes: "))
is_training = input("Training Loop?(y/n): ") == "y"

for i_episode in range(num_episodes):
    env.reset(is_training)
    state = env.get_state()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
    for t in count():
        action = select_action(state, not is_training)
        executed_action = action.item()
        if(executed_action == 3):
            executed_action = 6
        reward, terminated = env.do_game_tick(executed_action)

        
        match(executed_action):
            case 0:
                action_string = "None"
            case 1:
                action_string = "Walk Right"
            case 2:
                action_string = "Walk Left"
            case 4:
                action_string = "Jump"
            case 5:
                action_string = "Jump Right"
            case 6:
                action_string = "Jump Left"
        
        print(env.get_display())
        print("Action: ", action_string)
        print("Reward: ", reward)
        print("Random Action: ", is_random_action)

        observation = env.get_state()
        reward = torch.tensor([reward], device=device)
        
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(1)
 
        memory.push(state, action, next_state, reward)
        state = next_state
        if is_training:
            optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if terminated:
            break

    if (i_episode % 10) == 0 and is_training:
        torch.save(policy_net.state_dict(), backup_filepath)

print('Complete')
if is_training:
    model_save_pathway = input("Name the file where you are saving the model: ")
    torch.save(policy_net.state_dict(), "saved_models/" + model_save_pathway)