# imports
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Game Internals
class Player():
    WIDTH = 1000
    HEIGHT = 1000
    MOVE_ACCEL = 3
    JUMP_ACCEL = 350
    GRAV_ACCEL = 15
    FRIC_ACCEL = 0.98

    def __init__(self):
        self.posX = 0
        self.posY = 0
        self.velX = 0
        self.velY = 0
        self.state = 0

    def camCollisionCheck(self, x, y):
        playLeft = self.posX
        playRight = self.posX + Player.WIDTH
        playDown = self.posY
        playUp = self.posY + Player.HEIGHT
        return x >= playLeft and x <= playRight and y >= playDown and y <= playUp
    
    def handleMovement(self, input):
        rPressed = (input & 1) == 1
        lPressed = (input>>1 & 1) == 1
        uPressed = (input>>2 & 1) == 1

        if(rPressed and self.state != 2):
            self.velX += Player.MOVE_ACCEL
        if(lPressed and self.state != 2):
            self.velX -= Player.MOVE_ACCEL
        if(uPressed and self.state == 1):
            self.velY += Player.JUMP_ACCEL
            self.state = 0

        if(self.state == 0):
            self.velY -= Player.GRAV_ACCEL
        elif(self.state == 1):
            self.velX *= Player.FRIC_ACCEL

        self.velX = min(self.velX, 200)
        self.velX = max(self.velX, -200)
        self.velY = min(self.velY, 500)
        self.velY = max(self.velY, -500)

        self.posX += self.velX
        self.posY += self.velY

class GameObject():

    def __init__(self, x, y, width, height, isVisible=False, isTangible=False, isDeathPlane=False):
        self.posX = x
        self.posY = y
        self.width = width
        self.height = height
        self.isVisible = isVisible
        self.isTangible = isTangible
        self.isDeathPlane = isDeathPlane

    def camCollisionCheck(self, x, y):
        objLeft = self.posX
        objRight = self.posX + self.width
        objDown = self.posY
        objUp = self.posY + self.height
        return x >= objLeft and x <= objRight and y >= objDown and y <= objUp
    
    def collisionCheck(self, playerX, playerY):
        if not self.isTangible:
            return False
        playLeft = playerX
        playRight = playerX+Player.WIDTH
        playDown = playerY
        playUp = playerY + Player.HEIGHT
        objLeft = self.posX
        objRight = self.posX + self.width
        objDown = self.posY
        objUp = self.posY + self.height
        withinXBounds = (playLeft >= objLeft and playLeft <= objRight) or (playRight <= objRight and playRight >= objLeft)
        withinYBounds = (playDown >= objDown and playDown <= objUp) or (playUp <= objUp and playUp >= objDown)
        return withinXBounds and withinYBounds

    def getDisplacement(self, playerX, playerY):
        playLeft = playerX
        playRight = playerX+Player.WIDTH
        playDown = playerY
        playUp = playerY + Player.HEIGHT
        objLeft = self.posX
        objRight = self.posX + self.width
        objDown = self.posY
        objUp = self.posY + self.height

        yUp = objUp-playDown+1
        xLeft = playRight-objLeft+1
        yDown = playUp-objDown+1
        xRight = objRight-playLeft+1
        if(xLeft < xRight and xLeft < yDown and xLeft < yUp):
             return (-1*xLeft, 0)
        elif(xRight < yDown and xRight < yUp):
            return (xRight, 0)
        elif(yDown < yUp):
            return (0, -1*yDown)
        else:
            return (0, yUp)

class GameHandler():
    
    def __init__(self, x=0, y=0):
        self.player = Player()
        self.player.posX = x
        self.player.posY = y
        self.gameObjList = []
        self.gameObjList.append(GameObject(-500000, -10000, 1000000, 10000, True, True))

    def loadObjsFromText(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                vals = line.split(",")
                isV = int(vals[4]) & 1 == 1
                isT = (int(vals[4]) >> 1) & 1
                isDP = (int(vals[4]) >> 2) & 1
                obj = GameObject(int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), isV, isT, isDP)
                self.gameObjList.append(obj)

    def getCamCollision(self, x, y):
        for obj in self.gameObjList:
            if not obj.isVisible:
                continue
            if obj.camCollisionCheck(x, y):
                if obj.isDeathPlane:
                    return 2
                return 1
        if self.player.camCollisionCheck(x, y):
            return 3
        return 0
    
    def doTick(self, input):
        if self.player.state == 2:
            return False
        self.player.state = 0
        bFlag = False
        for i in range(0, 1001, 200):
            for obj in self.gameObjList:
                if not obj.isVisible:
                    continue
                if obj.camCollisionCheck(self.player.posX+i, self.player.posY-1):
                    self.player.state = 1
                    bFlag = True
                    break
            if(bFlag):
                break
        
        self.player.handleMovement(input)
        for obj in self.gameObjList:
            if obj.collisionCheck(self.player.posX, self.player.posY):
                if obj.isDeathPlane:
                    self.player.state = 2
                    return False
                displacement = obj.getDisplacement(self.player.posX, self.player.posY)
                if(displacement[1] == 0):
                    self.player.posX += displacement[0]
                    self.player.velX = 0
                elif(displacement[0] == 0):
                    self.player.posY += displacement[1]
                    self.player.velY = 0
                    if(displacement[1] > 0):
                        self.player.state = 1
        return True

# Graphics Handling

class DisplayHandler():

    def __init__(self, game, xRes=60, yRes=30, xRange=14000, yRange=10000):
        self.game = game
        self.xResolution = xRes
        self.yResolution = yRes
        self.xRange = xRange
        self.yRange = yRange

    def getPixelBuffer(self):
        xPosition = self.game.player.posX+500
        yPosition = self.game.player.posY+3000
        pBuffer = [[0 for i in range(0, self.yResolution)] for i in range(0, self.xResolution)]
        for i in range(0, self.xResolution):
            for j in range(0, self.yResolution):
                camX = i*self.xRange/self.xResolution+xPosition-self.xRange/2
                camY = j*self.yRange/self.yResolution+yPosition-self.yRange/2
                pBuffer[i][j] = self.game.getCamCollision(camX, camY)
        return pBuffer

    def getStringDisplay(self):
        pBuffer = self.getPixelBuffer()
        s = "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"
        for j in range(self.yResolution-1, -1, -1):
            for i in range(0, self.xResolution):
                match pBuffer[i][j]:
                    case 0:
                        s += " "
                    case 1:
                        s += "o"
                    case 2:
                        s += "/"
                    case 3:
                        s += "z"
            s += "\n"
        return s

# Environment
class Environment():

    def __init__(self, filepath):
        self.filepath = filepath
        self.reset()

    def reset(self):
        self.game = GameHandler(0, 0)
        self.game.loadObjsFromText(self.filepath)
        self.display = DisplayHandler(self.game)

    def getState(self):
        return self.display.getPixelBuffer()

    def doTick(self, input):
        p1 = self.game.player.posX
        if not self.game.doTick(input):
            return -50
        return self.game.player.posX-p1
    
    def getDisplay(self):
        return self.display.getStringDisplay()

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

    def __init__(self, width_observations, height_observations, n_actions):
        super(DQN, self).__init__()
        self.main = torch.nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(8, 4, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),

            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions))

    def forward(self, x):
        ret = self.main(x)
        return ret

plt.ion()
device = torch.device("cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = Environment("levels/test.txt")

# left still right left-jump still-jump right-jump
# 2    0     1     6         4          5
n_actions = 6

state = env.getState()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
width_observations = len(state)
height_observations = len(state[0])

policy_net = DQN(width_observations, height_observations, n_actions).to(device)
target_net = DQN(width_observations, height_observations, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
policy_net.forward(state)
target_net.forward(state)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
episode_durations = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START-EPS_END) * \
        math.exp(-1*steps_done/EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            print(policy_net(state).max(1).indices)
            return torch.randint(0, 6, (1,), device=device, dtype=torch.long)
            #.view(1)
    else:
        return torch.randint(0, 6, (1,), device=device, dtype=torch.long)

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title=('Result')
    else:
        plt.clf()
        plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)

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

    criterion = nn.smoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

num_episodes = 10

for i_episode in range(num_episodes):
    env.reset()
    state = env.getState()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        if(action[0] == 3):
            action[0] = 6
        reward = env.doTick(action)
        print(env.getDisplay())
        terminated = reward == -50
        observation = env.getState()
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if terminated:
            episode_durations.append(t+1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plot.ioff()
plot.show()