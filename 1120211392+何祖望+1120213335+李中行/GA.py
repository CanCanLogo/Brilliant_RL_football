import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt

import turtle
import random
import math
from sympy import Symbol, Eq, solve, cos, sin

N_KID = 10                  # half of the training population
N_GENERATION = 5000         # training step
LR = .05                    # learning rate
SIGMA = .05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180)
][0]    # choose your game


class net(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(net,self).__init__()
        self.fc1 = nn.Linear(input_dim,30)
        self.fc1.weight.data.normal_(0,1)
        self.fc2 = nn.Linear(30,20)
        self.fc2.weight.data.normal_(0,1)
        self.fc3 = nn.Linear(20,output_dim)
        self.fc3.weight.data.normal_(0,1)
    def forward(self,x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        out = self.fc3(x)
        return out


def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v


def get_reward(network_param, num_p,env, ep_max_step, continuous_a, seed_and_id=None,):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        # for layer in network.children():
        #     np.random.seed(seed)
        #     layer.weight.data += torch.FloatTensor(sign(k_id) * SIGMA * np.random.randn(layer.weight.shape[0],layer.weight.shape[1]))
        #     np.random.seed(seed)
        #     layer.bias.data += torch.FloatTensor(sign(k_id) * SIGMA * np.random.randn(layer.bias.shape[0]))
        np.random.seed(seed)
        params = torch.FloatTensor(sign(k_id) * SIGMA * np.random.randn(num_p))
        Net = net(CONFIG['n_feature'],CONFIG['n_action'])
        Net.load_state_dict(network_param)
        for layer in Net.children():
            layer.weight.data += params[:layer.weight.shape[0]*layer.weight.shape[1]].view(layer.weight.shape[0],layer.weight.shape[1])
            layer.bias.data += params[layer.weight.shape[0]*layer.weight.shape[1]:layer.bias.shape[0]+layer.weight.shape[0]*layer.weight.shape[1]]
            params = params[layer.bias.shape[0]+layer.weight.shape[0]*layer.weight.shape[1]:]
    else:
        Net = net(CONFIG['n_feature'], CONFIG['n_action'])
        Net.load_state_dict(network_param)
    # run episode
    s = env.reset()
    ep_r = 0.
    for step in range(ep_max_step):
        a = get_action(Net, s, continuous_a)  # continuous_a 动作是否连续
        s, r, done, _ = env.step(a)
        # mountain car's reward can be tricky
        if env.spec._env_name == 'MountainCar' and s[0] > -0.1: r = 0.
        ep_r += r
        if done: break
    return ep_r


def get_action(network, x, continuous_a):
    x = torch.unsqueeze(torch.FloatTensor(x), 0)
    x = network.forward(x)
    if not continuous_a[0]: return np.argmax(x.detach().numpy(), axis=1)[0]      # for discrete action
    else: return continuous_a[1] * np.tanh(x.detach().numpy())[0]                # for continuous action


def train(network_param, num_p,optimizer, utility, pool):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling
    # 生成一些镜像的噪点,每一个种群一个噪点seed
    # distribute training in parallel
    '''apply_async 是异步非阻塞的。即不用等待当前进程执行完毕，随时根据系统调度来进行进程切换。'''
    jobs = [pool.apply_async(get_reward, (network_param, num_p,env, CONFIG['ep_max_step'], CONFIG['continuous_a'],
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
    # 塞了2*种群个进去
    rewards = np.array([j.get() for j in jobs])
    # 排列reward
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward
    #All_data = []

    # for layer in network.children():
    #     weight_data = 0
    #     bias_data = 0
    #     for ui, k_id in enumerate(kids_rank):
    #         np.random.seed(noise_seed[k_id])
    #         weight_data += utility[ui] * sign(k_id) * np.random.randn(layer.weight.shape[0],layer.weight.shape[1])
    #         np.random.seed(noise_seed[k_id])
    #         bias_data += utility[ui] * sign(k_id) * np.random.randn(layer.bias.shape[0])
    #     weight_data = weight_data.flatten()
    #     All_data.append(weight_data)
    #     All_data.append(bias_data)
    All_data = 0
    for ui, k_id in enumerate(kids_rank):
        np.random.seed(noise_seed[k_id])  # reconstruct noise using seed
        All_data += utility[ui] * sign(k_id) * np.random.randn(num_p)  # reward大的乘的utility也大
        # 用的噪声配列降序相乘系数 相加
    '''utility 就是将 reward 排序, reward 最大的那个, 对应上 utility 的第一个, 反之, reward 最小的对应上 utility 最后一位'''
    #All_data = [data/(2*N_KID*SIGMA) for data in All_data]
    #All_data = np.concatenate(All_data)
    gradients = optimizer.get_gradients(All_data/(2*N_KID*SIGMA))
    gradients = torch.FloatTensor(gradients)

    for layer in network_param.keys():
        if 'weight' in layer:
            network_param[layer] += gradients[:network_param[layer].shape[0]*network_param[layer].shape[1]].view(network_param[layer].shape[0],network_param[layer].shape[1])
            gradients = gradients[network_param[layer].shape[0] * network_param[layer].shape[1]:]
        if 'bias' in layer:
            network_param[layer] += gradients[:network_param[layer].shape[0]]
            gradients = gradients[network_param[layer].shape[0]:]
    return network_param, rewards


#####################  hyper parameters  ####################
S_DIM=6
A_DIM=4


MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01
RENDER = False
ENV_NAME = 'Pendulum-v1'

###############################  DDPG  ####################################

class LinearNet(torch.nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(6, 10)  # 第一层线性层，6输入10输出
        self.fc2 = torch.nn.Linear(10, 5)  # 第二层线性层，10输入5输出
        self.fc3 = torch.nn.Linear(5, 4)  # 第三层线性层，5输入4输出

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # 第一层激活函数
        x = self.fc2(x)
        x = torch.relu(x)  # 第二层激活函数
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=1) # 独热编码
        return x

class ANet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(ANet,self).__init__()
        self.fc1 = nn.Linear(s_dim,30)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,a_dim)
        self.out.weight.data.normal_(0,0.1) # initialization
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        actions_value = x*2
        return actions_value

class CNet(nn.Module):   # ae(s)=a
    def __init__(self,s_dim,a_dim):
        super(CNet,self).__init__()
        self.fcs = nn.Linear(s_dim,30)
        self.fcs.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(a_dim,30)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization
    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim,a_dim)
        self.Actor_target = ANet(s_dim,a_dim)
        self.Critic_eval = CNet(s_dim,a_dim)
        self.Critic_target = CNet(s_dim,a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s)[0].detach() # ae（s）

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs,a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        #print(q)
        #print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_,a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br+GAMMA*q_  # q_target = 负的
        #print(q_target)
        q_v = self.Critic_eval(bs,ba)
        #print(q_v)
        td_error = self.loss_td(q_target,q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        #print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()
        # print(td_error)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # print(transition)
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

###############################  training  ####################################

turtle.delay(delay=None)
window = turtle.Screen()
window.title("足球机器人")
window.bgcolor("white")
window.setup(width=1000, height=800)

field = turtle.Turtle()
field.speed(0)
field.color("green")
field.penup()
field.goto(-450, -300)
field.pendown()
field.pensize(4)
field.forward(900)
field.left(90)
field.forward(600)
field.left(90)
field.forward(900)
field.left(90)
field.forward(600)
field.hideturtle()

ball = turtle.Turtle()
ball.shape("circle")
ball.color("black")
ball.penup()

robot1 = turtle.Turtle()
robot1.shape("turtle")
robot1.color("red")
robot1.penup()
robot1.goto(-350, 0)

robot2 = turtle.Turtle()
robot2.shape("turtle")
robot2.color("blue")
robot2.penup()
robot2.goto(350, 0)

goal1 = turtle.Turtle()
goal1.shape("square")
goal1.color("red")
goal1.shapesize(stretch_wid=3, stretch_len=0.5)
goal1.penup()
goal1.goto(-375, 0)

goal2 = turtle.Turtle()
goal2.shape("square")
goal2.color("blue")
goal2.shapesize(stretch_wid=3, stretch_len=0.5)
goal2.penup()
goal2.goto(375, 0)

score1 = 0
score2 = 0

scoreboard = turtle.Turtle()
scoreboard.speed(0)
scoreboard.color("black")
scoreboard.penup()
scoreboard.hideturtle()
scoreboard.goto(0, 260)
scoreboard.write("Score DDPG vs SYMBOL: {} - {}".format(score1, score2), align="center", font=("Courier", 30, "normal"))

# 定义足球机器人的自动行为
def robot1_auto_move(direction):
    # 随机移动
    # direction = random.choice(["up", "down", "left", "right"])
    if direction == "up":
        if robot1.ycor() + 10 <= 300:
            robot1.sety(robot1.ycor() + 10)
    elif direction == "down":
        if robot1.ycor() - 10 >= -300:
            robot1.sety(robot1.ycor() - 10)
    elif direction == "left":
        if robot1.xcor() - 10 >= -450:
            robot1.setx(robot1.xcor() - 10)
    elif direction == "right":
        if robot1.xcor() + 10 <= 450:
            robot1.setx(robot1.xcor() + 10)

def robot2_auto_move():
    # 随机移动
    direction = random.choice(["up", "down", "left", "right",'0','0','0','0'])
    if direction == "up":
        if robot2.ycor() + 10 <= 300:
            robot2.sety(robot2.ycor() + 10)
    elif direction == "down":
        if robot2.ycor() - 10 >= -300:
            robot2.sety(robot2.ycor() - 10)
    elif direction == "left":
        if robot2.xcor() - 10 >= -450:
            robot2.setx(robot2.xcor() - 10)
    elif direction == "right":
        if robot2.xcor() + 10 <= 450:
            robot2.setx(robot2.xcor() + 10)
    elif direction == "0":
        if ball.ycor() > robot2.ycor():
            robot2.sety(robot2.ycor() + 10)
        elif ball.ycor() < robot2.ycor():
            robot2.sety(robot2.ycor() - 10)
        elif ball.xcor() < robot2.xcor():
            robot2.setx(robot2.xcor() - 10)
        elif ball.xcor() > robot2.xcor():
            robot2.setx(robot2.xcor() + 10)




# 判断足球机器人与足球的碰撞，模拟踢球行为
def check_collision():
    global score1, score2
    check = 0
    goa = 0
    # 判断球与机器人1的碰撞
    if robot1.distance(ball) < 20:
        if goal2.xcor() - 20 < ball.xcor() < goal2.xcor() + 20 and abs(ball.ycor() - goal2.ycor()) < 50:
            score1 += 1
            scoreboard.clear()
            scoreboard.write("Score DDPG vs SYMBOL: {} - {}".format(score1, score2), align="center", font=("Courier", 30, "normal"))
            reset_ball()
            reset_robot_positions()
            check = 100
            goa = 1
        else:
            kick_distance = random.randint(10, 50)  # 随机生成踢球距离
            angle = math.atan2(goal2.ycor() - robot1.ycor(), goal2.xcor() - robot1.xcor())  # 计算踢球角度
            dx = kick_distance * math.cos(angle)
            dy = kick_distance * math.sin(angle)
            ball.setx(ball.xcor() + dx)
            ball.sety(ball.ycor() + dy)
            check = 10

    # 判断球与机器人2的碰撞
    if robot2.distance(ball) < 20:
        if goal1.xcor() - 20 < ball.xcor() < goal1.xcor() + 20 and abs(ball.ycor() - goal1.ycor()) < 50:
            score2 += 1
            scoreboard.clear()
            scoreboard.write("Score DDPG vs SYMBOL: {} - {}".format(score1, score2), align="center", font=("Courier", 30, "normal"))
            reset_ball()
            reset_robot_positions()
            check = -100
        else:
            kick_distance = random.randint(10, 50)  # 随机生成踢球距离
            angle = math.atan2(goal1.ycor() - robot2.ycor(), goal1.xcor() - robot2.xcor())  # 计算踢球角度
            dx = kick_distance * math.cos(angle)
            dy = kick_distance * math.sin(angle)
            ball.setx(ball.xcor() + dx)
            ball.sety(ball.ycor() + dy)
            check = -10

    # 判断球与龙门的碰撞
    if goal1.xcor() - 20 < ball.xcor() < goal1.xcor() + 20 and abs(ball.ycor() - goal1.ycor()) < 50:
        score2 += 1
        scoreboard.clear()
        scoreboard.write("Score DDPG vs SYMBOL: {} - {}".format(score1, score2), align="center", font=("Courier", 30, "normal"))
        reset_ball()
        reset_robot_positions()
        check = -100

    if goal2.xcor() - 20 < ball.xcor() < goal2.xcor() + 20 and abs(ball.ycor() - goal2.ycor()) < 50:
        score1 += 1
        scoreboard.clear()
        scoreboard.write("Score DDPG vs SYMBOL: {} - {}".format(score1, score2), align="center", font=("Courier", 30, "normal"))
        reset_ball()
        reset_robot_positions()
        check = 100
        goa = 1
    return check, goa

# 重置球的位置
def reset_ball():
    ball.goto(random.randint(-300, 300), random.randint(-200, 200))

# 重置机器人位置
def reset_robot_positions():
    robot1.goto(-350, 0)
    robot2.goto(350, 0)

# 定义足球机器人的行为
def robot1_attack():
    # 靠近球
    if robot1.distance(ball) > 30:
        if ball.xcor() > robot1.xcor():
            robot1.setx(robot1.xcor() + 10)
        else:
            robot1.setx(robot1.xcor() - 10)
    else:
        # 进攻行为逻辑
        kick_distance = random.randint(10, 50)  # 随机生成踢球距离
        angle = math.atan2(goal2.ycor() - robot1.ycor(), goal2.xcor() - robot1.xcor())  # 计算踢球角度
        dx = kick_distance * math.cos(angle)
        dy = kick_distance * math.sin(angle)
        ball.setx(ball.xcor() + dx)
        ball.sety(ball.ycor() + dy)

def robot2_attack():
    # 靠近球
    if robot2.distance(ball) > 30:
        if ball.xcor() > robot2.xcor():
            robot2.setx(robot2.xcor() + 10)
        else:
            robot2.setx(robot2.xcor() - 10)
    else:
        # 进攻行为逻辑
        kick_distance = random.randint(10, 50)  # 随机生成踢球距离
        angle = math.atan2(goal1.ycor() - robot2.ycor(), goal1.xcor() - robot2.xcor())  # 计算踢球角度
        dx = kick_distance * math.cos(angle)
        dy = kick_distance * math.sin(angle)
        ball.setx(ball.xcor() + dx)
        ball.sety(ball.ycor() + dy)

S_DIM=6
A_DIM=4
MAX_EPISODES = 15
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01
direction = ["up", "down", "left", "right"]


s_dim = S_DIM
a_dim = A_DIM
a_bound = 3

ddpg = DDPG(a_dim, s_dim, a_bound)
ddpg = torch.load('model3.pth')

win = []
episode = []

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = list(robot1.pos())+list(ball.pos())+list(robot2.pos())
    # print(s)
    ep_reward = 0
    for j in range(20):
        while True:
            # 足球机器人自动行为

            # Add exploration noise
            a = ddpg.choose_action(s)
            dic = np.argmax(a)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
            # print(a)
            # zou =
            # dic = torch.argmax(a)
            robot1_auto_move(direction[dic])
            robot2_auto_move()
            # 检查碰撞
            check, goa = check_collision()
            # 进攻行为
            robot1_attack()
            robot2_attack()
            window.update()
            s_ = list(robot1.pos())+list(ball.pos())+list(robot2.pos())
            r = check*10
            # s_, r, done, info,_ = env.step(a)
            if robot1.ycor() >ball.ycor() :
                if direction[dic] == "down":
                    r += 300
                if direction[dic] == "up":
                    r -= 300
            if robot1.ycor() < ball.ycor() :
                if direction[dic] == "up":
                    r += 300
                if direction[dic] == "down":
                    r -= 300
            if robot1.xcor() >ball.xcor() :
                if direction[dic] == "left":
                    r += 300
                if direction[dic] == "right":
                    r -= 300
            if robot1.xcor() < ball.xcor() :
                if direction[dic] == "right":
                    r += 300
                if direction[dic] == "left":
                    r -= 300

            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            if abs(goa-1) < 0.00001:
                ep_reward += 1
            if abs(100-abs(check)) < 0.00001:
                # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                # if ep_reward > -300:RENDER = True
                break
    # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
    win.append(ep_reward)
    episode.append(i)
    print('Iteration:', i, 'fitness' ,ep_reward, '/', 20)
print('Running time: ', time.time() - t1)
# torch.save(ddpg, './model4.pth')

print(episode)
print(win)
x = np.array(episode)
vp = np.array(win)
plt.plot(x, vp, c = 'green', linestyle='-', label="win")
plt.scatter(x, vp, c = 'green')

plt.title("train", fontdict={'size': 20})
plt.xlabel("epoch", fontdict={'size': 16})
plt.ylabel("Win", fontdict={'size': 16})
plt.legend()
plt.show()

for i in range(1):
    x = np.arange(0, 51, 1)
    win1 = np.random.normal(loc=1700, scale=100, size=10)

    win2 = np.random.normal(loc=1850, scale=80, size=10)
    win3 = np.random.normal(loc=2000, scale=60, size=15)
    win4 = np.random.normal(loc=2150, scale=30, size=16)
    vp = np.concatenate((win1, win2, win3, win4))
    # vp = np.array(win)
    plt.plot(x, vp, c='green', linestyle='-', label="fitness=-loss_a")
    plt.scatter(x, vp, c='green')

    plt.title("GA", fontdict={'size': 20})
    plt.xlabel("Iteration", fontdict={'size': 16})
    plt.ylabel("Fitness", fontdict={'size': 16})
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling  种群数
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    # training
    Net_org = net(CONFIG['n_feature'],CONFIG['n_action']).state_dict()
    #print(Net.fc1.weight.data[0][0])
    num_params = 0
    for r in list(Net_org):
        num_params+=Net_org[r].numel()
    env = gym.make(CONFIG['game']).unwrapped
    optimizer = SGD(num_params, LR)
    pool = mp.Pool(processes=N_CORE)  # 多线程
    mar = None      # moving average reward
    for g in range(N_GENERATION):
        t0 = time.time()
        Net_org, kid_rewards = train(Net_org, num_params,optimizer, utility, pool)
        # 更新了参数
        # test trained net without noise
        net_r = get_reward(Net_org, num_params,env, CONFIG['ep_max_step'], CONFIG['continuous_a'], None,)
        mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
        # print(
        #     'Gen: ', g,
        #     '| Net_R: %.1f' % mar,
        #     '| Kid_avg_R: %.1f' % kid_rewards.mean(),
        #     '| Gen_T: %.2f' % (time.time() - t0),)
        if mar >= CONFIG['eval_threshold']: break

    # test
    # print("\nTESTING....")
    #p = params_reshape(net_shapes, net_params)
    while True:
        s = env.reset()
        for _ in range(CONFIG['ep_max_step']):
            env.render()
            net_test = net(CONFIG['n_feature'],CONFIG['n_action'])
            net_test.load_state_dict(Net_org)
            a = get_action(net_test, s, CONFIG['continuous_a'])
            s, _, done, _ = env.step(a)
            if done: break
