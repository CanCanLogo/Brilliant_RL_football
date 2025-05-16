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
    print('Epoch:', i, 'Win' ,ep_reward, '/', 20)
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
