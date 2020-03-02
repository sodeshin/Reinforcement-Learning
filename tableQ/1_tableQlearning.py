from tqdm import tqdm
import random
import numpy as np
import torch
import gym
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


class Agent:
    def save_models(self,path):
        pass
    def load_models(self,path):
        pass
    def select_action(self,state):#ターゲット方策
        pass
    def select_exploratory_action(self,state):#行動方策
        pass
    def train(self,state,action,next_state,reward,done):
        pass

class QAgent(Agent):
    def save_models(self, q_table, seed, time):
        path = "4.2.1s"+str(seed)+"/Q.s"+str(seed)+".t"+str(time)+".txt"
        np.savetxt(path, q_table)

    def load_models(self,seed,time):
        path = "4.2.1s"+str(seed)+"/Q.s"+str(seed)+".t"+str(time)+".txt"
        q_tablet = np.loadtxt(path)
        return q_tablet

    def __init__(self,action_space):
        self.action_space = action_space
        
    def select_action(self,q_table,observation):#評価時はこっち
        next_state = digistate(observation)
        next_action = np.argmax(q_table[next_state][:])
        return next_action

    def select_exploratory_action(self,q_table,observation):
        next_state = digistate(observation)
        epsilon = 0.05
        if  epsilon <= np.random.uniform(0, 1):#1-ε
            next_action = np.argmax(q_table[next_state][:])
        else:#ε
            next_action = np.random.choice([0, 8])#k=9
        return next_action
     
    def train(self,q_table,action,observation,next_observation,reward):# Qテーブルの更新
        alpha = 3 * pow(10,-4)
        gamma = 0.99
        state = digistate(observation)
        next_state = digistate(next_observation)
        next_action = np.argmax(q_table[next_state]) 
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[next_state, next_action] - q_table[state, action])

    def createQtable(self,k,l):
        '''(参考)
        Observation
        Type: Box(3)
        Num	Observation	Min	Max
        0	cos(theta)	-1.0	1.0
        1	sin(theta)	-1.0	1.0
        2	theta dot	-8.0	8.0

        Actions
        Type: Box(1)
        Num	Action	Min	Max
        0	Joint
        '''
        q_table = np.random.normal(loc = 0,scale = 1,size = (pow(k,3), l)) * pow(10, -8)
        #状態3変数をk分割、行動1変数をl分割
        return q_table#平均、標準偏差、出力配列サイズ

def bins(bmin,bmax,num):#観測状態→離散値
    #num分割する場合仕切りはnum+1
    #np.digitizeは範囲外の分もカウント(つまり状態が12)
    #だから先頭から2番目から末尾の手前までをスライス
    return np.linspace(bmin, bmax, num + 1)[1:-1]

def digistate(observation):#Qテーブル用に離散値に
    cos,sin,theta = observation
    k = 10 #k=10分割
    digitized = [np.digitize(cos, bins=bins(-1.0, 1.0, k)),
                 np.digitize(sin, bins=bins(-1.0, 1.0, k)),
                 np.digitize(theta, bins=bins(-8.0, 8.0, k))]
    #enumerate:インデックス番号,要素を列挙
    #c=3,s=0,t=5なら503(1000の内一意的に定まる)
    return sum([x * (k ** i) for i, x in enumerate(digitized)])

def digichange(action):#step用。actionを-2~2に落とし込む
    '''(参考)
    Num	Action	Min	Max
    0	Joint effort	-2.0	2.0
    '''
    l = 9
    l_width = 4/(l-1)#-2~2
    action_one = float(-2.0 + l_width * action)#action:0の時－2、8の時2
    return action_one


#main
seed = 5
env = gym.make('Pendulum-v0')
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
step_count = 0

num_step = 500000#総ステップ
num_episode = int(num_step / 200) #pendulumの場合
save_table_step = 400#テーブルの保存が400ステップ毎

agent = QAgent(env.action_space)
q_table = agent.createQtable(10,9)#(1000,9)

for e in tqdm(range(num_episode)):
    observation = env.reset()
    state = digistate(observation)
    action = np.argmax(q_table[state])
    
    for s in range(200):#not done
        action = agent.select_exploratory_action(q_table,observation)
        action_one = digichange(action)
        next_observation , reward , done , info = env.step(np.array([action_one]))
        step_count += 1
        agent.train(q_table,action,observation,next_observation,reward)
        observation = next_observation

    if (e+1) % 2 == 0:
        agent.save_models(q_table, seed, step_count)

env.close()