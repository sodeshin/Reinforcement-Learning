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
        path = "4.2.3s"+str(seed)+"/Q.s"+str(seed)+".t"+str(time)+".txt"
        np.savetxt(path, q_table)

    def load_models(self,seed,time):
        path = "4.2.3s"+str(seed)+"/Q.s"+str(seed)+".t"+str(time)+".txt"
        q_tablet = np.loadtxt(path)
        return q_tablet

    def __init__(self,action_space):
        self.action_space = action_space
        
    def select_action(self,q_table,observation):#評価時はこっち
        next_state = digistate(observation)
        next_action = np.argmax(q_table[next_state][:])
        return next_action

    def select_exploratory_action(self,q_table,observation,episode):
        next_state = digistate(observation)

        #課題3ここを変更
        #epsilon = 0.05
        epsilon = 0.5 * (0.99 ** episode)
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

    #Qtable作成
    def createQtable(self,k,l):
        q_table = np.random.normal(loc = 0,scale = 1,size = (pow(k,3), l)) * pow(10, -8)
        #状態3変数をk分割、行動1変数をl分割(qtable内マイナス値OK)
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
        return q_table#平均、標準偏差、出力配列サイズ

class ReplayBuffer:
    #経験再⽣バッファー.FIFO.保存する最⼤サイズ:buffer_size
    def add(self,state,action,next_state,reward,done):
        pass
    def sample(self,batch_size):
        pass

class RBufferQ(ReplayBuffer):
    def add(self,observation,action,next_observation,reward,done,step_count):
        step_buff[step_count][0],step_buff[step_count][1],step_buff[step_count][2] = observation
        step_buff[step_count][3] = action
        step_buff[step_count][4],step_buff[step_count][5],step_buff[step_count][6] = next_observation
        step_buff[step_count][7] = reward
        step_buff[step_count][8] = done

    def sample(self,batch_size,step_count):#0からstepcountいかでバッチサイズ個乱数を作って対応する要素を代入
        l_s = list(range(0, step_count))
   
        l_256 = random.sample(l_s, 256)#被復元抽出
        p = 0
        for i in l_256:
            observation_array[p][0] = step_buff[i][0]
            observation_array[p][1] = step_buff[i][1]
            observation_array[p][2] = step_buff[i][2]
            action_array[p] = step_buff[i][3]
            n_observation_array[p][0] = step_buff[i][4]
            n_observation_array[p][1] = step_buff[i][5]
            n_observation_array[p][2] = step_buff[i][6]
            reward_array[p]= step_buff[i][7]
            done_array[p] = step_buff[i][8]
            p += 1
        return observation_array,action_array,n_observation_array,reward_array,done_array

    # def sample(self,batch_size,step_count):#0からstepcountいかでバッチサイズ個乱数を作って対応する要素を代入
    #     l_256 = np.random.randint(0,step_count,batch_size)#被復元抽出
    #     return step_buff[l_256][:,0:3], step_buff[l_256][:,3],step_buff[l_256][:,4:7], step_buff[l_256][:,7],step_buff[l_256][:,8]
    
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
    #正確に10分割されてる
    #enumerate:インデックス番号,要素を列挙
    #c=3,s=0,t=5なら503(1000の内一意的に定まる)
    return sum([x * (k ** i) for i, x in enumerate(digitized)])

def digichange(action):#step用。actionを-2~2に落とし込む
    l = 9
    l_width = 4/(l-1)#-2~2
    action_one = float(-2.0 + l_width * action)#action:0の時－2、8の時2
    return action_one
    '''(参考)
Num	Action	Min	Max
0	Joint effort	-2.0	2.0
    '''
#関数終わり


seed = 5
env = gym.make('Pendulum-v0')
#疑似乱数
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
step_count = 0


num_step = 500000#総ステップ
num_episode = int(num_step / 200) #pendulumの場合
save_table_step = 400#テーブルの保存が200ステップ毎(要検討)


agent = QAgent(env.action_space)
q_table = agent.createQtable(10,9)#(1000,9)
buff = RBufferQ()

batch_size = 256
buffer_size = num_step

observation_array = np.zeros((batch_size,3))
action_array = np.zeros(batch_size)
n_observation_array = np.zeros((batch_size,3))
reward_array = np.zeros(batch_size)
done_array= np.zeros(batch_size)
step_buff = np.zeros((buffer_size,9))

for e in tqdm(range(num_episode)):
    observation = env.reset()
    state = digistate(observation)
    action = np.argmax(q_table[state])
    #sumR = 0.0
    
    for s in range(200):#not done
        #env.render()
        action = agent.select_exploratory_action(q_table,observation,e)
        action_one = digichange(action)
        next_observation , reward , done , info = env.step(np.array([action_one]))
        #まずbufferに256ためる
        #257以降はその中からランダムに256選んで格納(buffには全部記憶しておく)
        #そいつ全部(256)を使って学習

        buff.add(observation,action,next_observation,reward,done,step_count)
        #sumR += reward
        step_count += 1
        observation = next_observation

        if step_count >= batch_size:
            observation_array,action_array,n_observation_array,reward_array,done_array = buff.sample(buffer_size,step_count)
            for i in range(batch_size):
                observation_l = observation_array[i]
                action_l = int(action_array[i])
                next_observation_l = n_observation_array[i]
                reward_l = reward_array[i]
                #done_l = done_array[i]
                
                agent.train(q_table,action_l,observation_l,next_observation_l,reward_l)  
    if (e+1) % 2 == 0:
        agent.save_models(q_table, seed, step_count)

env.close()