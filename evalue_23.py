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
        path = "Q.s"+str(seed)+".t"+str(time)+".txt"
        np.savetxt(path, q_table)

    def load_modelsT(self,seed,time):
        path = "4.2.3s"+str(seed)+"/Q.s"+str(seed)+".t"+str(time)+".txt"
        q_tablet = np.loadtxt(path)
        return q_tablet

    def load_modelsF(self,seed,time):
        path = "4.2.2s"+str(seed)+"/Q.s"+str(seed)+".t"+str(time)+".txt"
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

    #Qtable作成
    def createQtable(self,k,l):
        q_table = np.random.normal(loc = 0,scale = 1,size = (pow(k,3), l)) * pow(10, -8)
        #状態3変数をk分割、行動1変数をl分割(qtable内マイナス値OK)
        return q_table#平均、標準偏差、出力配列サイズ

def Evaluation(Eseed,lseed,TF):
    seed = Eseed+10
    envE = gym.make('Pendulum-v0')
    envE.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    agentE = QAgent(envE.action_space)
    q_tableE = agentE.createQtable(10,9)
    temp = np.zeros(1)
    sumR_ave = 0.0

    for st in range(save_table_step,num_step+1,save_table_step):
        if TF == True:
            q_tableE = agentE.load_modelsT(lseed,st)#4.2.3
        else:
            q_tableE = agentE.load_modelsF(lseed,st)#4.2.2
        sumR = 0.0
        for e in range(10):#10エピソードの平均
            observation = envE.reset()
            state = digistate(observation)
            action = np.argmax(q_tableE[state])
            for s in range(200):#not done
                action = agentE.select_action(q_tableE,observation)
                action_one = digichange(action)
                next_observation , reward , done , info = envE.step(np.array([action_one]))
                sumR += reward
                observation = next_observation
        sumR_ave = sumR / 10
        if st == save_table_step:
            temp = sumR_ave
        else:
            temp = np.vstack((temp,sumR_ave))
    envE.close()
    return temp

def percentileplot(ndarray,ndarrayN):
    num_plot = int(num_step/save_table_step)
    fig, ax = plt.subplots()
    y1 = np.zeros(num_plot)
    y2 = np.zeros(num_plot)
    y3 = np.zeros(num_plot)
    y1N = np.zeros(num_plot)
    y2N = np.zeros(num_plot)
    y3N = np.zeros(num_plot)

    t = np.linspace(0, num_step, num_plot)
    for i in range(num_plot):
        y1[i] = np.percentile(ndarray[i], q=[25])
        y2[i] = np.percentile(ndarray[i], q=[50])
        y3[i] = np.percentile(ndarray[i], q=[75])
        y1N[i] = np.percentile(ndarrayN[i], q=[25])
        y2N[i] = np.percentile(ndarrayN[i], q=[50])
        y3N[i] = np.percentile(ndarrayN[i], q=[75])

    c2 = "blue"      # 各プロットの色
    l2 = "change ε-greedy"   # 各ラベル
    c2N = "red"      # 各プロットの色
    l2N = "baseline"   # 各ラベル   
    ax.set_xlabel('step')  # x軸ラベル
    ax.set_ylabel('accumulated reward')  # y軸ラベル
    ax.grid()            # 罫線
    plt.xlim([0,num_step])######ここだけだと軸は増えるけど点がずれないt直す
    plt.xticks(np.arange(0, num_step + 1, 100000))

    plt.ylim([-2000,0])  
    ax.plot(t, y2, color=c2, label=l2, linewidth = 1.0)
    ax.plot(t, y2N, color=c2N, label=l2N, linewidth = 1.0)
    plt.fill_between(t,y1,y3,facecolor='DeepSkyBlue',alpha=0.6)
    plt.fill_between(t,y1N,y3N,facecolor='orange',alpha=0.6)
    ax.legend(loc=0)    # 凡例
    fig.tight_layout()  # レイアウトの設定
    pname = "QL23.se"+str(seed)+".st"+str(num_step)+".png"
    plt.savefig(pname) # 画像の保存

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
save_table_step = 400#400#ここを1000にする

Evalue = np.zeros((1,num_episode))
Fvalue = np.zeros((1,num_episode))
EvalueN = np.zeros((1,num_episode))
FvalueN = np.zeros((1,num_episode))

for i in tqdm(range(5)):
    Evalue = Evaluation(i,seed+i,True)#評価時のseed,読み出すための学習時のseed.s5~9をつかう
    EvalueN = Evaluation(i,seed+i,False)
    if i == 0:
        Fvalue = Evalue.copy()
        FvalueN = EvalueN.copy()
    else:
        Fvalue = np.hstack((Fvalue,Evalue))
        FvalueN = np.hstack((FvalueN,EvalueN))

#グラフ描画
percentileplot(Fvalue,FvalueN)
env.close()