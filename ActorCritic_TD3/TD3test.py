import gym
from TD3 import TD3
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def Evaluation(Eseed,lseed):
    seed = Eseed+10#試行毎seed変更（学習と同じにならないようにずらす）
    ######### パラメータ #########
    env_name = "Pendulum-v0"
    random_seed = 0
    save_interval = 10
    lr = 3 * pow(10,-4)
    gamma = 0.99                #減衰率
    batch_size = 256
    max_timesteps = 200
    max_episodes = 500         #最大エピソード数500
    num_step = max_timesteps * max_episodes
    batch_size = 256
    save_step = save_interval * max_timesteps
    directory = "./preTrained/TD3/{}".format(env_name) # save trained models
    filename = "TD3_{}_{}".format(env_name, random_seed)
    #############################

    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action,min_action, batch_size, gamma)
    
    temp = np.zeros(1)
    sumR_ave = 0.0

    for st in range(save_step,num_step+1,save_step):#1つけんと最後のデータ読み込まん
        policy.load_models(directory, filename, st)
        sumR = 0.0
        for e in range(10):#10エピソードの平均
            state = env.reset()
            action = policy.select_action(state)

            for s in range(200):#not done
                #env.render()
                action = policy.select_action(state)
                next_state, reward, done, _ = env.step(action)
                sumR += reward
                state = next_state

        sumR_ave = sumR / 10
        if st == save_step:
            temp = sumR_ave
        else:
            temp = np.vstack((temp,sumR_ave))
    env.close()
    return temp  

def percentileplot(ndarray):
    num_step = 100000
    save_step = 2000
    num_plot = int(num_step/save_step)
    fig, ax = plt.subplots()
    y1 = np.zeros(num_plot)
    y2 = np.zeros(num_plot)
    y3 = np.zeros(num_plot)

    t = np.linspace(0, num_step, num_plot)
    for i in range(num_plot):
        y1[i] = np.percentile(ndarray[i], q=[25])
        y2[i] = np.percentile(ndarray[i], q=[50])
        y3[i] = np.percentile(ndarray[i], q=[75])

    c2 = "blue"      # 各プロットの色
    l2 = "TD3"   # 各ラベル
    ax.set_xlabel('step')  # x軸ラベル
    ax.set_ylabel('accumulated reward')  # y軸ラベル
    ax.grid()            # 罫線
    plt.xlim([0,num_step])######ここだけだと軸は増えるけど点がずれないt直す
    plt.xticks(np.arange(0, num_step + 1, 20000))

    plt.ylim([-2000,0])  
    ax.plot(t, y2, color=c2, label=l2, linewidth = 1.0)
    plt.fill_between(t,y1,y3,facecolor='DeepSkyBlue',alpha=0.6)
    ax.legend(loc=0)    # 凡例
    fig.tight_layout()  # レイアウトの設定
    pname = "TD3.st"+str(num_step)+".png"
    plt.savefig(pname) # 画像の保存

                
def main():
    num_episode = 500
    Evalue = np.zeros((1,num_episode))
    Fvalue = np.zeros((1,num_episode))
    seed = 0

    for i in tqdm(range(5)):#これを全体のループに変更
        #q_table = agent.load_models(seed,step_count)
        Evalue = Evaluation(i,seed)#評価時のseed,読み出すための学習時のseed
        
        if i == 0:
            Fvalue = Evalue.copy()
        else:
            Fvalue = np.hstack((Fvalue,Evalue))#Fvalue:num_episode*5

    #折れ線グラフ
    percentileplot(Fvalue)

if __name__ == '__main__':
    main()