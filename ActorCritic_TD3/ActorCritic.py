import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
import numpy as np
from tqdm import tqdm
import sys


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,max_action, min_action):
        super(Actor, self).__init__()
        self.high = max_action
        self.low = min_action
      
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, action_dim)
      
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = (self.high + self.low) / 2 + ((self.high - self.low) / 2) * torch.tanh(self.l2(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)#1で横に連結する
        
        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

class ReplayBuffer:
    def __init__(self, max_size=10**5):
        self.buffer = []
        self.max_size = int(max_size)
    
    def add(self, transition):# transiton:(state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):        
        index = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in index:
            s, a, r, ns, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(ns, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

#モデルの保存torch.save(the_model.state_dict(), PATH)
#load_state_dict(torch.load(PATH))モデルの読み込み

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

class ActorCritic(Agent):
    """
    ActorCriticに以下を追加
    1:Target Actor & Target Critic
    2:Target Policy Smoothing Regularization
    3:Delayed Policy Update
    4:Clipped Double Q-Learning
    """
    def __init__(self, lr, state_dim, action_dim, max_action,min_action, batch_size, gamma):
        #1:Target Actor & Target Critic
        self.actor = Actor(state_dim, action_dim, max_action, min_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.max_action = max_action
        self.min_action = min_action
        self.batch_size = batch_size
        self.gamma = gamma

    def save_models(self, directory, name, step):#name = "AC_{}_{}".format(env_name, seed)
        torch.save(self.actor.state_dict(), '%s/%s_%sstep_actor.pth' % (directory, name, step))
        torch.save(self.critic.state_dict(), '%s/%s_%sstep_critic.pth' % (directory, name, step))
        
    def load_models(self, directory, name, step):
        self.actor.load_state_dict(torch.load('%s/%s_%sstep_actor.pth' % (directory, name, step), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load('%s/%s_%sstep_critic.pth' % (directory, name, step), map_location=lambda storage, loc: storage))
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)#行数を１にする。列数は自動的に
        return self.actor(state).cpu().data.numpy().flatten()#actorで現在の状態から行動を選択
        """
        .data で Variable 内の Tensor にアクセス
        .cpu() で GPU からデータを転送
        .numpy() で Tensor から NumPy に変換
        .flatten で 多次元配列を1次元に変換
        """

    def select_exploratory_action(self, state, Dnoise):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return_action = self.actor(state).cpu().data.numpy().flatten() + Dnoise
        return return_action.clip(self.min_action, self.max_action)#範囲に収める

    def train(self, state, action_re, next_state, reward, done, step_count):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action_re).to(device)
        reward = torch.FloatTensor(reward).reshape((self.batch_size,1)).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape((self.batch_size,1)).to(device)
        
        next_action = self.actor(next_state)

        Q = self.critic(next_state, next_action)
        delta = reward + (self.gamma * Q).detach()
        
        #criticの更新
        #順伝搬
        current_Q = self.critic(state, action)
        #ロスの計算
        loss_Q = F.mse_loss(current_Q, delta)
        #勾配の初期化
        self.critic_optimizer.zero_grad()
        #勾配の計算
        loss_Q.backward()
        #パラメータの更新
        self.critic_optimizer.step()
        

        #actorの損失関数（反転）(1を使う)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        #actor更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

def main():
    ######### パラメータ #########
    args = sys.argv#コマンドラインで指定
    if(len(args) == 2):
        seed = int(args[1])
    else:
        seed = 0
    print("seed is " + str(seed))
    env_name = "Pendulum-v0"
    save_interval = 10
    gamma = 0.99                #減衰率
    batch_size = 256
    lr = 3 * pow(10,-4)         #学習率
    max_episodes = 500         #最大エピソード数500
    max_timesteps = 200        #１エピソードの最大ステップ数(pendulum:200)
    directory = "./preTrained/actorcritic/{}".format(env_name)
    filename = "ActorCritic_{}_{}".format(env_name, seed)
    texpl = 10000               #ランダム行動ステップ数
    noise = 0.1                 #行動方策のノイズ
    #############################
    
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]#値として使いたいから0付ける
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    policy = ActorCritic(lr, state_dim, action_dim, max_action,min_action, batch_size, gamma)
    replay_buffer = ReplayBuffer()

    D2 = ((max_action-min_action) * noise/2)**2#行動方策のnoise
    Dnoise = np.random.normal(0, D2, size=env.action_space.shape[0])#pendulumなら１
    
    #log用
    avg_reward = 0
    ep_reward = 0
    log_f = open("logActorCritic.txt","w+")
    
    #学習
    step_count = 0#ステップ数カウント用
    for episode in tqdm(range(1, max_episodes+1)):
        state = env.reset()
        for t in range(max_timesteps):
            step_count += 1
            if step_count < texpl:
                # ランダム行動
                action = env.action_space.sample()
            else:
                action = policy.select_exploratory_action(state,Dnoise)
            
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            
            avg_reward += reward#学習確認用
            ep_reward += reward#log用

            #256たまったら開始
            if step_count >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                policy.train(states, actions, next_states, rewards, dones, step_count)

        # logの更新
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()
        ep_reward = 0
        
        if episode % save_interval == 0:
            policy.save_models(directory, filename,step_count)

            #経過表示
            avg_reward = int(avg_reward / save_interval)
            print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
            avg_reward = 0

if __name__ == '__main__':
    main()
