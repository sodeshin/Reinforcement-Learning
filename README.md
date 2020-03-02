# Reinforcement-Learning
強化学習の実験で作成したtableQLearning,ActorCritic,TD3のプログラムです.環境はOpen AIが提供しているgymのPendulum-v0を使用.
ActorCritic,TD3にはPytorchを利用.

# 結果
## tableQLearning & tableQwithRB:500000step
![QL se5 st500000](https://user-images.githubusercontent.com/52310645/75676655-0d55cf80-5ccd-11ea-9232-85b3a5381922.png)
## tableQwithRB & tableQchangeEpsilon:500000step
![QL23 se5 st500000](https://user-images.githubusercontent.com/52310645/75676667-134bb080-5ccd-11ea-8e23-d652f6f04ef8.png)
## ActorCritic:100000step
![ActorCritic st100000mix](https://user-images.githubusercontent.com/52310645/75676090-ddf29300-5ccb-11ea-9298-9413a98723cf.png)
## TD3:100000step
![TD3 st100000mix](https://user-images.githubusercontent.com/52310645/75676106-eb0f8200-5ccb-11ea-83fc-c6caec810d84.png)
## (参考)TD3の4つの工夫をそれぞれ取り除いたアブレーションテスト結果:100000step
![TD3_ALL st100000mix](https://user-images.githubusercontent.com/52310645/75676118-f2cf2680-5ccb-11ea-88d3-a35e7a30558e.png)
