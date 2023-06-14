# RL_test
## **This is a RL-learning record**
### now dqn & ddqn 230601
### now actor & critic 230606

## **what is difference between off-line and on-line**
### why dqn is off-line?
dqn 是 记录一局游戏每一步的state、action、reward、next_state，游戏结束后抽样进行训练
### why reinforce is on-line?
reinforce 是 记录一局游戏每一步的state、action、reward、next_state，
游戏结束后再次在每一步的选择上都对网络进行训练，使每一步最好的action概率最大。对每一个状态的选择都要训练

## **what is value-based and policy-based**
### value-based
基于值函数的 基础模型为dqn  Q(s,a)=r+γV(s')
### policy-based
策略：是指概率，即一个状态s下选择每个action到达下一个状态的概率（选择动作的概率）学习的是每一个状态下选择动作的概率
基础模型为reinforce

