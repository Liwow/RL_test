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
### why actor is on-line?
actor 是 与reinforce类似，但不同的是游戏结束后没有每一个选择都一个个去选择，而是使用价值函数对其进行指导来训练网络
