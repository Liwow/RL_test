import time

import gym

"""
1. 环境（environment）
2. 智能体agent（算法）
agent发送action至environment，environment返回观察和回报。
"""


def main():
    """
    用 make() 创建一个 gym 中的现成环境
    """
    env = gym.make("CartPole-v1")
    obs, reward, done, info, _ = env.reset()
    print("obs: {}".format(obs))
    print("reward: {}".format(reward))
    print("done: {}".format(done))
    print("info: {}".format(info))
    print("action_space: {}".format(env.action_space))
    print("observation_space: {}".format(env.observation_space))
    print("observation_space.high: {}".format(env.observation_space.high))
    print("observation_space.low: {}".format(env.observation_space.low))
    # 刷新当前环境，并显示
    for _ in range(1000):
        env.render()
        obs, reward, done, info, _ = env.step(env.action_space.sample())
        if done:
            break
        time.sleep(0.1)
    env.close()


if __name__ == "__main__":
    main()
