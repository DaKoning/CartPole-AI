import atexit
import time
import gym
import Qlearning
import numpy as np

Q_table = Qlearning.Q_table
print(Q_table)
episodes = 10000000
timesteps = 1000

env = gym.make('CartPole-v0')

def save(Q_table):
    np.save("Qtable", Q_table)
    print("Main: Q-table saved!")

ts = []

for _ in range(episodes):
    observation = env.reset()
    observation = observation.round(decimals=1)
    for t in range(timesteps):
        env.render()

        best_row_index = Qlearning.get_best_row(observation, Q_table)
        if t != 0:
            Q_table = Qlearning.determine_Q(Q_table, observation_old, action_old, reward, best_row_index)
        
        atexit.unregister(save)
        atexit.register(save, Q_table)
        
        action = Qlearning.determine_action(Q_table, best_row_index)
        observation_old = observation
        observation, reward, done, info = env.step(action)
        observation = observation.round(decimals=1)
        action_old = action
        
        if done:
            reward = 0.0
            Q_table = Qlearning.determine_Q(Q_table, observation_old, action_old, reward, best_row_index)
        if done:
            #print(f"Episode done after {t} timesteps")
            break
    ts.append(t)
    average = sum(ts) / len(ts)
    if len(ts) > 100:
        del ts[0]
    print(f"Average of last 100: {round(average, 2)}    Timesteps: {t}")

env.close()