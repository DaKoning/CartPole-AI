import numpy as np
import random
from os.path import exists

delete_progress = False

if delete_progress or not exists("Qtable.npy"):
    print("Q-learning: Creating Q-table")
    Q_table = np.array([[0, 0, 0, 0, 0, 0]], dtype=object)
    np.save("Qtable", Q_table)
else:
    print("Q-learning: Loading Q-table")
    Q_table = np.load("Qtable.npy", allow_pickle=True)


alpha = 1.0
gamma = 0.80
epsilon = 0.1
Q_old = 0

def determine_Q(Q_table, observation_old, action_old, reward, best_row_index):
    if best_row_index:
        Q_max = Q_table[best_row_index, 5]
    else:
        Q_max = 0.0

    row = list(observation_old).append(action_old)
    same_row_index = np.argwhere(np.all(Q_table[:, :5] == row, axis=1) == True)
    if same_row_index.size != 0:
        same_row_index = same_row_index[0, 0]
        Q_old = Q_table[same_row_index, 5]
        Q_new = Q_old + alpha * (reward + gamma * Q_max - Q_old)
        Q_table[same_row_index, 5] = Q_new
    else:
        Q_old = 0.0
        Q_new = Q_old + alpha * (reward + gamma * Q_max - Q_old)
        Q_table = np.append(Q_table, [[observation_old[0], observation_old[1], observation_old[2], observation_old[3], action_old, Q_new]], axis=0)
    
    return Q_table

def get_best_row(observation, Q_table):
    same_state_indexes = np.argwhere(np.all(Q_table[:, :4] == observation, axis=1) == True)
    if same_state_indexes.size != 0:
        Q_values = Q_table[same_state_indexes, 5]
        best_row_index = same_state_indexes[np.argmax(Q_values)][0]
    else:
        best_row_index = None 
    return best_row_index

def determine_action(Q_table, best_row_index):
    r = random.randint(1, 10000) / 10000
    if r <= epsilon:
        # exploration
        action = random.randint(0, 1)
    else:
        # exploitation
        if best_row_index:
            action = int(Q_table[best_row_index, 4])

        else:
            action = random.randint(0, 1)
    return action