import numpy as np
import matplotlib.pyplot as plt
from hsy2001_HW1_problem1 import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    # print("belief_states: \n", belief_states)
    # print("belief_states", belief_states)
    # print("belief_states", belief_states.shape)
    print("cmap", cmap)
    
    print(observations)
    # # print(observations)
    # print(actions)
    # print(actions.shape)
    
    #### Test your code here
    belief = np.ones((20, 20)) / (20*20)
    for i in range(30):
        action = actions[i]
        obs = observations[i]
        belief_state = belief_states[i, :]
        # print(belief_states)
        filter = HistogramFilter()

        belief = filter.histogram_filter(cmap, belief, action, obs)
        # print(np.max(belief), belief[belief_state[1], belief_state[0]])