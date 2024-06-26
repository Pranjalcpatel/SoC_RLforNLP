import numpy as np
import matplotlib.pyplot as plt
import random

def get_reward(action):
    if action == 0:
        return np.random.normal(2, 1)
    elif action == 1:
        return np.random.randint(2) * (-11) + 5
    elif action == 2:
        return np.random.poisson(2)
    elif action == 3:
        return np.random.exponential(3)
    else:
        n2 = np.random.randint(4)
        return get_reward(n2)

def epsilongreedy(episodes, timesteps, epsilon):
    q = np.zeros((timesteps, 5))  # action values
    n = np.zeros((timesteps, 5))  # number of times an action is taken
    netreward = []  # net reward post each episode
    
    for i in range(episodes):
        actionlist = []
        rewardlist = []
        
        for j in range(timesteps):
            if np.random.rand() <= epsilon:
                action = np.random.randint(5)
            else:
                max_q = np.max(q[j, :])
                max_q_states = [k for k in range(5) if q[j, k] == max_q]
                action = random.choice(max_q_states)
                
            reward = get_reward(action)
            actionlist.append(action)
            rewardlist.append(reward)
        
        # Updating the action values after episode
        for l in range(timesteps):
            action = actionlist[l]
            n[l, action] += 1
            stepsize = 1 / n[l, action]

            future_rewards = np.array(rewardlist[l:])
            discount = np.array([0.3 ** (timestep - l) for timestep in range(timesteps - l)])
            gt = np.sum(future_rewards * discount)

            error = gt - q[l, action]
            q[l, action] += stepsize * error
        
        netreward.append(sum(rewardlist))
    
    print(f"The matrix n is {n}")
    print(f"The action values are {q}")
    
    return netreward

# Running the epsilon-greedy algorithm
totalreward = epsilongreedy(1000, 100, 0.1)

# Plotting the total reward per episode
plt.plot(range(1000), totalreward)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()
