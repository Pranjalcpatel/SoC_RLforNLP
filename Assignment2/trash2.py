import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make("MountainCar-v0")
done = False
observation, info = env.reset()

n = 0.2  # discretization factor for position
nv = 0.02  # discretization factor for velocity

pos_rows = int((1.2 + 0.6) / n) + 1  # 1.2 is the range of position
vel_rows = int((0.07 + 0.07) / nv) + 1  # 0.07 is the range of velocity
cols = env.action_space.n

q_values = np.zeros((pos_rows, vel_rows, cols))

learning_rate = 0.01
discount_factor = 0.9
n_episodes = 3000

def discretize_state(state):
    pos, vel = state
    pos_discrete = int((pos + 1.2) / n)
    vel_discrete = int((vel + 0.07) / nv)
    return pos_discrete, vel_discrete

def episode(i):
    epsilon = 1/(i+1)
    total_reward = 0
    state = env.reset()[0]
    done = False
    
    while not done:
        pos_row, vel_row = discretize_state(state)
        
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values[pos_row, vel_row])
        
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        new_pos_row, new_vel_row = discretize_state(new_state)
        
        best_future_q = np.max(q_values[new_pos_row, new_vel_row])
        current_q = q_values[pos_row, vel_row, action]
        
        q_values[pos_row, vel_row, action] += learning_rate * (reward + discount_factor * best_future_q - current_q)

        state = new_state
        
    return total_reward 

reward = np.zeros(n_episodes)
for i in tqdm(range(n_episodes)):
    reward[i] = episode(i)
    print(f"Reward at episode {i} is {reward[i]}")

# Plotting the reward
plt.plot(reward)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward vs Episode')
plt.show()








# import gymnasium as gym
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from tqdm import tqdm
# import gymnasium as gym
# env = gym.make("MountainCar-v0")#, render_mode="human")
# done=False
# observation, info = env.reset()
# print(env.action_space.n)
# n=0.2 #descretisation factor #no. of states = 1.8/n+1 
# pos_rows = 1.8/n + 1
# pos_rows=int(round(pos_rows,2)) #to remove floating point error
# nv=0.02
# vel_rows=0.14/nv+1
# vel_rows=int(round(vel_rows,2))
# cols = env.action_space.n
# print(pos_rows,vel_rows,cols)
# q_values=np.zeros((pos_rows, vel_rows ,cols))
# print(q_values.shape)
# learning_rate = 0.01
# discount_factor = 0.9
# n_episodes = 3000

# def episode(i):
#     epsilon=1/(i+1)
#     total_reward=0

#     state=env.reset()[0]
#     # print((state[0]/n)//1+12)
#     done = False
    
#     while not done:
#         #choose action by epsilon-greedy policy
#         # print(f"state is {state}")
        
        
#         discretised_position = int(round(state[0]/n,2))
#         shift_pos=int(round(1.2/n,2))
#         pos_row=discretised_position + shift_pos

#         discretised_velocity = int(round(state[1]/nv,2))
#         shift_vel=int(round(0.07/nv))
#         vel_row=discretised_velocity + shift_vel
#         # print(f"pos and vel rows are {pos_row,vel_row}")
        
#         if np.random.uniform(0,1)<epsilon:
#             action = env.action_space.sample()
#         # elif np.max(q_values[pos_row,vel_row])>0:
#         #     action = np.argmax(q_values[pos_row,vel_row])
#         else:
#             action = np.argmax(q_values[pos_row,vel_row])        
        
#         new_state,reward,terminated,truncated,info = env.step(action)
#         # print(f"{new_state}")
#         done = truncated or terminated                                      #becomes true when episode terminates or gets truncated
#         total_reward+=reward

#         discretised_newposition = int(round(new_state[0]/n,2))
#         new_pos_row=discretised_newposition+shift_pos

#         discretised_newvelocity = int(round(new_state[1]/nv,2))
#         new_vel_row=discretised_newvelocity + shift_vel
        
#         #update q_values
#         q_values[pos_row,vel_row,action]+= learning_rate*(reward + discount_factor*np.max(q_values[new_pos_row,new_vel_row])-q_values[new_pos_row,new_vel_row,action])

#         state=new_state
        
#     return total_reward 
# reward=np.zeros(n_episodes)
# for i in range(n_episodes):
#     reward[i]=episode(i)
#     print(f"reward at {i}th episode is {reward[i]}")
