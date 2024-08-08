import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")#, render_mode="human")


def mountaincar(discretisationfactor_pos,discretisationfactor_vel,alpha,gamma,num_episodes):
    learning_rate = alpha
    discount_factor = gamma
    
    n = discretisationfactor_pos  # discretization factor for position
    nv = discretisationfactor_vel  # discretization factor for velocity

    pos_rows = int((1.2 + 0.6) / n) + 1  # num of rows for position
    vel_rows = int((0.07 + 0.07) / nv) + 1  # num of rows for velocity
    cols = env.action_space.n
    
    q_values=np.zeros((pos_rows, vel_rows ,cols))
    
    rewardlist=np.zeros(num_episodes)

    for i in range(num_episodes):
        epsilon = 1/(i+1) #epsilon decay
        total_reward=0

        state=env.reset()[0]
        # print((state[0]/n)//1+12)
        done = False
        
        while not done:
            # action chosen according to epsilon-greedy policy
            # print(f"state is {state}")
            
            
            # discretised_position = int(round(state[0]/n,2))
            # shift_pos=int(round(1.2/n,2))
            # pos_row=discretised_position + shift_pos

            #discretising states
            pos_row=int((state[0]+1.2)/n)
            vel_row=int((state[1]+0.07)/nv)


            # discretised_velocity = int(round(state[1]/nv,2))
            # shift_vel=int(round(0.07/nv))
            # vel_row=discretised_velocity + shift_vel
            # # print(f"pos and vel rows are {pos_row,vel_row}")
            
            if np.random.uniform(0,1)<epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_values[pos_row,vel_row])        
            
            new_state,reward,terminated,truncated,info = env.step(action)
            # print(f"{new_state}")
            done = truncated or terminated                                      #becomes true when episode terminates or gets truncated
            total_reward+=reward

            # discretised_newposition = int(round(new_state[0]/n,2))
            # new_pos_row=discretised_newposition+shift_pos

            # discretised_newvelocity = int(round(new_state[1]/nv,2))
            # new_vel_row=discretised_newvelocity + shift_vel
            
            # discretising new states
            new_pos_row=int((new_state[0]+1.2)/n)
            new_vel_row=int((new_state[1]+0.07)/nv)
            
            #update the q_values according to the Q-policy
            q_values[pos_row,vel_row,action]+= learning_rate*(reward + discount_factor*np.max(q_values[new_pos_row,new_vel_row])-q_values[pos_row,vel_row,action])

            state=new_state
            
        print(f"The reward in episode {i} is {total_reward}")
        rewardlist[i]=total_reward
    
    return rewardlist
    



rewardlist=mountaincar(0.2,0.02,0.01,0.9,3000)

