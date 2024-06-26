import numpy as np
import matplotlib.pyplot as plt
import random

#Monte Carlo Method
def epsilongreedy(episodes, timesteps, epsilon):
    q=np.zeros((timesteps,5)) #action values
    n=np.zeros((timesteps,5)) #no. of times I took an action
    netreward=list() #net reward post each episode (for plotting)
    for i in range(episodes): 
        #storing all actions and rewards for a particular episode
        actionlist=list()
        rewardlist=list()
        
        for j in range(timesteps):
            # print(f"entered iteration {j}")
            prob=np.random.rand()
            # print(f"The random no. is {prob}")
            if(prob<=epsilon):
                #choose all actions with equal probability
                action=np.random.randint(5)
                # print(f"The action is {action}")
                if(action==0):
                    #gaussian dist
                    reward=np.random.normal(2,1)
                    
                elif(action==1):
                    #fair coin toss
                    reward=np.random.randint(2)*(-11)+5
                elif(action==2):
                    #poisson dist
                    reward=np.random.poisson(2)
                elif(action==3):
                    #exp dist
                    reward=np.random.exponential(3)
                else:
                    #crazy button, choose any of the above 4 with equal prob
                    n2=np.random.randint(4)
                    if(n2==0):
                        #gaussian dist
                        reward=np.random.normal(2,1)
                    elif(n2==1):
                        #fair coin toss
                        reward=np.random.randint(2)*(-11)+5
                    elif(n2==2):
                        #poisson dist
                        reward=np.random.poisson(2)
                    elif(n2==3):
                        #exp dist
                        reward=np.random.exponential(3)

                # print(f"The reward is {reward}")
                actionlist.append(action)
                rewardlist.append(reward)

            else:
                
                # print(f"entered else")
                # print(f"The q is {q}")
                
                #find all indices of max q values
                max_q=np.max(q[j,:])
                max_q_states=list()
                for k in range(5):
                    if(q[j,k]==max_q): max_q_states.append(k)
                action=random.choice(max_q_states)
                # print(f"The action is {action}")
                
                if(action==0):
                    #gaussian dist
                    reward=np.random.normal(2,1)
                    
                elif(action==1):
                    #fair coin toss
                    reward=np.random.randint(2)*(-11)+5
                elif(action==2):
                    #poisson dist
                    reward=np.random.poisson(2)
                elif(action==3):
                    #exp dist
                    reward=np.random.exponential(3)
                else:
                    n3=np.random.randint(4)
                    if(n3==0):
                        #gaussian dist
                        reward=np.random.normal(2,1)
                    elif(n3==1):
                        #fair coin toss
                        reward=np.random.randint(2)*(-11)+5
                    elif(n3==2):
                        #poisson dist
                        reward=np.random.poisson(2)
                    elif(n3==3):
                        #exp dist
                        reward=np.random.exponential(3)
                
                # print(f"The reward is {reward}")
                actionlist.append(action)
                rewardlist.append(reward)

        #updating the action values after episode 
        for l in range(timesteps):
            n[l,actionlist[l]]=n[l,actionlist[l]]+1
            stepsize=1/n[l,actionlist[l]]

            future_rewards=np.array(rewardlist[l:])
            # print(f"the future rewards are {future_rewards} ")
            #discounting the future rewards
            discount=(np.geomspace(1, 0.3**(timesteps-l-1),timesteps-l))   
            # print(f"The discount is {discount}")  
              
            gt=np.sum(np.dot(future_rewards,discount))
            # print(f"gt is {gt}")
            error = gt - q[l,actionlist[l]]
            q[l,actionlist[l]]=q[l,actionlist[l]]+stepsize*error
            
        # print(f"The matrix n is {n}")
        # print(f"The action values are {q}")

        netreward.append(sum(rewardlist))
          
    # print(f"The matrix n is {n}")
    # print(f"The action values are {q}")
    
    return netreward



fig,axs=plt.subplots(3,1)
plt.figure(figsize=(32,30))
episodes=1000
timesteps=100

episodes_list = [i for i in range(episodes)]
for i in range(3):
    if (i==0): epsilon = 0
    elif(i==1): epsilon=0.01
    else: epsilon=0.1
    totalrewardlist=np.zeros((40,episodes))
    for j in range(40):
        print(f"inside iteration {j} for epsilon {epsilon}")
        totalrewardlist[j, : ]=(np.array((epsilongreedy(episodes,timesteps,epsilon)))).reshape(1,1000)
    totalreward=np.sum(totalrewardlist,axis=0)/40
    axs[i].plot(episodes_list,totalreward)
    axs[i].set_title(f"Epsilon={epsilon}")
    axs[i].set_xlabel("Episode No.")
    axs[i].set_ylabel("Net reward after episode")

fig.suptitle("Average reward after 40 iterations of 1000 episodes each")
fig.tight_layout()

fig.show()
fig.waitforbuttonpress()



        


