
import math
import numpy as np
import random as rd



NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500


DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1


#coding exercise 2
#Decay Greedy exploration
DECAY_EPSILON = False
EPSILON_MAX = 1
EPSILON_DECAY = 1 - (1/NUM_EPISODES)

#Boltzman exploration
BOLTZMAN = False





class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, map_name, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE): 
        self.name = "agent1"
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.q_table = np.zeros((num_states, num_actions))
        self.policy = []
        self.goal_dict = []
        self.timesteps = []
        self.map_name = map_name
        self.done = False

        if DECAY_EPSILON:
            self.epsilon = EPSILON_MAX
        else:
            self.epsilon = EPSILON
        self.reward = 0
       
       
        self.episodes = NUM_EPISODES
        self.episode_length = MAX_EPISODE_LENGTH
        self.episilon = EPSILON
       



    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        if BOLTZMAN:
            pass 
        else:
            if DECAY_EPSILON:
                self.epsilon *= EPSILON_DECAY
                if self.epsilon < 0.01:
                    self.epsilon = 0.01
            else:
                pass


    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """

        self.done = done

        if not done:
            next_q_value = np.max(self.q_table[next_state])
            self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state,action] + self.learning_rate*(reward + self.discount* next_q_value)
        
        else:
            self.reward = reward
            self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + self.learning_rate * reward
       
        


    def select_action(self, state): 
        """
        Returns an action, selected based on the current state
        """

        if BOLTZMAN:
            temp_random= rd.uniform(0,1)
            p = {}
            sumq = 0.0
            limit = {}
            T  = 2
            for i in range(0,self.num_actions):
                sumq += math.exp(self.q_table[state,i]/T)
            for i in range(0,self.num_actions):
                p[i] = math.exp(self.q_table[state,i]/T)/sumq
                if i == 0:
                    limit[i] = (p[i])
                else:
                    limit[i] = (p[i]+limit[i-1])
                if limit[i] > temp_random:
                    action = i
                    break
        
            return action

        else: 

            if rd.random() < self.epsilon:
                return rd.randint(0,self.num_actions-1)
            else:
                return np.random.choice(np.argwhere(self.q_table[state,:] == np.max(self.q_table[state,:])).flatten().tolist())

       
    
    def find_policy(self):

        self.policy = []
        
        for i in range(0,self.num_states):
            optimal_action = np.argmax(self.q_table[i])
            self.policy.append(optimal_action)
        
        self.policy = np.asarray(self.policy, dtype=object)
        
        if self.map_name == "walkInThePark":
            self.policy = np.reshape(self.policy, (6,8))
        else:
            self.policy = np.reshape(self.policy, (1,13))
        
        self.policy[self.policy==0] = ' ← '
        self.policy[self.policy==1] = ' ↓ '
        self.policy[self.policy==2] = ' → '
        self.policy[self.policy==3] = ' ↑ '

    def goal_dictionary(self):
            return self.goal_dict
    
    def get_timesteps(self):
            return self.timesteps


    def report(self, steps, episode):
        """
        Function to print useful information, printed during the main loop
        # """
        if BOLTZMAN:
           print('Exploration strategy is: Boltzman ')
        else:
            if DECAY_EPSILON:
                 print('Exploration strategy is: Decay Greedy')
            else:
                print('Exploration strategy is: Greedy')

        self.find_policy()
        print("policy:\n", self.policy)
        print(self.q_table)

        

       
        if self.done and self.reward == 10:
            self.goal_dict.append((episode, steps))
            self.timesteps.append(steps)


   


   

        

    

