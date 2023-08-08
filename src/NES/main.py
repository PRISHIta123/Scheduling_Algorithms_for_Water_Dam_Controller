import configparser
import json
import policy
import matplotlib.pyplot as plt
import utils
import traceback
import contextlib
import time
import tensorflow as tf
tf.random.set_seed(1000)
import numpy as np
import random
random.seed(0)
from flask import Flask, render_template, request

next_readings=[]

config = configparser.RawConfigParser()
config.read("../env_states.conf")

configs = {
    "s_inits": json.loads(config.get("env","s_inits")),
}

'''
app = Flask(__name__)

@app.route('/readings',methods = ['POST', 'GET'])
def get_readings():
  
    if request.method == 'POST':
      result = request.form

      with tf.device('/GPU:0'):
        agent.learn(result.values())
      
    return result

@app.route('/nextreadings',methods=['GET','POST'])
def get_next_readings():
  
   if request.method == 'POST':
     var= request.form

     global next_readings
     next_readings=var.values()

   return var
'''

def dam_level():
    dl = random.uniform(0.0, 6.0)

    return dl


def solar_power():
    sp = random.uniform(0.0, 15.3)

    return sp


def reservoir_level():
    rl = random.uniform(0.0, 2.0)

    return rl


def water_speed():
    ws = random.uniform(0.0, 2.0)

    return ws

class NES_Agent:

    def __init__(self, s_inits, T, T_res, h_min, h_max, ws_min, t_max, P_H, P_S, P_I, t_h, t_s, t_i, t, num_interactions, state_size, population_size, n_step, n_rollout, batch_size, learning_rate, sigma, l2_decay):

        self.s_inits= s_inits
        self.T= T
        self.T_res= T_res
        self.h_min= h_min
        self.h_max= h_max
        self.ws_min= ws_min
        self.t_max= t_max
        self.P_H= P_H
        self.P_S= P_S
        self.P_I= P_I
        self.t_h= t_h
        self.t_s= t_s
        self.t_i= t_i
        self.t= t
        self.state_size= state_size
        self.population_size= population_size
        self.learning_rate= learning_rate
        
        self.num_interactions= num_interactions
        self.n_step= n_step
        self.n_rollout= n_rollout
        
        self.s=[]
        self.a=[]
        self.D=[]
        self.r=[]
        self.total_rewards=[]

        self.total_reward= 0

        self.batch_size= batch_size 

        self.sigma= sigma

        self.l2_decay= l2_decay

        self.policy= policy.Policy(state_size,learning_rate,T, T_res, h_min, h_max, ws_min, t_max, P_H, P_S, P_I, t_h, t_s, t_i)
        self.dummy_policy= policy.Policy(state_size,learning_rate,T, T_res, h_min, h_max, ws_min, t_max, P_H, P_S, P_I, t_h, t_s, t_i)
        
    #Setup loggers
    def loggers(self):

        self.rewards_logger = utils.setup_logger('rewards_logger','./logs/rewards_logger.txt')
        self.states_logger = utils.setup_logger('states_logger','./logs/states_logger.txt')

    #Get scores
    def get_scores(self):

        return self.total_rewards

    #Sample perturbed parameters and epsilon values from a normal distribution
    def sample(self, param):

        #weights= list(param)[:]
        wts= param
        #wts=[]

        '''
        for i in range(0,len(weights)):
          if i==0 or i%2==0:
            wts.append(weights[i].tolist())
          else:
            wts.append(weights[i].tolist())
        '''

        #print(list(weights[0]))

        population_params=[]

        epsilon= tf.random.normal([int(self.population_size/2)], mean=0.0, stddev=self.sigma, dtype=tf.dtypes.float32)
        epsilons=tf.concat([epsilon, -epsilon],0)
        
        for n in range(0,self.population_size):
            dummy_weights=[]
            if n<self.population_size/2:
                for i in range(0,len(wts)):
                    if i==0 or i%2==0:
                        l1=[]
                        for j in range(0,len(wts[i])):
                            l2=[]
                            for k in range(0,len(wts[i][j])):
                                l2.append(wts[i][j][k]+epsilon[n])
                            l1.append(l2)
                        dummy_weights.append(np.array(l1))
                    else:
                        l1=[]
                        for j in range(0,len(wts[i])):
                            l1.append(wts[i][j]+epsilon[n])
                        dummy_weights.append(np.array(l1))

            else:
                for i in range(0,len(wts)):
                    if i==0 or i%2==0:
                        l1=[]
                        for j in range(0,len(wts[i])):
                            l2=[]
                            for k in range(0,len(wts[i][j])):
                                l2.append(wts[i][j][k]-epsilon[self.population_size-n-1])
                            l1.append(l2)
                        dummy_weights.append(np.array(l1))
                    else:
                        l1=[]
                        for j in range(0,len(wts[i])):
                            l1.append(wts[i][j]-epsilon[self.population_size-n-1])
                        dummy_weights.append(np.array(l1))
                            
            population_params.append(dummy_weights)

        epsilons=np.transpose(np.asarray(epsilons)).tolist()

        return population_params, epsilons
        
    #Evaluate policy on the environment for n_rollout times
    def eval_policy(self, policy, current_state, param, reading1):

        total_reward = 0
        policy.model.set_weights(param[0])
        for _ in range(self.n_rollout):
            total_reward += policy.rollout(current_state, reading1)[0]
        return total_reward / self.n_rollout

    #Returns array of rewards for a population of parameters
    def evaluate(self, population_params, current_state, reading1):

        rewards = []
        reward_array = np.zeros(self.population_size, dtype=np.float32)

        for i in range(0,len(population_params[0])):
            params=[]
            for j in range(0,len(population_params)):
                params.append(population_params[j][i])
            rewards.append(self.eval_policy(self.dummy_policy, current_state, params, reading1))

        return np.asarray(rewards)
        
    #Calculate Gradients based on rewards received
    def calculate_gradients(self, rewards, means, epsilons):

        rrs = utils.rank_transformation(rewards)
        rrs=rrs.tolist()

        ranked_rewards=[]
        grads=[]

        for i in range(0,len(rrs)):
          ranked_rewards.append(rrs[i])

        grads=-(np.matmul(np.asarray(ranked_rewards),epsilons) / (np.shape(rewards)[0] * self.sigma))

        return grads

    #parameter update function for policy network params to optimize gradient
    def parameter_update(self, rewards, means, epsilons):

        weights= np.array(self.policy.model.get_weights(),dtype=object)

        grads= np.array(self.calculate_gradients(rewards, means[0], epsilons[0]),dtype=object)

        weights= weights - grads
        
        self.policy.model.set_weights(weights) 
               

    #Learning function for the NES agent
    def learn(self):#,readings):

        while self.t<self.num_interactions:

          print("Interaction : {}".format(self.t))

          #s_init= [sr.dam_level(),sr.solar_power(),sr.reservoir_level(),0.0,sr.water_speed(),0.0,0]
          s_init= self.s_inits[self.t]
          #readings= sr.get_readings()
          #s_init=[float(x) for x in readings]
          self.s.append(s_init)

          current_state= s_init

          self.states_logger.info("\n\nInteraction : {} \n\n".format(self.t))
          self.rewards_logger.info("\n\nInteraction : {} \n\n".format(self.t))

          for gen in range(0,self.n_step):

            param1 = self.policy.model.get_weights()
            population_params1, epsilons1 = self.sample(param1)

            population_params=[]
            population_params.append(population_params1)

            time.sleep(10)    

            #global next_readings
            #result1 = next_readings

            #result1= [float(x) for x in result1]
            result1=[dam_level(),solar_power(),reservoir_level(),water_speed()]

            # Evaluate Population
            rewards= self.evaluate(population_params, current_state, result1)   

            self.r.append(np.asarray(rewards).tolist())   

            self.states_logger.info("N_Steps: {} States: {}".format(gen,self.s[gen]))
            self.rewards_logger.info("N_Steps: {} Rewards: {}".format(gen,self.r[gen]))

            print('\rN_Step {}'.format(gen))

            self.parameter_update(rewards, [param1], [epsilons1])

            next_state= self.policy.rollout(current_state, result1)[1]
            self.s.append(next_state)  

            action= self.policy.rollout(current_state, result1)[2]
            '''
            print("\n Power assigned for irrigation: ",action[0])
            print("\n Power assigned for reservoir storage: ",action[1])
            print("\n Power assigned for hydroelectricity generation: ",action[2])
            print("\n Amount of water left in dam: ", action[3])
            print("\n Number of actions completed: ",action[4],"\n")
            '''

            current_state= next_state

          #result1 = next_readings
          result1= [dam_level(),solar_power(),reservoir_level(),water_speed()]

          param = self.policy.model.get_weights()
          self.total_reward=  self.total_reward+self.eval_policy(self.policy, current_state, [param], result1)
          total_reward= np.asarray(self.total_reward).tolist()/(self.t+1)

          print('\rInteraction {}  Reward: {:.10f}\n'.format(self.t, total_reward))

          self.s.clear()
          self.r.clear()
          self.total_rewards.append(total_reward)
          self.t= self.t+1

          if self.t== self.num_interactions:
            self.plot()

    #Plotting rewards vs interactions
    def plot(self):        

        x = np.linspace(0, self.num_interactions, self.num_interactions)
        plt.plot(x, np.array(self.total_rewards), color='green')
        plt.title("Average returns vs Number of Interactions for NES")
        plt.xlabel("Number of Interactions")
        plt.ylabel("Average returns");
        plt.savefig("../../plots/Avg_returns_vs_interactions_NES.jpg",dpi=300)
        plt.close()


if __name__ == '__main__':

  #Add constraints
  #T= 3.0
  T= 4.0
  T_res= 4.0
  h_min= 1.5
  h_max= 2.5
  ws_min= 1.5
  t_max= 10

  #Add power constant values for each type of action (per unit volume)
  P_H= 6.0
  P_S= 5.0
  P_I= 3.0

  #Add completion time values for each type of action (per unit volume)
  t_h= 0.1
  t_s= 0.05
  t_i= 0.2

  #Interaction count
  t=0

  print("\nAutomated Water Dam Controller using NES")
  print("\nDam Threshold: ", T)
  print("\nReservoir Threshold: ",T_res)
  print("\nMin. Irrigation Volume: ",h_min)
  print("\nMax. Irrigation Volume: ",h_max)
  print("\nMin. Water Speed for Hydroelectricity Generation: ",ws_min)
  print("\nMax. no. of timesteps to perform an action: ",t_max)

  print("\nPower to generate hydroelectricity from unit volume of water (to run turbine): ",P_H)
  print("\nPower to store unit volume of water (control motors to open dam gates): ",P_S)
  print("\nPower to irrigate field using unit volume of water (control motors to open dam gates and run the water pump): ",P_I)

  print("\nTime to generate hydroelectricity from unit volume of water (to run turbine): ",t_h)
  print("\nTime to store unit volume of water (control motors to open dam gates): ",t_s)
  print("\nTime to irrigate field using unit volume of water (control motors to open dam gates and run the water pump): \n",t_i)


  #State and action space sizes
  state_size= 7

  #Number of time steps to run model for/ number of interactions
  num_interactions= 15

  n_rollout = 5
  n_step = 25
  l2_decay = 0.005
  population_size = 50
  sigma = 0.02

  #Network training constants
  batch_size= 50
  learning_rate= 5e-4         

  s_inits= configs["s_inits"]  

  #with tf.device('/gpu:0'):
  agent=NES_Agent(s_inits, T, T_res, h_min, h_max, ws_min, t_max, P_H, P_S, P_I, t_h, t_s, t_i, t, num_interactions, state_size, population_size, n_step, n_rollout, batch_size, learning_rate, sigma, l2_decay)            
  agent.loggers()
      
  with tf.device('/GPU:0'):
      start= time.time()
      agent.learn() 
      end= time.time()
      print("Scores: ",agent.get_scores())
      print("Time elapsed: ", end- start," secs")   
            

            
        

    

    
