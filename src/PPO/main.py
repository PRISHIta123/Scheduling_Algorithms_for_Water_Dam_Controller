import configparser
import json
import numpy as np
import tensorflow as tf
tf.random.set_seed(1000)
import actor
import critic
import matplotlib.pyplot as plt
from utils import setup_logger
import traceback
import contextlib
import time
import random
random.seed(0)
#from flask import Flask, render_template, request

config = configparser.RawConfigParser()
config.read("../env_states.conf")

configs = {
    "s_inits": json.loads(config.get("env","s_inits")),
}

#next_readings=[]

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
  
   var= request.form

   global next_readings
   next_readings=var.values()
   
   return var
'''

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))


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

class PPO_Agent:

    def __init__(self, s_inits, T, T_res, h_min, h_max, ws_min, t_max, P_H, P_S, P_I, t_h, t_s, t_i, t, alpha, state_size, action_size, num_interactions, max_ep_len, gamma,epsilon):

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
        self.alpha= alpha
        self.state_size= state_size
        self.action_size= action_size
        self.learning_rate= learning_rate

        self.actor= actor.Actor(self.T, self.T_res, self.h_min, self.h_max, self.ws_min, self.t_max, self.P_H, self.P_S, self.P_I, self.t_h, self.t_s, self.t_i, self.alpha, self.learning_rate)
        self.critic= critic.Critic(self.state_size, self.action_size, self.learning_rate)
        
        self.num_interactions= num_interactions

        self.max_ep_len = max_ep_len 

        self.K= 0
        
        self.s=[]
        self.a=[]
        self.D=[]
        self.r=[]
        self.scores=[]
        
        self.score= 0

        self.gamma= gamma

        #Clipping Threshold
        self.epsilon= epsilon
        
    #Setup loggers
    def loggers(self):

        self.rewards_logger = setup_logger('rewards_logger','./logs/rewards_logger.txt')
        self.states_logger = setup_logger('states_logger','./logs/states_logger.txt')
        self.actions_logger = setup_logger('actions_logger','./logs/actions_logger.txt')
        self.next_states_logger = setup_logger('next_states_logger','./logs/next_states_logger.txt')

    #Get scores
    def get_scores(self):

        return self.scores
               

    #Learning function for the PPO agent
    def learn(self):
            
          while self.t<self.num_interactions:

            print("Interaction : {}".format(self.t))

            #s_init= [sr.dam_level(),sr.solar_power(),sr.reservoir_level(),0.0,sr.water_speed(),0.0,0]
            s_init= self.s_inits[self.t]
            #readings= sr.get_readings()
            #s_init=[float(x) for x in readings]
            self.s.append(s_init)

            self.states_logger.info("\n\nInteraction : {} \n\n".format(self.t))
            self.actions_logger.info("\n\nInteraction : {} \n\n".format(self.t))
            self.rewards_logger.info("\n\nInteraction : {} \n\n".format(self.t))
            self.next_states_logger.info("\n\nInteraction : {} \n\n".format(self.t))

            for i in range(0,self.max_ep_len):
                
                #Sample action from policy
                self.a.append(self.actor.get_action(self.s[i]))

                #global next_readings
                #result1 = next_readings

                result1=[dam_level(),solar_power(),reservoir_level(),water_speed()]

                #Sample transition from environment
                self.s.append(self.actor.get_next_state(self.s[i],self.a[i],result1))

                self.r.append(self.actor.reward(self.s[i],self.a[i]))                
                
                rb=[self.s[i],self.a[i],self.r[i],self.s[i+1]]
                self.D.append(rb)

                self.states_logger.info("Timesteps: {} States: {}".format(i,self.s[i]))
                self.actions_logger.info("Timesteps: {} Actions: {}".format(i,self.a[i]))
                self.rewards_logger.info("Timesteps: {} Rewards: {}".format(i,self.r[i]))
                self.next_states_logger.info("Timesteps: {} Next_States: {}".format(i,self.s[i+1]))

                #time.sleep(30)

                print('\rTimestep {}'.format(i))
                
                
            states_actions_batch=[]
            states_batch=[]
            actions_batch=[]
            next_states_batch=[]
            rewards_batch=[]
            
            for b in self.D:

                self.score= self.score + b[2]

                states_batch.append(b[0])

                #Advantage function estimate 
                #values_batch.append(self.critic.value.predict(np.array([b[3]]))[0][0])
                    
                #Unbiased Estimate of advantage function using TD Error: Q(s,a)- V(s)
                advantage= b[2] + self.gamma*self.critic.value.predict(np.array([b[3]])) - self.critic.value.predict(np.array([b[0]]))
                
                policy_prev= self.actor.prev_policy(b[0])
                policy= b[2]
              
                rewards_batch.append(b[2])

                # Finding the ratio of old and new policies (pi_theta / pi_theta__old)
                ratio = policy/policy_prev
                
                #Surrogate loss computations
                surr1 = ratio * advantage

                #Clamp ratios to allowed range of proximity
                if ratio<1-self.epsilon:
                    ratio= 1-self.epsilon
                elif ratio>1+self.epsilon:
                    ratio= 1+self.epsilon

                surr2 = ratio * advantage

                # final loss of clipped objective PPO
                expctd = -min(surr1[0][0], surr2[0][0]) 

                total_loss= expctd 

                actions=[]

                actions.append(b[1][0]+total_loss)
                actions.append(b[1][1]+total_loss)
                actions.append(b[1][2]+total_loss)

                actions_batch.append(actions)

            #Value Function Update
            self.critic.value.fit(np.asarray(states_batch), np.asarray(rewards_batch), epochs=1, verbose=0)
            
            self.actor.prev_policy_update()

            #Clipped objective update           
            self.actor.model.fit(np.asarray(states_batch), np.asarray(actions_batch), epochs=1, verbose=0)

            score= np.asarray(self.score).tolist()/(self.max_ep_len*(self.t+1))

            print('\rInteraction {}  Reward: {:.10f}\n'.format(self.t, score))

            self.K=0
            self.s.clear()
            self.a.clear()
            self.r.clear()
            self.D.clear()
            self.scores.append(score)

            self.t= self.t+1

            if self.t== self.num_interactions:
              self.plot()

    #Plotting rewards vs interactions
    def plot(self):        

        x = np.linspace(0, self.num_interactions, self.num_interactions)
        plt.plot(x, np.array(self.scores), color='red')
        plt.title("Average returns vs Number of Interactions for PPO")
        plt.xlabel("Number of Interactions")
        plt.ylabel("Average returns");
        plt.savefig("../../plots/Avg_returns_vs_interactions_PPO.jpg",dpi=300)
        plt.close()


if __name__ == '__main__':

  #Add constraints
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

  print("\nAutomated Water Dam Controller using PPO")
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

  #Exploration factor
  alpha= 0.2

  #State and action space sizes
  state_size= 7
  action_size= 5

  #Number of time steps to run model for/ number of interactions
  num_interactions= 15

  max_ep_len = 25 # original = 1000

  #Learning Rates
  learning_rate= 5e-4

  #Other constants
  gamma= 0.99
  epsilon= 0.2      

  s_inits= configs["s_inits"]       

  #with tf.device('/gpu:0'):
  agent=PPO_Agent(s_inits, T, T_res, h_min, h_max, ws_min, t_max, P_H, P_S, P_I, t_h, t_s, t_i, t, alpha, state_size, action_size, num_interactions, max_ep_len, gamma,epsilon)            
  agent.loggers()
  
  #app.run(host= '192.168.43.27')            
  with tf.device('/GPU:0'):
      start= time.time()
      agent.learn() 
      end= time.time()
      print("Scores: ",agent.get_scores())
      print("Time elapsed: ", end- start," secs")        

            
        

    

    
