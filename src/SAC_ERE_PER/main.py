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


class SAC_Agent:

    def __init__(self, s_inits, T, T_res, h_min, h_max, ws_min, t_max, P_H, P_S, P_I, t_h, t_s, t_i, t, alpha, state_size, action_size, num_interactions, max_ep_len, c_k_min, eta_0, eta_T, buffer_size, batch_size, learning_rate, gamma, beta1, beta_start, beta_frames, epsilon, tau):

        self.s_inits=s_inits
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
        self.c_k_min = c_k_min

        self.K= 0
        
        self.s=[]
        self.a=[]
        self.D=[]
        self.priorities=[]
        self.r=[]
        self.scores=[]
        
        self.eta_0= eta_0
        self.eta_T= eta_T
        self.score= 0
        self.c_k= 0

        self.buffer_size= buffer_size

        self.batch_size= batch_size 

        self.gamma= gamma
        self.beta1= beta1
        self.beta_start= beta_start
        self.beta_frames= beta_frames
        self.epsilon= epsilon
        self.tau= tau
        
    #Setup loggers
    def loggers(self):

        self.rewards_logger = setup_logger('rewards_logger','./logs/rewards_logger.txt')
        self.states_logger = setup_logger('states_logger','./logs/states_logger.txt')
        self.actions_logger = setup_logger('actions_logger','./logs/actions_logger.txt')
        self.next_states_logger = setup_logger('next_states_logger','./logs/next_states_logger.txt')

    #Get scores
    def get_scores(self):

        return self.scores
        
    #Prioritized Replay
    def get_mini_batch(self, c_k, beta2):

        priorities= self.priorities

        N = self.buffer_size

        if c_k > N:
            c_k = N
                
        prios = np.array(list(priorities)[:c_k])
        
        #(prios)
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.beta1
        P = probs/probs.sum()
            
        #gets the indices depending on the probability p and the c_k range of the buffer
        indices = np.random.choice(c_k, self.batch_size, p=P)

        samples = [self.D[idx] for idx in indices]
                    
        #Compute importance-sampling weight for PER, should beta increase based on frame value?
        weights  = (c_k * P[indices]) ** (-beta2)

        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32)

        return samples

    #soft update function for target network params based on local network params
    def soft_update(self):

        v_weights= self.critic.value.get_weights()
        v_target_weights= self.critic.value_target.get_weights()

        for i in range(0,len(v_weights)):
            for j in range(0,len(v_weights[i])):
                v_target_weights[i][j]= (1.0- self.tau)*v_target_weights[i][j] + self.tau*v_weights[i][j]
        
        self.critic.value_target.set_weights(v_weights) 
               

    #Learning function for the SAC agent
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

                eta_t= self.eta_0 + (((self.eta_T- self.eta_0)*t)//self.num_interactions)

                self.K= self.K+1

                max_prio = 0

                if self.priorities:
                    max_prio= max(self.priorities)

                else:
                    max_prio= 1.0 # gives max priority if buffer is not empty else 1
                
                self.priorities.append(max_prio)

                #print(self.priorities)

                frame=1

                #time.sleep(30)

                print('\rTimestep {}'.format(i))
                
                if i==self.max_ep_len - 1:

                    for k in range(1,self.K):

                        #print(i," ",k)

                        N= len(self.D)

                        if N < self.c_k_min:

                          self.score= -3.75

                        else:
                          
                          #Emphasizing Recent Experience
                          c_k= max(int(N*(eta_t**((k*1000)/self.K))),self.c_k_min)

                          self.c_k= c_k
                          
                          beta2= min(1.0, self.beta_start + frame *(1.0 - self.beta_start) / self.beta_frames)
                          B= self.get_mini_batch(c_k,beta2)

                          frame= frame + 1

                          states_actions_batch=[]
                          states_batch=[]
                          actions_batch=[]
                          next_states_batch=[]
                          targets_batch=[]
                          target_values_batch=[]
                          Q_batch1=[]
                          Q_batch2=[]
                          
                          for b in B:

                              """
                              Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
                              Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
                              Critic_loss = MSE(Q, Q_target)
                              Actor_loss = α * log_pi(a|s) - Q(s,a)
                              where:
                                  actor_target(state) -> action
                                  critic_target(state, action) -> Q-value
                              """

                              delta1=0
                              delta2=0

                              self.score= self.score + b[2]/c_k

                              ip= b[0]+b[1]

                              states_batch.append(b[0])

                              states_actions_batch.append(ip)

                              next_states_batch.append(b[3])
                              targets_batch.append(b[2] + self.gamma*self.critic.value_target.predict(np.array([b[3]])))
                              target_values_batch.append(self.critic.value_target.predict(np.array([b[3]])))
                                  
                              #td errors for critics 1 and 2
                              delta1= b[2] + self.gamma*self.critic.value_target.predict(np.array([b[3]])) - self.critic.Q1.predict(np.array([ip]))
                              delta2= b[2] + self.gamma*self.critic.value_target.predict(np.array([b[3]])) - self.critic.Q2.predict(np.array([ip]))

                              expctd= (delta1[0][0] + delta2[0][0])//2.0 + self.epsilon

                              total_loss= expctd

                              if total_loss>0:
                                total_loss=0

                              #Priority updates
                              self.priorities[self.D.index(b)]= 1.0+(delta1[0][0] + delta2[0][0])//2.0 + self.epsilon

                              actions=[]

                              actions.append(b[1][0]+total_loss)
                              actions.append(b[1][1]+total_loss)
                              actions.append(b[1][2]+total_loss)

                              actions_batch.append(actions)

                              if self.priorities[self.D.index(b)]<0.0:
                                self.priorities[self.D.index(b)]=1e-6

                          self.critic.value.fit(np.asarray(next_states_batch), np.asarray(target_values_batch), epochs=1, verbose=0)
                          self.critic.Q1.fit(np.asarray(states_actions_batch), np.asarray(targets_batch), epochs=1, verbose=0)
                          self.critic.Q2.fit(np.asarray(states_actions_batch), np.asarray(targets_batch), epochs=1, verbose=0)                    
                          self.actor.model.fit(np.asarray(states_batch), np.asarray(actions_batch), epochs=1, verbose=0)
                          
                          #Soft update on critics 1 and 2
                          self.soft_update()

                    score= np.asarray(self.score).tolist()/(self.K*(self.t+1))

                    print('\rInteraction {}  Reward: {:.10f}\n'.format(self.t, score))

                    self.K=0
                    self.s.clear()
                    self.a.clear()
                    self.r.clear()
                    self.scores.append(score)

                    self.t= self.t+1

          if self.t== self.num_interactions:
            self.plot()

    #Plotting rewards vs interactions
    def plot(self):        

        x = np.linspace(0, self.num_interactions, self.num_interactions)
        plt.plot(x, np.array(self.scores), color='blue')
        plt.title("Average returns vs Number of Interactions for SAC+ERE+PER")
        plt.xlabel("Number of Interactions")
        plt.ylabel("Average returns");
        plt.savefig("../../plots/Avg_returns_vs_interactions_SAC.jpg",dpi=300)
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

  print("\nAutomated Water Dam Controller using SAC+ERE+PER")
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
  c_k_min = 50 # original = 5000

  eta_0= 0.996
  eta_T= 1

  #Buffer size for PER
  buffer_size= int(1e6)

  #Network training constants
  batch_size= 50
  learning_rate= 5e-4

  #Other constants
  gamma= 0.99
  beta1= 1
  beta_start= 0.4
  beta_frames= int(1e5)
  epsilon= 1e-5
  tau= 1e-2         

  s_inits= configs["s_inits"]
  

  #with tf.device('/gpu:0'):
  agent=SAC_Agent(s_inits, T, T_res, h_min, h_max, ws_min, t_max, P_H, P_S, P_I, t_h, t_s, t_i, t, alpha, state_size, action_size, num_interactions, max_ep_len, c_k_min, eta_0, eta_T, buffer_size, batch_size, learning_rate, gamma, beta1, beta_start, beta_frames, epsilon, tau)            
  agent.loggers()
  
  #app.run(host= '192.168.43.27')            
  with tf.device('/GPU:0'):
      start= time.time()
      agent.learn() 
      end= time.time()
      print("Scores: ",agent.get_scores())
      print("Time elapsed: ", end- start," secs")      

            
        

    

    
