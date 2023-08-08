import os
import sys
import random
random.seed(0)
import numpy as np
import utils
import tensorflow as tf
tf.random.set_seed(1000)

class Policy:

    def __init__(self,state_size,learning_rate, T, T_res, h_min, h_max, ws_min, t_max, P_H, P_S, P_I, t_h, t_s, t_i,):

            self.state_size = state_size
            self.learning_rate = learning_rate
            self.model= self.make_model()

            #Add constraints
            self.T= T
            self.T_res= T_res
            self.h_min= h_min
            self.h_max= h_max
            self.ws_min= ws_min
            self.t_max= t_max

            #Add power constant values for each type of action (per unit volume)
            self.P_H= P_H
            self.P_S= P_S
            self.P_I= P_I

            #Add completion time values for each type of action (per unit volume)
            self.t_h= t_h
            self.t_s= t_s
            self.t_i= t_i

            self.max_hydro=10

            self.log_std = tf.Variable( name= 'LOG_STD', initial_value= -0.5 * 
                                    np.ones(1, dtype= np.float32))

    def custom_loss(self, y_true, y_pred):
        
            loss = tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=1)) # need to put axis
            return loss

    #create a policy network
    def make_model(self):
            
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(32,input_shape=(7,), activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
            model.add(tf.keras.layers.Dense(32, activation = tf.keras.activations.relu,kernel_regularizer=tf.keras.regularizers.l2(2e-4)))
            model.add(tf.keras.layers.Dense(3))
            optim = tf.keras.optimizers.Adam(lr = self.learning_rate)
            model.compile(loss = self.custom_loss, optimizer = optim)

            return model

    def get_next_state(self, current_state, action, nextreadings):

            irr=0

            if action[0]>0.0:
                irr=1

            res= current_state[2]

            #newvals=[float(x) for x in nextreadings]

            if action[1]>0.0:
                res= res+ action[1]/self.P_S

            nextreadings=[float(x) for x in nextreadings]

            # Reservoir water level may increase or decrease in next time step due to rain/ drying up
            # Therefore nextreadings[2] is used instead of res
            next_state= [nextreadings[0],nextreadings[1],nextreadings[2],action[2],nextreadings[3],action[3],irr]

            return next_state

    #Return reward for chosen policy based on current state
    def rollout(self, current_state, nextreadings):

            nextreadings=[float(x) for x in nextreadings]

            action=[]

            for i in range(0,5):
                action.append(0)

            state_clipped=[current_state[0]- self.T + current_state[5], max(0.0,self.T_res-current_state[2]), current_state[4], current_state[6]]

            #If water in dam has crossed threshold
            if state_clipped[0]>0.0 :

                mu = self.model.predict(np.array([current_state]))                            # Get mu from NN
                std = tf.exp(self.log_std)                                            # Take exp. of Std deviation
                act= np.asarray(mu + tf.random.normal(tf.shape(mu)) * std).tolist()                        # Sample action from Gaussian Dist
                act= np.asarray(act).tolist()

                #If field is not irrigated, get irrigation water amount
                if state_clipped[3]==0:
                    action[0] = sorted((0,np.asarray(act[0][0]).tolist(),self.h_max-self.h_min))[1]

                else:
                    action[0]=0.0

                #Get hydroelectricity amount
                if state_clipped[2]>self.ws_min:
                    action[2] = sorted((np.asarray(act[0][2]).tolist(), 0, state_clipped[0]- action[0]))[1]

                #Get reservoir storage water amount                      
                action[1] = sorted((np.asarray(act[0][1]).tolist(), 0, min(state_clipped[1],state_clipped[0]- action[0]- action[2])))[1] 

                #Get residual excess water amount
                action[3]= state_clipped[0]- sum(action[0:3])

            if action[0]<0.0:
                action[0]=0.0

            if action[1]<0.0:
                action[1]=0.0

            if action[2]<0.0:
                action[2]=0.0

            if action[3]<0.0:
                action[3]=0.0

            nc_t=0

            if action[0]>0.0:
                time_i = action[0]*self.t_i

                if time_i< self.t_max:
                    nc_t+=1
                    self.t_max-=time_i

            if action[2]>0.0:
                time_h = action[2]*self.t_h
                
                if time_h< self.t_max:
                    nc_t+=1
                    self.t_max-=time_h

            if action[1]>0.0:
                time_s = action[1]*self.t_s
            
                if time_s< self.t_max:
                    nc_t+=1
                    self.t_max-=time_s

            #Count of actions completed in current time step
            action[4]=nc_t

            #Multiply power constants with resp. actions
            action[0]= action[0]*self.P_I
            action[1]= action[1]*self.P_S
            action[2]= action[2]*self.P_H

            for i in range(0,5):
                action[i]= np.asarray(action[i]).tolist()

            next_state= self.get_next_state(current_state, action, nextreadings)

            PG_t= current_state[1]+current_state[3]
     
            P_t= sum(action[0:3])

            net= (PG_t- P_t)

            n_t= len([x for x in action[0:4] if x > 0])
            nc_t= action[4]
            
            n_rem= n_t - nc_t

            EP_t=0 

            if action[3]!=0:
                EP_t= action[3]

            c1= 0.0005
            c2= 0.1
            c3= 0.5

            reward= -c1*(net**2) - c2*n_rem - c3*EP_t
                
            return reward, next_state, action
                    






 

    
