import numpy as np
import sensor_readings as sr

#global environment initial states
global s_inits
s_inits=[]

num_interactions=15

for i in range(0,num_interactions):
	s_init= [sr.dam_level(),sr.solar_power(),sr.reservoir_level(),0.0,sr.water_speed(),0.0,0]
	s_inits.append(s_init)

