import numpy as np
import random
from scipy.stats import truncnorm

def get_truncated_normal(low, upp, mean=0, sd=1):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_irrigation_amt(water,h_min,h_max):

    if water > 0.0:
        amt= random.uniform(0.0,h_max-h_min)
        return amt

    else:
        return 0

def get_hydro_amt(ws_min, ws , max_amt):

    if ws> ws_min:
        amt= random.uniform(0,max_amt)
        return amt

    else:
        return 0

def get_storage_amt(res_available, water):

    if res_available > 0.0:
        amt= random.uniform(0,min(water,res_available))
        return amt

    else:
        return 0



    
        
        
