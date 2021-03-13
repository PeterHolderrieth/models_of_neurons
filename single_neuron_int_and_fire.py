import numpy as np
import matplotlib.pyplot as plt

def step_int_and_fire(v,v_th,v_reset,e_l,r_m,i_e,tau_mem,dt):
    '''
    Function to simulate next time step of integrate and fire model (using Euler method).
    Input:
        v_th - float - threshold potential when neuron starts firing
        v_reset - float - po
        e_l - float - resting membrane potential
        r_m - float>0 - membrane resistance
        i_e - float - input current
        tau_mem - float - time constant of membrane (capacitance X resistance)
        dt - float >0 - discretization length (time step to simulate)
     Output:
         float - next potential after time step dt
         int in [0,1] - indicates whether potential was reset
        
    '''
    
    #Compute derivative:
    deriv=(e_l-v+r_m*i_e)/tau_mem
    
    #Compute next step according to Euler method:
    step=v+dt*deriv

    #Reset if threshold is exceeded:
    if step>v_th:
        return(v_reset,1)
    else:
        return(step,0)
    

def sim_int_and_fire(v0,n_steps,v_th,v_reset,e_l,r_m,i_e,tau_mem,dt):
    '''
    Function to simulate integrate and fire model over 
    Input: 
        v0 - float - initial condition
        n_steps - int - number of time steps to simulate
        for remaining parameters, see step_int_and_fire
    Output: 
    time_list - list of floats - times of simulation 
    v_list - list of floats - potentials over time
    n_spikes - int - number of spikes 
        
    '''
    #Initialize time and potential lists:
    time_list=[0.]
    v_list=[v0]
    
    #Number of spikes:
    n_spikes=0
    
    for it in range(n_steps):
        
        #Simulate next step:
        v_new,reset=step_int_and_fire(v_list[-1],v_th,v_reset,e_l,r_m,i_e,tau_mem,dt)
        
        #Update time and potential list:
        time_list.append(time_list[-1]+dt)
        v_list.append(v_new)
        
        #Update number of spikes:
        n_spikes+=reset
        
    return(time_list,v_list,n_spikes)

def min_current_to_prod_action_potential(e_l,v_th,r_m):
    '''
    This function gives the minimum current necessary to produce an action potential
    Input: e_l - float - resting membrane potential 
           v_th - float - firing threshold potential
           r_m - float - membrane resistance 
    Output: float - minimum current
    '''
    return((v_th-e_l)/r_m)