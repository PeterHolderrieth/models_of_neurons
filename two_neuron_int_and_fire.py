import numpy as np 
def prob_firing(p_max,tau_s,t,t_f):
    '''
    Input: 
        p_max - float>=0,<=1 - maximum probability of a neuron to fire
        tau_s - float>0 - time constant of synapse
        t_f - float - time of last spike
        t - float - time to evaluate probability
    Output: 
        float - probability of synapse to open
    '''
    diff=-(t-t_f)/tau_s
    return(p_max*np.exp(diff))

def step_two_neuron_int_and_fire(dt,v1_0,v2_0,p_max,tau_s,t,t_f_1,t_f_2,e_l,r_m,g_s,e_s,i_e,tau_m,v_reset,v_th):
    '''
    Function to simulate two neurons connected by a synapse from neuron 1 to neuron 2 and a synapse 
    from neuron 2 to neuron 1.
    Input: 
        dt - float - time step to simulate
        v1_0,v2_0 - float - current voltage of neuron 1 and 2 
        p_max,tau_s,t,t_f_1,t_f_2 - parameters for probability of prob_firing
        e_l - float - resting potential
        r_m - float - membrane resistance
        g_s - float - synaptic conductance
        e_s - float - synaptic resting potential
        i_e - float -input current to the two neurons
        tau_m - float - membrane time constant
        v_reset - float - reset potential after spikes
        v_th - float - threshold potential for spikes
    Output: 
        step_1,step_2 - float - next potential of neuron 1,2
        t_f_1,t_f_2 - float - time of last spike of neuron 1,2
        reset_1,reset_2 - int in [0,1] - indicate whether voltage has been reset
    '''
    #Compute the probability that the synapse 2-->1 (resp. 1-->2) opens
    prob_2to1=prob_firing(p_max,tau_s,t,t_f_2)
    prob_1to2=prob_firing(p_max,tau_s,t,t_f_1)

    #Compute the derivative of the potential of each neuron
    diff_1=(e_l-v1_0-r_m*prob_2to1*g_s*(v1_0-e_s)+r_m*i_e)/tau_m
    diff_2=(e_l-v2_0-r_m*prob_1to2*g_s*(v2_0-e_s)+r_m*i_e)/tau_m

    #Compute next step (potential) by the Euler method
    step_1=v1_0+dt*diff_1
    step_2=v2_0+dt*diff_2

    #Reset if voltage exceed threshold:
    if step_1>v_th:
        t_f_1=t+dt
        step_1=v_reset
        reset_1=1
    else: 
        reset_1=0

    if step_2>v_th:
        t_f_2=t+dt
        step_2=v_reset
        reset_2=1
    else: 
        reset_2=0
    
    return(step_1,step_2,t_f_1,t_f_2,reset_1,reset_2)


def sim_two_neuron_int_and_fire(n_steps,dt,v1_0,v2_0,p_max,tau_s,e_l,r_m,g_s,e_s,i_e,tau_m,v_reset,v_th,t_f_1_init=0.,t_f_2_init=0.):
    '''
    Function to simulate two neurons over time.
    Input:
        n_steps - int - number of steps simulate 
        dt - float - time step
        v1_0,v2_0 - float - initial condition
        for remainining parameters, see step_two_neuron_int_and_fire above
    Output: 
        time_list - list of simulate times
        v1_list,v2_list - list potentials of neuron 1 and 2
        n_act_pots_1,n_act_pots_2 - number of action potentials of neuron 1 and 2 
    '''
    #Initialize time and voltage lists
    time_list=[0.]
    v1_list=[v1_0]
    v2_list=[v2_0]

    #Initialize number of action potentials fired and spike times:
    n_act_pots_1=0
    n_act_pots_2=0

    #Initialize time of last spike:
    t_f_1=t_f_1_init
    t_f_2=t_f_2_init

    for it in range(n_steps):

        #Go one time step of simulation:
        v1_new,v2_new,t_f_1,t_f_2,reset_1,reset_2=step_two_neuron_int_and_fire(dt,v1_list[-1],v2_list[-1],p_max,tau_s,time_list[-1],t_f_1,t_f_2,e_l,r_m,g_s,e_s,i_e,tau_m,v_reset,v_th)

        #Update time, voltage and number of action potentials:
        time_list.append(time_list[-1]+dt)

        
        v1_list.append(v1_new)
        v2_list.append(v2_new)
        
        n_act_pots_1+=reset_1
        n_act_pots_2+=reset_2

    return(time_list,v1_list,v2_list,n_act_pots_1,n_act_pots_2)
