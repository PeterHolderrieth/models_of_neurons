import numpy as np 
import matplotlib.pyplot as plt 

def compute_alpha_beta_n(v_t):
    '''
    Input: v_t -float - membrane potential at current time step
    Output: alpha_n,beta_n - floats -  see Dayan and Abbott (2005) equation (5.22)
    '''
    alpha_n=0.01*(v_t+55)/(1-np.exp(-0.1*(v_t+55)))
    beta_n=0.125*np.exp(-0.0125*(v_t+65))

    return(alpha_n,beta_n)

def compute_alpha_beta_m(v_t):
    '''
    Input: v_t -float - membrane potential at current time step
    Output: alpha_m,beta_m - floats - see Dayan and Abbott (2005) equation (5.24)
    
    '''
    alpha_m=0.1*(v_t+40)/(1-np.exp(-0.1*(v_t+40)))
    beta_m=4*np.exp(-0.0556*(v_t+65))
    
    return(alpha_m,beta_m)
    
def compute_alpha_beta_h(v_t):
    '''
    Input: v_t -float - membrane potential at current time step
    Output: alpha_h,beta_h - floats -  see Dayan and Abbott (2005) equation (5.24)
    '''
    alpha_h=0.07*np.exp(-0.05*(v_t+65))
    beta_h=1/(np.exp(-0.1*(v_t+35))+1)

    return(alpha_h,beta_h)

def dn_dt(v_t,n_t):
    '''
    Computes equation (7) in Hodgkin and Huxley (1952) - we use the notation from there.
    Input: v_t -float - membrane potential at current time step
           n_t -float - share n of potassium channel activating molecules at current time step t
    Output: float - derivative of n at current time step
    '''
    alpha_n,beta_n=compute_alpha_beta_n(v_t)
    return (alpha_n*(1-n_t)-beta_n*n_t)

def dm_dt(v_t,m_t):
    '''
    Computes equation (15) in Hodgkin and Huxley (1952) - we use the notation from there.
    Input: v_t -float - membrane potential at current time step
           m_t -float - share m of sodium channel activating molecules at current time step t
    Output: float - derivative of m at current time step
    '''
    alpha_m,beta_m=compute_alpha_beta_m(v_t)
    return (alpha_m*(1-m_t)-beta_m*m_t)

def dh_dt(v_t,h_t):
    '''
    Computes equation (16) in Hodgkin and Huxley (1952) - we use the notation from there.
    Input: v_t -float - membrane potential at current time step
           h_t -float - share h of potassium channel inactivating molecules at current time step t
    Output: float - derivative of h at current time step
    '''
    alpha_h,beta_h=compute_alpha_beta_h(v_t)
    return (alpha_h*(1-h_t)-beta_h*h_t)

def dv_dt(v_t,i_ind,c_m,g_k_const,g_na_const,g_l_const,v_na,v_k,v_l,n_t,m_t,h_t):
    '''
    Input:
        v_t - float - membrane potential at time point 
        n_t,m_t,h_t - float - share of potassium/sodium channel activating/inactivating molecules (see 
                        Hodgkin and Huxley (1952) equation (7),(15), and (16))
        i_ind - float - induced current at time point
        c_m - float - membrane capacitance
        g_k_const,g_na_const,g_l_const - constants for conductances (for potassium (K) and sodium (Na),
                                        these constants do descripe the slope the the conductance depending 
                                        on time and v_t)
        v_na,v_k,v_l - float - 
    Output:
        float - derivative of membrane potential v at current time step
    '''
    #Compute 
    pot_term=g_k_const*(n_t**4)*(v_t-v_k)
    sod_term=g_na_const*(m_t**3)*h_t*(v_t-v_na)
    leak_term=g_l_const*(v_t-v_l)

    #Compute derivative:
    dv_dt=(i_ind-pot_term-sod_term-leak_term)/c_m
    
    return(dv_dt)


def step_hodgkin_huxley(dt,v_t,n_t,m_t,h_t,i_ind,c_m,g_k_const,g_na_const,g_l_const,v_na,v_k,v_l):
    '''
    Input: dt - float - time difference to simulate 
           for remaining parameters, see function dv_dt 
    Output: floats - new parameters for v_t,n_t,m_t,h_t
    '''
    #Compute derivatives over four dynamic parameters:
    v_t_slope=dv_dt(v_t,i_ind,c_m,g_k_const,g_na_const,g_l_const,v_na,v_k,v_l,n_t,m_t,h_t)
    n_t_slope=dn_dt(v_t,n_t)
    m_t_slope=dm_dt(v_t,m_t)
    h_t_slope=dh_dt(v_t,h_t)



    #Use Euler method to compute next step:
    v_t_plus_1=v_t+dt*v_t_slope
    n_t_plus_1=n_t+dt*n_t_slope
    m_t_plus_1=m_t+dt*m_t_slope
    h_t_plus_1=h_t+dt*h_t_slope

    return(v_t_plus_1,n_t_plus_1,m_t_plus_1,h_t_plus_1)


def sim_hodgkin_huxley(n_steps,dt,v_0,n_0,m_0,h_0,i_ind,c_m,g_k_const,g_na_const,g_l_const,v_na,v_k,v_l):
    '''
    Input: n_steps - int - number of time steps to simulate
           dt - float - time difference to simulate 
           v_0,n_0,m_0,h_0 - float - initial conditions
           i_ind - either: 1. float: then we have a constant current given by this value  
                           2. function: giving the induced current at time t
           for remaining parameters, see function step_hodgkin_huxley
    Output: lists of times, membrane potential v_t, and parameters n_t,m_t,h_t (see Hodgkin and Huxley (1952) 
                                                                                equation (7),(15), and (16) for details)
    '''
    if isinstance(i_ind,float):
        def i_at_time_t(t):
            return(i_ind)
    else: 
        print("Use original function.")
        i_at_time_t=i_ind

    #Initialize empty lists:
    v_t_list=[v_0]
    n_t_list=[n_0]
    m_t_list=[m_0]
    h_t_list=[h_0]

    i_t_list=[i_at_time_t(0.)]
    t_list=[0.]

    for it in range(n_steps):
        #Simulate next step:
        v_t_new,n_t_new,m_t_new,h_t_new=step_hodgkin_huxley(dt,v_t_list[-1],n_t_list[-1],m_t_list[-1],h_t_list[-1],\
                                i_t_list[-1],c_m,g_k_const,g_na_const,g_l_const,v_na,v_k,v_l)
        
        #Update times and currents:
        t_new=t_list[-1]+dt
        t_list.append(t_new)
        i_t_list.append(i_at_time_t(t_new))

        #Update voltage and ion channel states:
        v_t_list.append(v_t_new)
        n_t_list.append(n_t_new)
        m_t_list.append(m_t_new)
        h_t_list.append(h_t_new)

    return(t_list,v_t_list,n_t_list,m_t_list,h_t_list,i_t_list)



