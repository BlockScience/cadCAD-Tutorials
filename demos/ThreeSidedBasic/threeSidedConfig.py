# import libraries
from decimal import Decimal
import numpy as np
from datetime import timedelta
from cadCAD.configuration import append_configs
from cadCAD.configuration.utils import proc_trigger, bound_norm_random, ep_time_step, config_sim

seeds = {
    # 'z': np.random.RandomState(1),
    # 'a': np.random.RandomState(2)
}

sim_config = config_sim({
    'T': range(360), #number of discrete iterations in each experiement
    'N': 1, #number of times the simulation will be run (Monte Carlo runs)
})

# parameters (later add tuning)
eta = .04 # for tx_volume_generator
tampw = 10000 # transactions limit
alpha = .5 # for data acquisition cost
beta = .2 # for data acquisition cost
cost = 2.5

# external states
#cell defines the model function for the tx_volume generator stochastic process
def tx_volume_generator(_g, step, sL, s, _input):
    y = 'tx_volume'
    x = s['tx_volume']*(1+2*eta*np.random.rand()*(1-s['tx_volume']/tampw))
    return (y, x)

def product_cost_generator(_g, step, sL, s, _input):
    y = 'product_cost'
    x = alpha*s['product_cost']+beta*np.random.rand()
    return (y, x)

def investors_generator(_g, step, sL, s, _input):
    y = 'investors_money'
    x = s['investors_money']*(1 -.03)**step #* (np.random.gamma(1,1))
    return (y, x)

def update_overhead_costs(_g, step, sL, s, _input):
    y = 'overhead_cost'
    x = s['overhead_cost'] + 10
    return (y, x)

# define the time deltas for the discrete increments in the model
ts_format = '%Y-%m-%d %H:%M:%S'
t_delta = timedelta(days=1, minutes=0, seconds=0)
def time_model(_g, step, sL, s, _input):
    y = 'time'
    x = ep_time_step(s, dt_str=s['time'], fromat_str=ts_format, _timedelta=t_delta)
    return (y, x)


# Behaviors
def inflow(_g, step, sL, s):
    # Receive money from relevant parties
    return {'Recieve': 1}

def outflow(_g, step, sL, s):
    # Pay relevant parties
    return {'Pay': 1}

def investors(_g, step, sL, s):
    # Pay relevant parties
    return {'Invest': 1}

# Mechanisms
def receive_fiat_from_consumers(_g, step, sL, s, _input):
    y = 'fiat_reserve'
    if _input['Recieve'] == 1:
        x = s['fiat_reserve'] + s['tx_volume'] * cost
    else:
        x = s['fiat_reserve']
    return (y, x)

def receive_revenue_from_consumers(_g, step, sL, s, _input):
    y = 'revenue'
    if _input['Recieve'] == 1:
        x = s['revenue'] + s['tx_volume'] * cost
    else:
        x = s['revenue']
    return (y, x)


def receive_fiat_from_investors(_g, step, sL, s, _input):
    y = 'fiat_reserve'
    if _input['Invest'] == 1:
        x = s['fiat_reserve'] + s['investors_money']
    else:
        x = s['fiat_reserve']
    return (y, x)

def pay_fiat_to_producers(_g, step, sL, s, _input):
    y = 'fiat_reserve'
    if _input['Pay'] == 1:
        x = s['fiat_reserve'] -  (s['product_cost'] * s['tx_volume'] * .5) # assuming only 50% of data is purchased each transaction
    else:
        x = s['fiat_reserve']
    return (y, x)

def pay_investment_expenses(_g, step, sL, s, _input):
    y = 'fiat_reserve'
    if _input['Pay'] == 1:
        x = s['fiat_reserve'] - 1000
    else:
        x = s['fiat_reserve']
    return (y, x)

def pay_overhead_costs(_g, step, sL, s, _input):
    y = 'fiat_reserve'
    if _input['Pay'] == 1:
        x = s['fiat_reserve'] - s['overhead_cost']
    else:
        x = s['fiat_reserve']
    return (y, x)


# Initial States
genesis_states = {
    'tx_volume': float(10), #unit: fiat
    'product_cost': float(.3), #unit: fiat cost
    'revenue': float(0), # cost revenue
    'fiat_reserve': float(0),#unit: fiat
    'overhead_cost': float(10000), #unit: fiat
    'investors_money': float(10000),
    'time': '2018-01-01 00:00:00'
}

exogenous_states = {
    'time': time_model
}

env_processes = {
}

#build mechanism dictionary to "wire up the circuit"
mechanisms = {
    #mechstep 0
    'evolve':
    {
        'policies':
        {
        },
        'variables':
        {
            'tx_volume': tx_volume_generator,
            'product_cost': product_cost_generator,
            'investors_money': investors_generator,
	    'overhead_cost': update_overhead_costs
        }

    },
    'fiat inflow':
    {
        'policies':
        {
            'action': inflow
        },
        'variables':
        {
            'fiat_reserve': receive_fiat_from_consumers,
            'revenue': receive_revenue_from_consumers
        }
    },

    'investors':
    {
        'policies':
        {
            'action': investors
        },
        'variables':
        {
            'investors_money': receive_fiat_from_investors
        }
    },
    'fiat outflow':
    {
        'policies':
        {
            'action': outflow
        },
        'variables':
        {
            'fiat_reserve': pay_fiat_to_producers,
            'fiat_reserve': pay_investment_expenses,
            'fiat_reserve': pay_overhead_costs
        }
    },

}


append_configs(
    sim_configs=sim_config,
    initial_state=genesis_states,
    seeds=seeds,
    raw_exogenous_states=exogenous_states,
    env_processes=env_processes,
    partial_state_update_blocks=mechanisms
)
