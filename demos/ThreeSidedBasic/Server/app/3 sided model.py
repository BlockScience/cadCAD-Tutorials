# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from cadCAD import configs
import cadCadFunctions as c2F
from decimal import Decimal
from datetime import timedelta
from cadCAD.configuration import append_configs
from cadCAD.configuration.utils import proc_trigger, bound_norm_random, ep_time_step, config_sim

exec_mode = ExecutionMode()


seeds = {
    # 'z': np.random.RandomState(1),
    'a': np.random.RandomState(2)
}

sim_config = config_sim({
    'T': range(36), #number of discrete iterations in each experiement
    'N': 1, #number of times the simulation will be run (Monte Carlo runs)
})

# parameters (later add tuning)
eta = .33 # for tx_volume_generator
tampw = 100000 # transactions limit
alpha = .5 # for data acquisition cost generator
beta = .2 # for data acquisition cost generator
costDecrease = .015 # decrease in cost
price = priceWidget = 1 # sales price
vcRoundFunding = 100000
overHeadCosts = 5000

# external states
#cell defines the model function for the tx_volume generator stochastic process
def tx_volume_generator(_g, step, sL, s, _input):
    y = 'tx_volume'
    x = s['tx_volume']*(1+2*eta*np.random.rand()*(1-s['tx_volume']/tampw))
    return (y, x)

def product_cost_generator(_g, step, sL, s, _input):
    y = 'product_cost'
    x = alpha*s['product_cost']+beta*np.random.rand() - costDecrease
    return (y, x)

def investors_generator(_g, step, sL, s, _input):
    y = 'seed_money'
    if s['timestep'] == 1:
        x = s['seed_money'] + vcRoundFunding
    elif s['timestep'] == 10:
        x = s['seed_money'] + vcRoundFunding
#     elif s['timestep'] == 50:
#         x = s['seed_money'] + 100000
    else:
        x = s['seed_money'] + 0
    return (y, x)

def update_overhead_costs(_g, step, sL, s, _input):
    # Create step function for updating overhead costs
    y = 'overhead_cost'
    if s['timestep']%15 == 0:
        x = s['overhead_cost'] + overHeadCosts
    else:
        x = s['overhead_cost'] + 0
    return (y, x)

def R_and_D(_g, step, sL, s, _input):
    y = 'R&D'
    if s['timestep']%17 == 0:
        x = s['R&D'] + 1000
    else:
        x = s['R&D'] + 0
    return (y, x)

# define the time deltas for the discrete increments in the model
ts_format = '%Y-%m-%d %H:%M:%S'
t_delta = timedelta(days=30, minutes=0, seconds=0)
def time_model(_g, step, sL, s, _input):
    y = 'time'
    x = ep_time_step(s, dt_str=s['time'], fromat_str=ts_format, _timedelta=t_delta)
    return (y, x)


# Behaviors
def inflow(_g, step, sL, s):
    # Receive money from relevant parties
    return {'Receive': 1}

def outflow(_g, step, sL, s):
    # Pay relevant parties
    return {'Pay': 1}

def investors(_g, step, sL, s):
    # Pay relevant parties
    if s['timestep'] == 1:
        return {'Invest': 1}
    elif s['timestep'] == 10:
        return {'Invest': 1}
    else:
        return {'Invest': 0}

def metrics(_g, step, sL, s):
    return {'Stat': 1}


# Mechanisms
def receive_fiat_from_consumers(_g, step, sL, s, _input):
    y = 'fiat_reserve'
    if _input['Receive'] == 1:
        x = s['fiat_reserve'] + s['tx_volume'] * price
    else:
        x = s['fiat_reserve']
    return (y, x)

def receive_revenue_from_consumers(_g, step, sL, s, _input):
    y = 'revenue'
    if _input['Receive'] == 1:
        x = s['tx_volume'] * price
    else:
        x = s['revenue']
    return (y, x)


def receive_fiat_from_investors(_g, step, sL, s, _input):
    y = 'fiat_reserve'
    if _input['Invest'] == 1:
        x = s['fiat_reserve'] + s['seed_money']
    else:
        x = s['fiat_reserve']
    return (y, x)

def pay_fiat_to_producers(_g, step, sL, s, _input):
    y = 'fiat_reserve'
    if _input['Pay'] == 1:
        x = s['fiat_reserve'] -  (s['product_cost'] * s['tx_volume'])
        x = s['fiat_reserve']
    return (y, x)

def pay_investment_expenses(_g, step, sL, s, _input):
    y = 'fiat_reserve'
    if _input['Pay'] == 1:
        x = s['fiat_reserve'] - s['R&D']
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

# Metrics

def COGS(_g, step, sL, s, _input):
    y = 'COGS'
    if _input['Stat'] == 1:
        x = (s['product_cost'] * s['tx_volume'])
    else:
        x = s['COGS']
    return (y, x)

# Initial States
genesis_states = {
    'tx_volume': float(100), #unit: fiat
    'product_cost': float(.3), #unit: fiat cost
    'revenue': float(0), # revenue per month
    'fiat_reserve': float(0),#unit: fiat
    'overhead_cost': float(100), #unit: fiat per month
    'seed_money': float(0),
    'R&D': float(0), #per month
    'COGS': float(0), #per month
    'time': '2018-01-01 00:00:00'
}

exogenous_states = {
    'time': time_model
}

env_processes = {
}


#build mechanism dictionary to "wire up the circuit"
mechanisms = [
    #'exogenous':
    {
        'policies':
        {
        },
        'variables':
        {
            'tx_volume': tx_volume_generator,
            'product_cost': product_cost_generator,
            'seed_money': investors_generator,
            'overhead_cost': update_overhead_costs,
            'R&D': R_and_D
        }

    },
    #'fiat inflow':
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

    #'investors':
    {
        'policies':
        {
            'action': investors
        },
        'variables':
        {
            'seed_money': receive_fiat_from_investors
        }
    },
    #'fiat outflow':
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
    #'metrics':
    {
        'policies':
        {
            'action': metrics
        },
        'variables':
        {
            'COGS': COGS
        }
    },

]

append_configs(
    sim_configs=sim_config,
    initial_state=genesis_states,
    seeds=seeds,
    raw_exogenous_states=exogenous_states,
    env_processes=env_processes,
    partial_state_update_blocks=mechanisms
)



# Run Cad^2

first_config = configs
single_proc_ctx = ExecutionContext(context=exec_mode.single_proc)
run = Executor(exec_context=single_proc_ctx, configs=first_config)

raw_result, tensor_field = run.main()
df = pd.DataFrame(raw_result)
df = df.round(2)

mean_df,median_df,std_df,min_df = c2F.aggregate_runs(df,'timestep')

# Calculate the metrics
mean_df['GrossMargin'] = df['revenue'] - df['COGS']
mean_df['EBITDA'] = df['revenue'] - df['COGS'] - df['overhead_cost'] - df['R&D']


mean_df.head()
mean_df.tail()


fig, axes = plt.subplots(nrows=5, ncols=2,figsize=(15, 15),sharey=False)
fig.tight_layout(pad=8)

fig.suptitle("Monte Carlo 100 Run Mean Simulation Results", fontsize=16)

mean_df.plot(x = 'timestep',y='fiat_reserve',title='Fiat Reserve Balance',logy=False,ax=axes[0,0],grid=True)
mean_df.plot(x = 'timestep',y='seed_money',title='Seed Money Inflow Balance',logy=False,ax=axes[0,1],grid=True)
mean_df.plot(x = 'timestep',y='overhead_cost',title='Overhead Cost per month',logy=False,ax=axes[1,0],grid=True)
mean_df.plot(x = 'timestep',y='COGS',title='COGS',logy=False,ax=axes[1,1],grid=True)
mean_df.plot(x = 'timestep',y='revenue',title='Revenue per month',logy=False,ax=axes[2,0],grid=True)
mean_df.plot(x = 'timestep',y='tx_volume',title='Transaction Volume per month',logy=False,ax=axes[2,1],grid=True)
mean_df.plot(x = 'timestep',y='product_cost',title='Product Cost per transaction',logy=False,ax=axes[3,0],grid=True)
mean_df.plot(x = 'timestep',y='R&D',title='R&D per month',logy=False,ax=axes[3,1],grid=True)
#mean_df.plot(x = 'timestep',y='RevenueQuarterlyPercentageChange',title='Revenue Percentage Change per quarter',logy=False,ax=axes[3,1],grid=True)
mean_df.plot(x = 'timestep',y='GrossMargin',title='Gross Margin per month',logy=False,ax=axes[4,0],grid=True)
mean_df.plot(x = 'timestep',y='EBITDA',title='EBITDA per month',logy=False,ax=axes[4,1],grid=True)
fig.savefig('images/Results.eps', format='eps', dpi=300)
