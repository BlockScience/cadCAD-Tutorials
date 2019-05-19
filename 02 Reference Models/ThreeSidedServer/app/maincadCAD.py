from flask import Flask,request,jsonify,Response,render_template
from functools import wraps
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from cadCAD.configuration import append_configs
from cadCAD.configuration.utils import proc_trigger, bound_norm_random, ep_time_step, config_sim
import pandas as pd
import cadCadFunctions as c2F
from tabulate import tabulate
# The following imports NEED to be in the exact order
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from simulations.validation import config1
from cadCAD import configs

# TODO: Fix api,

app = Flask(__name__)
app.secret_key = 'mysecretkey'

# For downloading files
@app.route('/data/<path:filename>')
def download(filename):
	try:
		return send_from_directory("data/",filename)

	except Exception as e:
		return str(e)

def run_cadCAD(conversion_fee,gamma,final_supply,tampw):
    '''function to run cadCAD simulation'''
    seeds = {
    }

    sim_config = config_sim({
        'T': range(260), #number of discrete iterations in each experiement
        'N': 5, #number of times the simulation will be run (Monte Carlo runs)
    })

    # parameters (later add tuning)
    eta = .065
    tampw = 100000 #A_max
    alpha = .9
    beta = 1.0
    flat = 500
    a = 1000.0
    b = 100.0
    c = 1.0
    d = 1.0
    conversion_rate_gain=1
    release_rate = .01 #percent of remaining
    theta = .6
    #estimatable parameters
    roi_threshold = .2
    attrition_rate = .5
    roi_gain = .025
    #this function is a parameter of this policy which determines diminishing value of more labor
    base_value = 1000.0
    buffer_runway = 3.0
    reserve_threshold = .25
    min_budget_release = 0
    #forgetting rate parameter for smoothin
    rho = .1 #1 means you forget all past times, must be >0
    #forgetting rate
    rho2 = rho #same as for the other smoothed averages

    # external states
    #cell defines the model function for the tx_volume generator process
    def tx_volume_generator(_g, step, sL, s, _input):
        y = 'tx_volume'
        x = s['tx_volume']*(1+2*eta*np.random.rand()*(1-s['tx_volume']/tampw))
        return (y, x)

    def cost_of_production_generator(_g, step, sL, s, _input):
        y = 'cost_of_production'
        x = alpha*s['cost_of_production']+beta*np.random.rand()
        return (y, x)

    #log quadratic overhead model; parameters
    #this model is general and designed to have paramters fit from real data
    #to fit stochastic processes, import using pandas and use scikit-learn for parameter estimation
    def overhead_cost_generator(_g, step, sL, s, _input):
        #unit fiat
        y = 'overhead_cost'
        q = a+b*s['tx_volume']+c*s['volume_of_production']+d*s['tx_volume']*s['volume_of_production']
        x = flat+np.log(q)
        return (y, x)

    # define the time deltas for the discrete increments in the model
    ts_format = '%Y-%m-%d'
    t_delta = timedelta(days=7, minutes=0, seconds=0)
    def time_model(_g, step, sL, s, _input):
        y = 'time'
        x = ep_time_step(s, dt_str=s['time'], fromat_str=ts_format, _timedelta=t_delta)
        return (y, x)


    # Behaviors ~ Decision Polices
    # Behavoirs (two types: controlled and uncontrolled)
    # user behavoir is uncontrolled (estimated or modeled),
    #governance policies are controlled (implemented or enforced)
    #governance decision ~ system policy for token/fiat unit of value conversion
    #static policy
    def conversion_policy(_g, step, sL, s):
        ncr = conversion_rate_gain*s['smooth_avg_token_reserve']/s['smooth_avg_fiat_reserve']
        return {'new_conversion_rate': ncr}

    #governance decision ~ determines the conditions or schedule of new tokens minted
    def minting_policy(_g, step, sL, s):
        mint =  (final_supply-s['token_supply'])*release_rate
        return {'mint': mint}

    #these are uncontrollerd choices of users in the provider consumer
    def consumer_choice(_g, step, sL, s):
        #fiat paid by consumers
        #note: balance of consumption vol covered in tokens (computed later)

        #simple heuristic ~ the fraction of token payment is proportion to free supply share
        free_supply = s['token_supply']-s['token_reserve']
        share_of_free_supply = free_supply/ s['token_supply']

        txi_fiat= (1.0-share_of_free_supply)*s['tx_volume']
        return {'txi_fiat': txi_fiat}
        #these are uncontrollerd choices of users in the provider role
        #simple heuristic ~ convex combination of 2 policies; parameterized by theta
        #part1: fixed base rate ~ theta percent of service providers ALWAYS use fiat
        #part2: remaining fraction is determined by the size of the platform community "economy"
        #recall that both of the below states have 'fiat' as their unit, ratio is unitless
        #simplifying math note:
        #production_consumption_ratio = s['volume_of_production']*s['cost_of_production']/s['tx_volume']
        #production_consumption_ratio * s['tx_volume'] = s['volume_of_production']*s['cost_of_production']
    def provider_choice(_g, step, sL, s):
        #fiat claimed by providers
        #note: balance of provided vol covered in tokens (computed later)
        txo_fiat = theta*s['tx_volume']+ (1-theta)*gamma*s['volume_of_production']*s['cost_of_production']
        return {'txo_fiat': txo_fiat}

    #governance decision ~ system policy for compensating producers
    #consider transaction volume, labor committed and token reserve and supply to determine payout
    def marginal_utility_function(x):
        #this is how much the platform value a total amount of production (in fiat)
        return base_value+np.sqrt(x)

    def producer_compensation_policy(_g, step, sL, s):
        tokens_paid = s['conversion_rate']*marginal_utility_function(s['volume_of_production'])
        return {'tokens_paid': tokens_paid}

    #these are uncontrollerd choices of users in the producer role
    def producer_choice(_g, step, sL, s):
        #ROI heuristic
        # add or remove resources based on deviation from threshold
        if s['producer_roi_estimate'] < 0:
            delta_labor = s['volume_of_production']*(attrition_rate-1.0)
        else:
            ratio = s['producer_roi_estimate']/roi_threshold
            delta_labor = roi_gain*s['volume_of_production']*(ratio-1.0)

        return {'delta_labor': delta_labor}

    #governance decision ~ system policy for budgeting to cover overhead costs
    #note that this is a naive Heuristic control policy based on
    # Strategies described by human operators
    # there exist formal alternatives using stochastic optimal control
    def budgeting_policy(_g, step, sL, s):
        #define an estimate of future overhead
        proj_overhead = s['overhead_cost'] #simple naive

        #simple threshold based conditional logic
        if s['operational_budget']< buffer_runway*proj_overhead:
            target_release = buffer_runway*proj_overhead-s['operational_budget']
            if  s['fiat_reserve']-target_release > reserve_threshold*s['fiat_reserve']:
                budget_released =  target_release
            else:
                budget_released = (1.0-reserve_threshold)*s['fiat_reserve']
        else:
            budget_released = min_budget_release*s['fiat_reserve']

        return {'budget_released': budget_released}

    # Mechanisms
    def update_conversion_rate(_g, step, sL, s, _input):
        y = 'conversion_rate'
        x = _input['new_conversion_rate']
        return (y, x)

    #smooth averaging signals
    def update_smooth_avg_fiat_reserve(_g, step, sL, s, _input):
        y = 'smooth_avg_fiat_reserve'
        x = s['fiat_reserve']*rho+s['smooth_avg_fiat_reserve']*(1-rho)
        return (y, x)

    def update_smooth_avg_token_reserve(_g, step, sL, s, _input):
        y = 'smooth_avg_token_reserve'
        x = s['token_reserve']*rho+s['smooth_avg_token_reserve']*(1-rho)
        return (y, x)

    #minting process mints into the reserve
    def mint1(_g, step, sL, s, _input):
        y = 'token_supply'
        x = s['token_supply'] + _input['mint']
        return (y, x)

    def mint2(_g, step, sL, s, _input):
        y = 'token_reserve'
        x = s['token_reserve'] + _input['mint']
        return (y, x)

    def commit_delta_production(_g, step, sL, s, _input):
        y = 'volume_of_production'
        x = s['volume_of_production']+_input['delta_labor']
        return (y, x)

    def compensate_production(_g, step, sL, s, _input):
        y = 'token_reserve'
        x = s['token_reserve']-_input['tokens_paid']
        return (y, x)

    def update_producer_roi_estimate(_g, step, sL, s, _input):
        revenue = _input['tokens_paid']/s['conversion_rate']
        cost = s['cost_of_production']*s['volume_of_production']
        spot_ROI_estimate =   (revenue-cost)/cost
        y = 'producer_roi_estimate'
        x = rho2*spot_ROI_estimate + s['producer_roi_estimate']*(1.0-rho2)
        return (y, x)

    def capture_consumer_payments1(_g, step, sL, s, _input):
        #fiat inbound
        y = 'fiat_reserve'
        x = s['fiat_reserve']+_input['txi_fiat']
        return (y, x)

    def capture_consumer_payments2(_g, step, sL, s, _input):
        #tokens inbound
        y = 'token_reserve'
        fiat_eq = s['tx_volume']-_input['txi_fiat']
        x = s['token_reserve']+s['conversion_rate']*fiat_eq*(1.0+conversion_fee)
        return (y, x)

    platform_fee = 0.075

    def compensate_providers1(_g, step, sL, s, _input):
        #fiat outbound
        y = 'fiat_reserve'
        x = s['fiat_reserve']-_input['txo_fiat']*(1.0-platform_fee)
        return (y, x)

    def compensate_providers2(_g, step, sL, s, _input):
        #tokens outbound
        y = 'token_reserve'
        fiat_eq = s['tx_volume']-_input['txo_fiat']
        x = s['token_reserve']-s['conversion_rate']*fiat_eq*(1.0-platform_fee-conversion_fee)
        return (y, x)

    def release_funds(_g, step, sL, s, _input):
        #tokens outbound
        y = 'fiat_reserve'
        x = s['fiat_reserve'] - _input['budget_released']
        return (y, x)

    def update_budget(_g, step, sL, s, _input):
        #tokens outbound
        y = 'operational_budget'
        x = s['operational_budget'] + _input['budget_released']
        return (y, x)

    # Initial States
    genesis_states = {
        'fiat_reserve': float(25000),#unit: fiat
        'overhead_cost': float(500), #unit: fiat
        'operational_budget': float(25000), #unit: fiat
        'token_reserve': float(25000),#unit: tok
        'token_supply': float(25000),#unit: tok
        'tx_volume': float(1000), #unit: fiat
        'conversion_rate': float(1), #unit: tok/fiat
        'cost_of_production': float(10), #unit: fiat/labor
        'volume_of_production': float(20), #unit: labor
        'producer_roi_estimate': float(1.1), #unitless //signal for informing policies
        'smooth_avg_fiat_reserve': float(25000), #unit: fiat //signal for informing policies
        'smooth_avg_token_reserve': float(25000), #unit: token //signal for informing policies
        'time': '2019-01-01'
    }

    exogenous_states = {
        'time': time_model
    }

    env_processes = {
    }

    #build mechanism dictionary to "wire up the circuit"
    mechanisms = [
        #mechstep 0
        #'evolve':
        {
            'policies':
            {
            },
            'variables':
            {
                'cost_of_production': cost_of_production_generator,
                'tx_volume': tx_volume_generator,
                'overhead_cost': overhead_cost_generator
            }
        },
        #'producers_act':
        {
            'policies':
            {
                'action': producer_choice
            },
            'variables':
            {
                'volume_of_production': commit_delta_production,
                'fiat_reserve': compensate_providers1,
                'token_reserve': compensate_providers2
            }
        },

        #'consumers_act':
        {
            'policies':
            {
                'action': consumer_choice
            },
            'variables':
            {
                'fiat_reserve': capture_consumer_payments1,
                'token_reserve': capture_consumer_payments2
            }
        },
        #'pay_producers':
        {
            'policies':
            {
                'action': producer_compensation_policy
            },
            'variables':
            {
                'token_reserve': compensate_production,
                'producer_roi_estimate': update_producer_roi_estimate
            }
        },

        #'budgeting':
        {
            'policies':
            {
                'action': budgeting_policy
            },
            'variables':
            {
                'fiat_reserve': release_funds,
                'operational_budget': update_budget
            }
        },
        #'minting':
        {
            'policies':
            {
                'action': minting_policy
            },
            'variables':
            {
                'token_reserve': mint1,
                'token_supply': mint2
            }
        },
        #'signal_updates':
        {
            'policies':
            {
            },
            'variables':
            {
                'smooth_avg_fiat_reserve': update_smooth_avg_fiat_reserve,
                'smooth_avg_token_reserve':update_smooth_avg_token_reserve
            }
        },
        #'price_controller':
        {
            'policies':
            {
                'action': conversion_policy
            },
            'variables':
            {
                'conversion_rate': update_conversion_rate
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

    exec_mode = ExecutionMode()


    first_config = configs # only contains config1
    single_proc_ctx = ExecutionContext(context=exec_mode.single_proc)
    run = Executor(exec_context=single_proc_ctx, configs=first_config)

    raw_result, tensor_field = run.main()
    result = pd.DataFrame(raw_result)

    return result

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/health")
def health():
    return 'Healthy'

@app.route('/result', methods=['POST'])
def result():
    '''
    Route for UI
    '''

    # Get the data from the index page
    conversion_fee=request.form['conversion_fee']
    gamma=request.form['gamma']
    final_supply=request.form['final_supply']
    tampw=request.form['tampw']

    # TODO: Make function
    # remove commas and convert to floats
    conversion_fee = float(conversion_fee.replace(',', ''))
    gamma = float(gamma.replace(',', ''))
    final_supply = float(final_supply.replace(',', ''))
    tampw = float(tampw.replace(',', ''))

    df = run_cadCAD(conversion_fee,gamma,final_supply,tampw)

    mean_df,median_df,std_df,min_df = c2F.aggregate_runs(df,'timestep')
    mean_df["price"] = rdf.conversion_rate.apply(lambda r: 1/r)
    fileName = 'simulationsResults'
    mean_df.to_csv('data/simulationsResults.csv')

    fig, axes = plt.subplots(nrows=3, ncols=2,figsize=(20, 20),sharey=False)
    fig.tight_layout(pad=8)

    fig.suptitle("Three Sided Model Monte Carlo 5 Run Mean Simulation Results", fontsize=16)
    mean_df.plot(x = 'timestep', y=['fiat_reserve','token_reserve'],title='Fiat and Token Reserves',logy=False,ax=axes[0,0],grid=True)
    mean_df.plot(x = 'timestep', y=['fiat_reserve','token_reserve','price'],title='Dual Token Model -Implied Asset Price Fiat per Token',logy=False,ax=axes[0,1],grid=True)
    mean_df.plot(x = 'timestep', y='price',title='Price',logy=False,ax=axes[1,0],grid=True)
    mean_df.plot(x = 'timestep', y='volume_of_production',title='Volume of production',logy=False,ax=axes[1,1],grid=True)
    mean_df.plot(x = 'timestep', y='cost_of_production',title='Cost of production',logy=False,ax=axes[2,0],grid=True)
    mean_df.plot(x = 'timestep', y='producer_roi_estimate',title='Producer ROI Estimate',logy=False,ax=axes[2,1],grid=True)

    fig.savefig('static/images/simulationResults.jpeg')
    return render_template('result.html', fileName=fileName)

# TODO: Update example
@app.route('/run', methods=['POST'])
def result_API():
    '''
    Example API call:
        Curl:

        curl -X POST -H "Content-Type: application/json" -d '{"conversion_fee": ".03",
        "gamma": ".1",
        "final_supply": "1000000",
        "tampw": "100000"
        }' http://localhost:8000/run


        Python:
        import requests

        headers = {
            'Content-Type': 'application/json',
            }

        data = '{"conversion_fee": ".03","gamma": ".1","final_supply": "1000000","tampw": "100000"}'

        response = requests.post('http://localhost:8000/run', headers=headers,data=data)
        response.json()

    '''

    conversion_fee = request.get_json()['conversion_fee']
    gamma = request.get_json()['gamma']
    final_supply = request.get_json()['final_supply']
    tampw = request.get_json()['tampw']
    # remove commas and convert to floats
    conversion_fee = float(conversion_fee.replace(',', ''))
    gamma = float(gamma.replace(',', ''))
    final_supply = float(final_supply.replace(',', ''))
    tampw = float(tampw.replace(',', ''))
    Result = run_cadCAD(conversion_fee,gamma,final_supply,tampw)
    Result = Result.to_json()
    return Result

@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('error.html'), 404

@app.errorhandler(405)
def page_not_found2(e):
    # note that we set the 405 status explicitly
    return render_template('error.html'), 405

@app.errorhandler(500)
def page_not_found3(e):
    # note that we set the 500 status explicitly
    return render_template('error.html'), 500

if __name__ == "__main__":
     app.run(host='localhost', port=8000,debug=True)
