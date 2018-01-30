import pandas as pd
import numpy as np
from scipy.stats import norm
from Optimal_Scheduling_Utilities import *

# load pooled data
db0 = pd.read_csv('~/Dropbox/OptimalSampling/AntiVEGF/antivegf_python/AntiVEGF_aulcsf_pooled_nonNorm_PreProc_2.csv',
                  header=None)
# convert to array
db0_mat = db0.values

# ------------
# Enter params
# ------------
# get optimal schedule for Niter days
Niter = 61
# sample rate (gives 8 initial samples)
sam_rate = 9
# enter kernel and noise params from full fit
kparams = [1.876, 9.445]
nparam = [0.084]
# create a variable to count iterations
iter = 0

# crop the dataframe
db0_mat = db0_mat[:, :Niter]
# get median sensitivities
med_cs = np.nanmedian(db0_mat, axis=0)

# create dictionary 'D'
D = {'data': [],
     'dmn': [],
     'rng': np.arange(0,2,.01), # range of the GPR fits
     'iter': Niter,
     'LL': [],
     'LLhyp': [],
     'topt': [],
     'Fhyp': [],
     'entFhyp': [],
     'Eobs': {},
     'S': [],
     'Sind': [],
     'sdmax': [],
     'sdObs': [],
     'sdEst': [],
     'sdPrior': [],
     'type': 'exp2', # define the covariance model
     'Kparams': []
     }

# get intial samples at equal intervals
D['data'] = med_cs
D['dmn'] = np.arange(D['data'].size)  # should this start at 1?
D['S'] = np.empty((D['iter'], D['dmn'].size))
D['S'][:] = np.nan
D['Sind'] = np.arange(0, D['dmn'].size+1, sam_rate)
# sample the last point
D['Sind'] = np.append(D['Sind'], D['dmn'].size-1)
# now get sample values
samples = D['data'][D['Sind']]
# insert samples into S matrix
for ind, val in enumerate(D['Sind']):
    D['S'][iter, val] = samples[ind]

# create an array to hold the estimates
D['Fhyp'] = np.empty((D['dmn'].size, Niter))
D['Fhyp'][:] = np.nan

# store the model params
D['Kparams'] = kparams
D['sdEst'] = nparam

# get first GPR estimate
y_est, y_var = get_gp_estimate(D['type'], D['dmn'], D['Kparams'], D['sdEst'], D['Sind'], D['S'][iter, D['Sind']])

# store the estimate
for idx, val in enumerate(y_est):
    D['Fhyp'][idx, iter] = val

# error surf of the estimate
err = np.empty((D['rng'].size, len(y_est)))
for i in range(len(y_est)):
    err[:, i] = norm.pdf(D['rng'].T, y_est[i], y_var[i])
    # handle err == 0
    err[:, i] = [j if j > 0 else 10 ** -6 for j in err[:, i]]
    err[:, i] = err[:, i]/np.sum(err[:, i])
D['Eobs'][iter] = err

# get likelihood of the samples
samind, samval = sort_sams(D['S'])
samval = np.round(samval, 2)
L = get_sam_lik(samind, samval, D['rng'], err)

# Generate the optimal schedule
D = get_optimal_sample_time(D, iter)

