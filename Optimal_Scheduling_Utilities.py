import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from numpy.linalg import pinv
from numpy.linalg import matrix_rank
from scipy.optimize import fmin
from scipy.stats import norm

def tp(x):
    # vector transpose
    return(np.array(np.matrix(x).transpose()))


def sort_sams(S):
    # get mean values
    samval = np.nanmean(S, axis=0)
    # get time indices
    samind = np.argwhere(~np.isnan(samval))
    # extract nans
    samval = samval[~np.isnan(samval)]
    return samind, samval


def get_index(x, y):
    # for each value in x
    # get index in y where x == y
    x = np.round(x, 2)
    index = []
    for i in x:
        for ind, val in enumerate(y):
            if i == val:
                index.append(ind)
                continue
    return index


def get_kld(p1, p2):
    kld = np.sum(np.multiply(p1, np.log2(np.true_divide(p1, p2))))
    return kld


def get_sam_lik(samind, samval, rng, errsurf):
    # get sample indices
    samvalind = get_index(samval, rng)
    # get likelihoods
    L = []
    for i in range(len(samvalind)):
        l = errsurf[samvalind[i], samind[i]]
        if np.isscalar(l):
            L.append(l)
        else:
            L.append(l[0])
    return L


def fit_gp_kernel(type, samval, x_dif, x_sig, sd_prior):

    def fit_kernel(type, samval, x_dif, x_sig):
        # initial parameter values for exp2 model
        p_init = [2, 8]
        x_sigmat = np.diag(x_sig)
        xopt = fmin(fit_k, x0=np.log(p_init), args=(type, samval, x_dif, x_sigmat), full_output=True, disp=False)
        p_opt = np.exp(xopt[0])
        nll = xopt[1]
        return p_opt, nll

    def fit_k(params, type, samval, x_dif, x_sigmat):
        # params are constrained
        params = np.exp(params)
        sig_1 = params[0]
        sig_2 = params[1]
        # covariance model
        K = sig_1 * np.exp(-(x_dif)**2 / (2*sig_2**2))
        # add measurement noise
        Kpsig = K + x_sigmat

        # det of cov matrix
        det_K = det(Kpsig)

        # check whether matrix is singular,
        # if K has rank 0, exit
        if matrix_rank(Kpsig) == 0:
           return 10**6
        # if det == 0 or matrix rank is deficient, set det = 1 and get pseudo inv of K
        elif (det_K == 0) | (matrix_rank(Kpsig) != samval.size):
            det_K = 1
            K_inv = pinv(Kpsig)
        #  otherwise, get inverse of K
        else:
           K_inv = inv(Kpsig)

        # get negative log likelihood
        ll = (-(tp(samval) * K_inv * samval) \
              -(np.log(det_K)) \
              -samval.size * np.log(2 * np.pi)) / 2
        nll = np.sum(ll)  #log likelihood
        return nll

    params = np.array([])
    nlls = []
    for sig in x_sig:
        sigs = sig**2 * np.ones((samval.size, 1))[0]
        p_opt, nll = fit_kernel(type, samval, x_dif, sigs)
        p_opt = tp(tp(p_opt))
        if params.size == 0:
            params = p_opt
        else:
            params = np.vstack((params, p_opt))
        nlls.append(nll)
    # weight log likelihood according to noise prior
    nlls = np.multiply(nlls, sd_prior)
    # get minLL
    minnll = np.nanmin(nlls)
    nllind = np.nanargmin(nlls)
    n_param = x_sig[nllind]
    k_params = params[nllind, :]
    return k_params, n_param, minnll


def get_kernel_params(D):
    # get sample values and time indices
    samind, samval = sort_sams(D['S'])
    # subtract the mean for GPR
    samval_n = samval - np.mean(samval)
    # get indices of observed and unobserved timepoints
    x_known = D['dmn'][samind]
    x_unknown = D['dmn']
    # get difference mat for the covariance model
    x, y = np.meshgrid(x_known, x_known)
    x_dif = np.abs(x - y)
    # possible noise parameter values
    x_sig = D['sdObs']
    # fit kernel
    k_params, n_param, nll = fit_gp_kernel(D['type'], samval_n, x_dif, x_sig, D['sdPrior'])

    return k_params, n_param


def gp_estimate(type, Kparams, x_known, x_unknown, x_sig, y_known):


    def calc_gp_kernel(type, Kparams, x_known, x_unknown, x_sig):
        # contruct the covariance matrix
        x_sigmat = np.eye(x_sig.size) * x_sig**2
        x, y = np.meshgrid(x_known, x_known)
        x_dif = x - y
        x_dif = np.vstack((x_dif, x_unknown - x_known))
        x_dif = np.hstack((x_dif, tp(np.concatenate((x_dif[-1, :], [0])))))
        # the kernel
        K = Kparams[0] * np.exp(-(x_dif)**2 / (2*Kparams[1]**2))
        # add measurement noise
        K = K + x_sigmat
        return K

    # subtract the mean
    mu_y = np.mean(y_known)
    y_known = tp(y_known - mu_y)
    # calculate the covariance kernel and parse it
    K = calc_gp_kernel(type, Kparams, x_known, x_unknown, x_sig)
    K_known = K[:-1, :-1]
    K_unknown = tp(tp(K[-1, :-1]))
    # function estimate and variance for this point
    y_est = mu_y + np.dot(np.inner(K_unknown, inv(K_known)), y_known)
    y_var = K[-1, -1] - np.dot(np.inner(K_unknown, inv(K_known)), tp(K_unknown))
    return y_est, y_var


def get_gp_estimate(type, dmn, k_params, n_param, samind, samval):
    # get new GP estimate
    x_sig_est = n_param * np.ones((samind.size + 1, 1))
    x_unknown = dmn
    x_known = dmn[samind]
    y_known = samval
    y_est = []
    y_var = []
    for x in x_unknown:
        yest, yvar = gp_estimate(type, k_params, x_known, x, x_sig_est, y_known)
        y_est.append(yest[0][0])
        y_var.append(yvar[0][0])

    return y_est, y_var

def get_error_surf(rng, y_est, y_var):
    # get error surf of the new estimate
    err = np.empty((rng.size, len(y_est)))
    for idx, yval in enumerate(y_est):
        err[:, idx] = norm.pdf(rng, yval, np.sqrt(y_var[idx]))
        err[:, idx] = err[:, idx] / np.sum(err[:, idx])
    return err

def get_sample(D, iter):

    # get sample at optimal time
    D['S'][iter, D['topt'][-1]] = D['data'][D['topt'][-1]]

    # update th sample index
    dsind = np.append(D['Sind'], D['topt'][-1])
    D['Sind'] = np.unique(np.sort(dsind))

    # sort the samples
    samind, samval = sort_sams(D['S'])
    samind = tp(samind)[0]

    # get covariance kernel and noise params
    k_params = D['Kparams']
    n_param = D['sdEst']

    # get new GP estimate
    y_est, y_var = get_gp_estimate(D['type'], D['dmn'], k_params, n_param, samind, samval)

    # update function estimate
    D['Fhyp'][:, iter] = y_est

    # get error surf of the new estimate
    err = get_error_surf(D['rng'], y_est, y_var)

    # update error surf
    D['Eobs'][iter] = err

    # get sample likleihood
    L = get_sam_lik(samind, samval, D['rng'], err)
    D['LLhyp'] = np.append(D['LLhyp'], L)

    # get new optimal sample time
    D = get_optimal_sample_time(D, iter)

    return D

def get_optimal_sample_time(D, iter):

    # return if all data have been sampled
    if D['Sind'].size == D['data'].size:
        print('Done!')
        return D

    # get current function value
    #f_est = D['Fhyp'][:, iter] # expected value...
    f_est = D['data']           # ...or true value

    # allocate a variable for the KLD
    kld = np.empty((1, f_est.size))[0]
    kld.fill(np.nan)

    # index unmeasured sample times
    newsamind = np.setdiff1d(np.arange(0, f_est.size), D['Sind'])

    for ind in newsamind:
        # reset sample matrix
        S = np.copy(D['S'])

        # get new sample from current function estimate and add to current samples
        newsam = f_est[ind]
        S[iter+1, ind] = newsam

        # sort the samples
        samind, samval = sort_sams(S)
        samind = tp(samind)[0]

        # get covariance kernel params
        k_params = D['Kparams']
        n_param = D['sdEst']

        # get new GP estimate
        y_est, y_var = get_gp_estimate(D['type'], D['dmn'], k_params, n_param, samind, samval)

        # get error surf of the new estimate
        err = get_error_surf(D['rng'], y_est, y_var)

        # get KLD of the error surfs
        kld[ind] = get_kld(D['Eobs'][iter], err)

    # optimal sample time
    D['topt'].append(np.nanargmax(kld))

    # get sample at optimal time and repeat
    iter = iter + 1
    D = get_sample(D, iter)

    return D




