"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K = mixture.mu.shape[0]
    
    N =np.zeros((n,K))
    for u in range(n):
        for j in range(K):
            if all(X[u]==0):
                N[u,j]= np.log(1/K) # 1/K
            else:
                N[u,j] = (multivariate_normal.logpdf(x= X[u, X[u]!=0] , mean= mixture.mu[j, X[u]!=0] , cov= mixture.var[j]*np.identity(len(X[u, X[u]!=0]))) )
            
    print('Hola')
    f_ui =  np.log(mixture.p + 1e-16) + N  # N is already the ln of N                  # (n,K)
    l_ju = f_ui - np.resize(logsumexp(f_ui, axis=1) , (n,1))  
    post = np.exp(l_ju)

    log_LL = np.sum(logsumexp(f_ui[np.sum(X, axis=1)!=0], axis=1))

    return post , log_LL



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, _ = X.shape
    K = mixture.mu.shape[0]

    p = np.sum(post, axis=0) / n

    f = np.vectorize(lambda x_ul: 1 if x_ul!=0 else x_ul) # a function to convert X to a matrix delta(l,Cu)
    den_mu = np.sum(post[:,:,np.newaxis] * np.transpose(f(X)[:,:, np.newaxis], axes=(0,2,1)), axis=0)
    mu = np.sum(post[:,:, np.newaxis] * np.transpose(X[:,:,np.newaxis], axes=(0,2,1)), axis=0) / den_mu
    mu[den_mu<1] = mixture.mu[den_mu<1] # returns the mean to the last value if there are not a significant amount of ratings for the movie in the cluster
    
    X_r= np.repeat(X[:,:,np.newaxis] , K , axis=2)
    mu_r = np.repeat(np.transpose(mu[:,:,np.newaxis], axes=(2,1,0)) , n , axis=0 )
    X_r[X_r!=0] = X_r[X_r!=0]- mu_r[X_r!=0]
    var = np.sum(np.sum(X_r ** 2, axis=1) * post , axis=0) / np.sum(np.resize(np.sum(f(X), axis=1) , (n,1)) * post, axis=0)
    var[var<min_variance] = min_variance

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_LL = None
    log_LL = None
    
    while ((old_log_LL is None) or (log_LL - old_log_LL > pow(10,-6)*np.abs(log_LL))):
        old_log_LL = log_LL
        post, log_LL = estep(X, mixture)
        mixture = mstep(X, post, mixture)


    return mixture, post, log_LL


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, _ = X.shape
    K = mixture.mu.shape[0]
    
    N = []
    for u in range(n):
        N_u = []
        for j in range(K):
            if np.sum(X[u])==0:
                N_u.append(np.log(1/K)) # 1/K
            else:
                N_u.append(multivariate_normal.logpdf(x= X[u, X[u]!=0] , mean= mixture.mu[j, X[u]!=0] , cov= mixture.var[j]*np.identity(len(X[u, X[u]!=0]))) )
        N.append(N_u)

    N = np.array(N) # in fact ln(N)
    
    f_ui =  np.log(mixture.p + 1e-16) + N  # N is already the ln of N                  # (n,K)
    l_ju = f_ui - np.resize(logsumexp(f_ui, axis=1) , (n,1))  
    post = np.exp(l_ju)

    mu_pon = np.sum(post[:,:, np.newaxis] * np.transpose(mixture.mu[:,:, np.newaxis], axes=(2,0,1)), axis=1)
    X_pred = X.copy()
    X_pred[X==0] = mu_pon[X==0]

    return X_pred
