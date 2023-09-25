"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
import scipy.stats as ss



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d= X.shape
    K= mixture.mu.shape[0]
    
    pjn= mixture.p * np.transpose(np.array([ss.multivariate_normal.pdf(x=X , mean= mixture.mu[j], cov=mixture.var[j]*np.identity(d)) for j in range(K)]))
    post =  pjn / np.resize(np.sum(pjn, axis=1), (n,1))

    log_LL= np.sum(np.log(np.resize(np.sum(pjn, axis=1), (n,1))))

    return post , log_LL
    


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d= X.shape
    K= post.shape[1]
    
    hn = np.sum(post, axis=0)
    p = hn / n
    mu = np.sum((np.transpose(post[:,:,np.newaxis], axes=(1,2,0)) * np.transpose(X[:,:,np.newaxis], axes=(2,1,0))), axis=2)/ np.resize(hn, (K,1))
    var = np.sum(np.sum((np.transpose(X[:,:, np.newaxis], axes=(2,1,0)) - mu[:,:,np.newaxis])**2, axis=1) * np.transpose(post) , axis=1) / (d*hn)
    
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
        mixture = mstep(X, post)

    return mixture, post, log_LL
