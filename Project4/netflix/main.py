import numpy as np
import kmeans
import common
import naive_em
import em



K = 1

#X = np.loadtxt("toy_data.txt")

    # run k means for seed=[0,1,2,3,4] 

# for seed in range(5):
#     init_mixture, init_post = common.init(X=X, K=K, seed=seed) # initialize mean, variance, mixing proportion(weights), post(delta(j|i))
#     final_mixture, final_post, final_cost = kmeans.run(X=X, mixture=init_mixture, post=init_post) # run K-means
#     common.plot(X= X, mixture= final_mixture, post= final_post, title= f"K={K}, seed={seed}, Cost={final_cost}") # graph results

    # run naive_em for seed=[0,1,2,3,4]

# seed =4
# for K in range(1,5):
#     init_mixture, _ = common.init(X=X, K=K, seed=seed) # initialize mean, variance, mixing proportion(weights)
#     final_mixture, final_post, final_log_LL = naive_em.run(X=X, mixture=init_mixture, post=None)
#     common.plot(X= X, mixture= final_mixture, post= final_post, title= f"K={K}, seed={seed}, log_LL={final_log_LL}") # graph results
#     print(f'for K={K} , bic={common.bic(X , final_mixture, final_log_LL)}')

    # run em


X = np.loadtxt("netflix_incomplete.txt")


# from scipy.stats import multivariate_normal
# # from  scipy.special import logsumexp 
# from time import time
# init_mixture, _ = common.init(X=X, K=K, seed=0) 

# n, _ = X.shape
# K = init_mixture.mu.shape[0]


# N =np.zeros((n,K))
# for u in range(1):
#     for j in range(K):
        
#         if all(X[u]==0):
#             N[u,j]= np.log(1/K) # 1/K
#         else:
#             inicio = time()
#             N[u,j] = (multivariate_normal.logpdf(x= X[u, X[u]!=0] , mean= init_mixture.mu[j, X[u]!=0] , cov= init_mixture.var[j]*np.identity(len(X[u, X[u]!=0]))) )
#             fin = time()
#         print(fin-inicio) 


# search for the best seed={0,1,2,3,4}
max_final_log_LL = None
for seed in range(5):
    init_mixture, _ = common.init(X=X, K=K, seed=seed) # initialize mean, variance, mixing proportion(weights)
    final_mixture , final_post , final_log_LL = em.run(X=X, mixture=init_mixture, post= None)
    print(f'seed={seed}, log_LL={final_log_LL}') 
    if (max_final_log_LL is None) or (final_log_LL>max_final_log_LL):
        max_final_log_LL = final_log_LL
        best_final_mixture = final_mixture

print(f'the best mixture is: {best_final_mixture} \n')
X_pred = em.fill_matrix(X=X, mixture=best_final_mixture)
X_gold = np.loadtxt('netflix_complete.txt')
rmse = common.rmse(X_gold, X_pred)
print(f'RMSE  for K={K}, seed={seed}:{rmse}')

# TODO find the way to make the e-step faster



