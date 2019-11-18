import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats
import scipy.optimize

def GMM(N, prior, mu, Sigma):
    dist = [0]
    dist.extend(np.cumsum(prior))
    u = np.random.rand(N)
    L = np.zeros((N))
    X = np.zeros((N, 2))
    for i in range(len(prior)):
        indices = np.where(np.logical_and(u >= dist[i], u < dist[i+1]) == False)
        L[indices] = i * np.ones(len(indices))
        for k in indices[0]:
            X[k,:] = np.random.multivariate_normal(mu[i,:], Sigma[:,:,i])
    return X, L

def fisherLDA(X,L):
    X_1 = X[np.where(L == 1),:]
    X_0 = X[np.where(L == 0),:]
    muhat_1 = np.mean(X_1, axis=1)
    muhat_0 = np.mean(X_0, axis=1)
    Sigmahat_1 = np.cov(np.transpose(X_1[0]))
    Sigmahat_0 = np.cov(np.transpose(X_0[0]))
    S_b = (muhat_1 - muhat_0) * np.transpose(muhat_1 - muhat_0)
    S_w = Sigmahat_1 + Sigmahat_0
    D, V = np.linalg.eig(np.dot(np.linalg.inv(S_w),S_b))
    w = V[:, np.argmax(D)]
    Y = np.dot(w,np.transpose(X))
    w = (np.sign(np.mean(Y[np.where(L == 1)]) - np.mean(Y[np.where(L == 0)])))  * w
    Y = (np.sign(np.mean(Y[np.where(L == 1)]) - np.mean(Y[np.where(L == 0)])))  * Y
    mu_y1 = np.mean(Y[np.where(L == 1)])
    mu_y0 = np.mean(Y[np.where(L == 0)])
    std_y1 = np.std(Y[np.where(L == 1)])
    std_y0 = np.std(Y[np.where(L == 0)])
    Pr_1 = len(X_1) / (len(X_1)+len(X_0))
    Pr_0 = len(X_0) / (len(X_1)+len(X_0))
    func = lambda y: (std_y0*Pr_1/std_y1*Pr_0)*math.e**(-((y-mu_y1)**2/(2*std_y1**2))+((y-mu_y0)**2/(2*std_y0**2)))-1
    b = -scipy.optimize.fsolve(func, 0)
    D_ind = np.where(Y+b >= 0)
    D = np.zeros(len(L))
    D[D_ind] = 1 * np.ones(len(D_ind))
    return Y, D, w, b

def logisticModelEstimator(X, L, w, b):
    X_p = X[np.where(L == 1),:]
    X_n = X[np.where(L == 0),:]
    np.reshape(w,[1,2])
    theta0 = np.hstack((w,b))
    Lfunc = lambda theta: sum(np.log(1+math.e**(np.dot(theta[:2],np.transpose(X_p[0]))+theta[2]))) - sum(np.log(1-1/(1+math.e**(np.dot(theta[:2],np.transpose(X_n[0]))+theta[2]))))
    argmin = scipy.optimize.minimize(Lfunc, theta0, method='Nelder-Mead',options={'disp':False})
    w_e = argmin.x[:2]
    b_e = argmin.x[2]
    y = 1/(1+math.e**(np.dot(w_e,np.transpose(X))+b_e))
    D_indices = np.where(y >= 0.5)
    D = np.zeros(len(L))
    D[D_indices] = len(D_indices)
    return w_e, b_e, D

def MAPestimate(X, mu, Sigma, Pr):
    rv_0 = scipy.stats.multivariate_normal(np.reshape(mu[0],(2)), Sigma[:,:,0])
    rv_1 = scipy.stats.multivariate_normal(np.reshape(mu[1],(2)), Sigma[:,:,1])
    D = np.zeros(len(X))
    for i in range(len(X)):
        postiori_0 = rv_0.pdf(X[i,:])*Pr[0]
        postiori_1 = rv_1.pdf(X[i,:])*Pr[1]
        if postiori_0 >= postiori_1 :
            D[i] = 0
        else:
            D[i] = 1
    return D

mu_true = np.array([[1,1],[-1,1]])
Sigma_true = np.zeros([2,2,2])
Sigma_true[:,:,0] = 0.4*np.array([[1,0.5],
                                  [0.5,1]])
Sigma_true[:,:,1] = 0.4*np.array([[1,-0.5],
                                  [-0.5,1]])
N = 999
prior = np.array([0.3,0.7])

X,L = GMM(N, prior, mu_true, Sigma_true)
plt.figure(figsize = (10,10))
plt.scatter(X[np.where(L == 0),0],X[np.where(L == 0),1], marker='x', label='Lable-')
plt.scatter(X[np.where(L == 1),0],X[np.where(L == 1),1], marker='.', label='Lable+')
plt.legend(fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Samples with true label', fontsize=20)

Y, D, w, b = fisherLDA(X, L)
errors = np.count_nonzero(D!= L)
plt.figure(figsize=(10,10))
plt.scatter(X[np.where(np.logical_and(D == 0,L == 0)),0],X[np.where(np.logical_and(D == 0,L == 0)),1], c='g', marker='.', label='Decision -, Label -')
plt.scatter(X[np.where(np.logical_and(D == 1,L == 0)),0],X[np.where(np.logical_and(D == 1,L == 0)),1], c='r', marker='x', label='Decision -, Label +')
plt.scatter(X[np.where(np.logical_and(D == 0,L == 1)),0],X[np.where(np.logical_and(D == 0,L == 1)),1], c='r', marker='.', label='Decision +, Label -')
plt.scatter(X[np.where(np.logical_and(D == 1,L == 1)),0],X[np.where(np.logical_and(D == 1,L == 1)),1], c='g', marker='x', label='Decision +, Label +')
plt.legend(fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('FisherLDA')
print('errors= %i'%(errors))

w, b, D_logistic = logisticModelEstimator(X,L,w, b)
errors = np.count_nonzero(D_logistic != L)
plt.figure(figsize=(10,10))
plt.scatter(X[np.where(np.logical_and(D_logistic == 0,L == 0)),0],X[np.where(np.logical_and(D_logistic == 0,L == 0)),1], c='r', marker='.', label='D = -, L = -')
plt.scatter(X[np.where(np.logical_and(D_logistic == 1,L == 0)),0],X[np.where(np.logical_and(D_logistic == 1,L == 0)),1], c='r', marker='+', label='D = +, L = -')
plt.scatter(X[np.where(np.logical_and(D_logistic == 0,L == 1)),0],X[np.where(np.logical_and(D_logistic == 0,L == 1)),1], c='b', marker='.', label='D = -, L = +')
plt.scatter(X[np.where(np.logical_and(D_logistic == 1,L == 1)),0],X[np.where(np.logical_and(D_logistic == 1,L == 1)),1], c='b', marker='+', label='D = +, L = +')
plt.legend(fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Logistic estimat', fontsize=20)
print('w=[ %f , %f ]'%(w[0],w[1]))
print('errors= %i'%(errors))

D_MAP = MAPestimate(X, mu_true, Sigma_true, prior)
errors = np.count_nonzero(D_MAP != L)
plt.figure(figsize=(10,10))
plt.scatter(X[np.where(np.logical_and(D_MAP == 0,L == 0)),0],X[np.where(np.logical_and(D_MAP == 0,L == 0)),1], c='r', marker='.', label='D = -, L = -')
plt.scatter(X[np.where(np.logical_and(D_MAP == 1,L == 0)),0],X[np.where(np.logical_and(D_MAP == 1,L == 0)),1], c='r', marker='+', label='D = +, L = -')
plt.scatter(X[np.where(np.logical_and(D_MAP == 0,L == 1)),0],X[np.where(np.logical_and(D_MAP == 0,L == 1)),1], c='b', marker='.', label='D = -, L = +')
plt.scatter(X[np.where(np.logical_and(D_MAP == 1,L == 1)),0],X[np.where(np.logical_and(D_MAP == 1,L == 1)),1], c='b', marker='+', label='D = +, L = +')
plt.legend(fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('MAP estimate', fontsize=20)
print(w)
print('errors= %i'%(errors))