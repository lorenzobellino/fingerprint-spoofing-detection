import numpy as np
import scipy as sc

from Utils.utils import vcol, vrow


# def polynomial_kernel_with_bias(x1, x2,xi,ci):
#      d=2
#      return ((np.dot(x1.T, x2) + ci) ** d) + xi

# def rbf_kernel_with_bias(x1, x2,xi, gamma):
#      return np.exp(-gamma * np.sum((x1 - x2) ** 2)) + xi


# class SVMClass:
    
#     def __init__(self,K,C,piT,mode,ci):
#         self.K = K
#         self.C = C
#         self.piT = piT
#         self.mode = mode
#         self.alfa = []
#         self.ci  = ci               #if you use mode = "rbf" this is used as GAMMA
#         self.DTR = []
#         self.LTR = []
        
        
#         if mode == "poly":
#             self.kernel_func = polynomial_kernel_with_bias
#         elif mode == "rbf":
#             self.kernel_func = rbf_kernel_with_bias
#         else:
#             print("Unknown mode")
#             self.kernel_func = polynomial_kernel_with_bias
            
    
#     def compute_kernel_score(self,alpha, DTR, L, kernel_func, x,xi,ci):
#          Z = np.zeros(L.shape)
#          Z[L == 1] = 1
#          Z[L == 0] = -1
#          score = 0
#          for i in range(alpha.shape[0]):
#              if alpha[i] > 0:
#                  score += alpha[i]*Z[i]* kernel_func(DTR[:, i],x,xi,ci)
#          return score
    
    
#     def compute_lagrangian_wrapper(self,H):
#         def compute_lagrangian(alpha):
#             alpha = alpha.reshape(-1, 1)
#             Ld_alpha = 0.5 * alpha.T @ H @ alpha - np.sum(alpha)
#             gradient = H @ alpha - 1
#             return Ld_alpha.item(), gradient.flatten()
#         return compute_lagrangian
    
#     # def accuracy(predicted_labels, original_labels):
#     #     total_samples = len(predicted_labels)
#     #     correct = (predicted_labels == original_labels).sum()
#     #     return (correct / total_samples) * 100
    
#     # def error_rate(predicted_labels, original_labels):
#     #     return 100 - accuracy(predicted_labels, original_labels)
    
#     def compute_H(self,DTR,LTR,kernel_func,xi,ci):
#          n_samples = DTR.shape[1]
#          Hc = np.zeros((n_samples, n_samples))
#          Z = np.where(LTR == 0, -1, 1)
#          for i in range(n_samples):
#              for j in range(n_samples):
#                  Hc[i, j] = Z[i]*Z[j]* kernel_func(DTR[:, i], DTR[:, j],xi,ci)
#          return Hc

    
#     def train(self,DTR,LTR):
#         # print("dentro train")
#         self.DTR = DTR
#         self.LTR = LTR
        
#         nf = DTR[:,LTR==0].shape[1]
#         nt = DTR[:,LTR==1].shape[1]
#         emp_prior_f = nf/ DTR.shape[1]
#         emp_prior_t = nt/ DTR.shape[1]
#         Cf = self.C * self.piT / emp_prior_f
#         Ct = self.C * self.piT / emp_prior_t
       
#         xi = self.K * self.K
#         H_=self.compute_H(DTR,LTR,self.kernel_func,xi,self.ci)
#         compute_lag=self.compute_lagrangian_wrapper(H_)
#         bound_list=[(-1,-1)]*DTR.shape[1]
        
#         for i in range(DTR.shape[1]):
#             if LTR[i] == 0:
#                 bound_list[i] = (0,Cf)
#             else:
#                 bound_list[i] = (0,Ct)
        
        
#         #factr=1 requires more iteration but returns more accurate results
#         (alfa,f,d)=sc.optimize.fmin_l_bfgs_b(compute_lag,x0=np.zeros(LTR.size),approx_grad=False,factr=1.0,bounds=bound_list)
#         self.alfa = alfa
        
#     def compute_scores(self,DTE):
#         # print("dentro comp_score")
#         score=np.array([self.compute_kernel_score(self.alfa,self.DTR,self.LTR,self.kernel_func,x,self.K*self.K,self.ci) for x in DTE.T])
#         return score

def SVM_linear(DTR, LTR, DTE, C, K):
    D_ext = np.vstack((DTR, K * np.ones(DTR.shape[1])))

    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = np.dot(D_ext.T,D_ext)
    H = vcol(Z) * vrow(Z) * H
    
    def JDual(alpha):
        Ha = np.dot(H,vcol(alpha))
        aHa = np.dot(vrow(alpha),Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    def JPrimal(w):
        S = np.dot(vrow(w), D_ext)
        loss = np.maximum(np.zeros(S.shape), 1-Z*S).sum()
        return 0.5 * np.linalg.norm(w)**2 + C * loss
    
    alphaStar , _x, _y = sc.optimize.fmin_l_bfgs_b(
    LDual,
    np.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = np.dot(D_ext, vcol(alphaStar) * vcol(Z))
    
    DTEEXT = np.vstack((DTE,np.array([K for i in range(DTE.shape[1])])))
    Scores = np.dot(wStar.T,DTEEXT)
    
    return Scores.ravel()

def SVM_Poly(DTR, LTR, DTE, C, K, d, c):
    
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = ((np.dot(DTR.T,DTR) + c) ** d) + K * K
    H = vcol(Z) * vrow(Z) * H
    
    def JDual(alpha):
        Ha = np.dot(H,vcol(alpha))
        aHa = np.dot(vrow(alpha),Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    
    alphaStar , _x, _y = sc.optimize.fmin_l_bfgs_b(
    LDual,
    np.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = np.dot(DTR, vcol(alphaStar) * vcol(Z))
    kernel = ((np.dot(DTR.T,DTE) + c) ** d) + K * K 
    scores = np.sum(np.dot(alphaStar * vrow(Z), kernel), axis=0)
    return scores.ravel()  

def SVM_RBF(DTR, LTR, DTE, C, K, gamma):
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    H = np.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            H[i, j] = np.exp(-gamma * (np.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + K * K
    H = vcol(Z) * vrow(Z) * H
    
    def JDual(alpha):
        Ha = np.dot(H,vcol(alpha))
        aHa = np.dot(vrow(alpha),Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + np.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    

    alphaStar , _x, _y = sc.optimize.fmin_l_bfgs_b(
    LDual,
    np.zeros(DTR.shape[1]),
    bounds = [(0,C)] * DTR.shape[1],
    factr = 1.0,
    maxiter = 100000,
    maxfun = 100000,
    )
    
    wStar = np.dot(DTR, vcol(alphaStar) * vcol(Z))
    
    kernel = np.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kernel[i, j] = np.exp(-gamma * (np.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K
    
    scores = np.sum(np.dot(alphaStar * vrow(Z), kernel), axis=0)
    return scores.ravel()
   