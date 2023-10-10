import numpy as np
import scipy as sc
import scipy.optimize as opt
from Utils.utils import load, split_db_2to1,vcol,vrow,plot_logreg
from matplotlib import pyplot as plt
from GaussianClassifiers.main import KFCV
from LogisticRegression.models import logRegClass,quadLogRegClass

# class logRegClass:
#     def __init__(self,l,piT):
#         self.l = l
#         self.piT = piT
        
        
#     def logreg_obj(self,v):
#         loss = 0
#         loss_c0 = 0
#         loss_c1 = 0
#         w,b = v[0:-1],v[-1]
#         w = vcol(w)
#         n = self.DTR.shape[1]
        
#         regularization = (self.l / 2) * np.sum(w ** 2) 
        
#         for i in range(n):
            
#             if (self.LTR[i:i+1] == 1):
#                 zi = 1
#                 loss_c1 += np.logaddexp(0,-zi * (np.dot(w.T,self.DTR[:,i:i+1]) + b))
#             else:
#                 zi=-1
#                 loss_c0 += np.logaddexp(0,-zi * (np.dot(w.T,self.DTR[:,i:i+1]) + b))
        
#         J = regularization + (self.piT / self.nT) * loss_c1 + (1-self.piT)/self.nF * loss_c0
        
#         return J
   

    
#     def train(self,DTR,LTR):
#         self.DTR  = DTR
#         self.LTR = LTR
#         x0 = np.zeros(DTR.shape[0] + 1)
        
#         self.nT = len(np.where(LTR == 1)[0])
#         self.nF = len(np.where(LTR == 0)[0])
        
#         params,f_min,_ = sc.optimize.fmin_l_bfgs_b(self.logreg_obj, x0,approx_grad=True)
        
#         self.b = params[-1]
        
#         self.w = np.array(params[0:-1])
        
#         self.S = []
        
        
#         return self.b,self.w
    
#     def compute_scores(self,DTE):
#         S = self.S
#         for i in range(DTE.shape[1]):
#             x = DTE[:,i:i+1]
#             x = np.array(x)
#             x = x.reshape((x.shape[0],1))
#             self.S.append(np.dot(self.w.T,x) + self.b)
        
#         S = [1 if x > 0 else 0 for x in S] #I transform in 1 all the pos values and in 0 all the negative ones
        
#         llr = np.dot(self.w.T, DTE) + self.b
        
#         return llr
 
def binary_logaristic_regression(DTR,LTR,args,logger):
    logger.info(f"Binary Regression")
    point = (0.5,1,10)
    lambda_ = np.logspace(-4, 2, num=7)
    # lambda_ = np.logspace(-2, 1, num=4)
    for piT in [0.1,0.5,0.9]:
        logger.debug(f"piT = {piT}")
        lr_pca = dict()
        for l in lambda_:
            logger.debug(f"l = {l}")
            for pca in [6,7,0]:
                logger.debug(f"pca = {pca}")
                logObj = logRegClass(l,piT)
                args.pca = pca
                min_DCF, scores, labels = KFCV(DTR, LTR,logObj,point,args,logger)
                logger.info(f"Log Reg min_DCF con K = {args.k} , pca = {pca}, l = {l} , piT = {piT}: {min_DCF} ")
    
                try:
                    lr_pca[f"{pca}"].append(min_DCF)
                except:
                    lr_pca[f"{pca}"] = [min_DCF]
        
        logger.info(f"plotting")
        plot_logreg(lambda_,piT,lr_pca,f"LR_DCF_l_k{args.k}_z-{args.znorm}_piT_{piT}_v2.png",args)

def quadratic_logaristic_regression(DTR,LTR,args,logger):
    logger.info(f"Quadratic Regression")
    point = (0.5,1,10)
    lambda_ = np.logspace(-4, 2, num=7)
    # lambda_ = np.logspace(-2, 1, num=4)
    for piT in [0.1,0.5,0.9]:
        logger.debug(f"piT = {piT}")
        lr_pca = dict()
        for l in lambda_:
            logger.debug(f"l = {l}")
            for pca in [6,7,0]:
                logger.debug(f"pca = {pca}")
                logObj = quadLogRegClass(l,piT)
                args.pca = pca
                min_DCF, scores, labels = KFCV(DTR, LTR,logObj,point,args,logger)
                logger.info(f"Log Reg min_DCF con K = {args.k} , pca = {pca}, l = {l} , piT = {piT}: {min_DCF} ")
    
                try:
                    lr_pca[f"{pca}"].append(min_DCF)
                except:
                    lr_pca[f"{pca}"] = [min_DCF]
        
        logger.info(f"plotting")
        plot_logreg(lambda_,piT,lr_pca,f"QLR_DCF_l_k{args.k}_z-{args.znorm}_piT_{piT}_v2.png",args)

def main(args,logger):
    logger.info("Starting main function on logistic regression")
    # print(args.znorm)
    D,L = load(logger)
    # binary_logaristic_regression(D,L,args,logger)
    logger.info(f"_"*60)
    quadratic_logaristic_regression(D,L,args,logger)
    