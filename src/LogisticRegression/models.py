import numpy as np
import scipy as sc
import scipy.optimize as opt
from Utils.utils import load, split_db_2to1,vcol,vrow,plot_logreg

class quadLogRegClass:
    """Quadratic logistic regression class
    """
    def __init__(self,l,piT):
        self.l = l        
        self.piT = piT

    def gradient_test(self,DTR, LTR, l, pt,nt,nf):
        """grdient test

        Args:
            DTR (np.array): dataset
            LTR (np.array): labels

        Returns:
            gradient : callable
        """
        z=np.empty((LTR.shape[0]))    
        z=2*LTR-1
        def gradient(v):        
            w, b = v[0:-1], v[-1]
            
            second_term=0        
            third_term = 0
     
            first_term = l*w

            for i in range(DTR.shape[1]):
                S=np.dot(w.T,DTR[:,i])+b            
                ziSi = np.dot(z[i], S)
                if LTR[i] == 1:
                    internal_term = np.dot(np.exp(-ziSi),(np.dot(-z[i],DTR[:,i])))/(1+np.exp(-ziSi))                #print(1+np.exp(-ziSi))
                    second_term += internal_term            
                else :
                    internal_term_2 = np.dot(np.exp(-ziSi),(np.dot(-z[i],DTR[:,i])))/(1+np.exp(-ziSi))                
                    third_term += internal_term_2
            derivative_w= first_term + (pt/nt)*second_term + (1-pt)/(nf) * third_term
            first_term = 0                   
            second_term=0
    
            for i in range(DTR.shape[1]):
                S=np.dot(w.T,DTR[:,i])+b
                ziSi = np.dot(z[i], S)
                if LTR[i] == 1:                
                    internal_term = (np.exp(-ziSi) * (-z[i]))/(1+np.exp(-ziSi))
                    first_term += internal_term            
                else :
                    internal_term_2 = (np.exp(-ziSi) * (-z[i]))/(1+np.exp(-ziSi))                
                    second_term += internal_term_2
    
            derivative_b= (pt/nt)*first_term + (1-pt)/(nf) * second_term        
            grad = np.hstack((derivative_w,derivative_b))
            return grad
        return gradient
        
    def quad_logreg_obj(self,v):
        """quad logreg object
        """
        loss = 0
        loss_c0 = 0
        loss_c1 = 0
        
        w,b = v[0:-1],v[-1]
        w = vcol(w)
        n = self.fi_x.shape[1]
        
        regularization = (self.l / 2) * np.sum(w ** 2) 
        
        for i in range(n):
            
            if (self.LTR[i:i+1] == 1):
                zi = 1
                loss_c1 += np.logaddexp(0,-zi * (np.dot(w.T,self.fi_x[:,i:i+1]) + b))
            else:
                zi=-1
                loss_c0 += np.logaddexp(0,-zi * (np.dot(w.T,self.fi_x[:,i:i+1]) + b))
        
        J = regularization + (self.piT / self.nT) * loss_c1 + (1-self.piT)/self.nF * loss_c0
        grad = self.grad_funct(v)
        return J,grad
    
    def train(self,DTR,LTR):
        """training method

        Args:
            DTR (np.array): dataset
            LTR (np.asrray): labels

        Returns:
            b , w
        """
        self.DTR  = DTR
        self.LTR = LTR
        self.nt = DTR[:, LTR == 1].shape[1]
        self.nf = DTR.shape[1]-self.nt
        
        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT
        
        n_features = DTR.shape[1]
        expanded_DTR = np.apply_along_axis(vecxxT, 0, DTR)
        self.fi_x = np.vstack([expanded_DTR, DTR])
        
        x0 = np.zeros(self.fi_x.shape[0] + 1)
        
        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])
        self.grad_funct = self.gradient_test(self.fi_x, self.LTR, self.l, self.piT,self.nt,self.nf)
        
        params,f_min,_ = sc.optimize.fmin_l_bfgs_b(self.quad_logreg_obj, x0)
      
        self.b = params[-1]
        
        self.w = np.array(params[0:-1])
        
        self.S = []
        
        
        return self.b,self.w
    
    def compute_scores(self,DTE):
        """compute scores

        Args:
            DTE (np.array): dataset

        Returns:
            S: scores
        """
        
        S = self.S

        for i in range(DTE.shape[1]):
            x = vcol(DTE[:,i:i+1])
            mat_x = np.dot(x,x.T)
            vec_x= vcol(np.hstack(mat_x))
            fi_x = np.vstack((vec_x,x))
            self.S.append(np.dot(self.w.T,fi_x) + self.b)
            
        pred = [1 if x > 0 else 0 for x in S] #I transform in 1 all the pos values and in 0 all the negative ones
        
        return S



class logRegClass:
    """
    Class that implements the logistic regression classifier

    """
    def __init__(self,l,piT):
        self.l = l
        self.piT = piT
        
        
    def logreg_obj(self,v):
        """logreg object

        Args:
            v : vector

        Returns:
            J : 
        """
        loss = 0
        loss_c0 = 0
        loss_c1 = 0
        w,b = v[0:-1],v[-1]
        w = vcol(w)
        n = self.DTR.shape[1]
        
        regularization = (self.l / 2) * np.sum(w ** 2) 
        
        for i in range(n):
            
            if (self.LTR[i:i+1] == 1):
                zi = 1
                loss_c1 += np.logaddexp(0,-zi * (np.dot(w.T,self.DTR[:,i:i+1]) + b))
            else:
                zi=-1
                loss_c0 += np.logaddexp(0,-zi * (np.dot(w.T,self.DTR[:,i:i+1]) + b))
        
        J = regularization + (self.piT / self.nT) * loss_c1 + (1-self.piT)/self.nF * loss_c0
        
        return J
   

    
    def train(self,DTR,LTR):
        """training method

        Args:
            DTR (np.array): dataset
            LTR (np.array): labels

        Returns:
            b,w: optimized parameters
        """
        self.DTR  = DTR
        self.LTR = LTR
        x0 = np.zeros(DTR.shape[0] + 1)
        
        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])
        
        params,f_min,_ = sc.optimize.fmin_l_bfgs_b(self.logreg_obj, x0,approx_grad=True)
        
        self.b = params[-1]
        
        self.w = np.array(params[0:-1])
        
        self.S = []
        
        
        return self.b,self.w
    
    def compute_scores(self,DTE):
        """compute scores

        Args:
            DTE (np.array): dataset

        Returns:
            llr : scores
        """
        S = self.S
        for i in range(DTE.shape[1]):
            x = DTE[:,i:i+1]
            x = np.array(x)
            x = x.reshape((x.shape[0],1))
            self.S.append(np.dot(self.w.T,x) + self.b)
        
        S = [1 if x > 0 else 0 for x in S] #I transform in 1 all the pos values and in 0 all the negative ones
        
        llr = np.dot(self.w.T, DTE) + self.b
        
        return llr
 