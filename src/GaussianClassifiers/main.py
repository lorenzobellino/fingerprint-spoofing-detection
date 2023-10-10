import numpy as np
import scipy as sc
from Utils.utils import load,vcol, vrow, split_db_n, split_db_kfold, znorm, PCA_projection
import matplotlib.pyplot as plt

class GaussClass:
    
    def __init__(self,mode,prior,Cfp,Cfn):
       
        self.means = 0
        self.eff_prior=0
        self.S_matrices = 0
        self.ll0 = 0
        self.ll1 = 0
        self.mode = mode
        self.eff_prior = (prior*Cfn)/(prior*Cfn + (1-prior)*Cfp)
        
        
    def name(self):
        return self.mode
        
    def train(self,DTR,LTR):
        if self.mode == "MVG":
            means,S_matrices,_ = MVG_model(DTR,LTR)
        elif self.mode == "NB":
            means,S_matrices,_ = MVG_model(DTR,LTR) #3 means and 3 S_matrices -> 1 for each class (3 classes)
            for i in range(np.array(S_matrices).shape[0]):
                S_matrices[i] = S_matrices[i]*np.eye(S_matrices[i].shape[0],S_matrices[i].shape[1])
        elif self.mode == "TCG":
            means,S_matrix = TCG_model(DTR,LTR) #3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes
            S_matrices = [S_matrix,S_matrix,S_matrix]
        elif self.mode == "TCGNB":
            means,S_matrix = TCG_model(DTR,LTR) #3 means and 1 S_matrix -> tied matrix because of strong dipendence among the classes
            S_matrix = S_matrix * np.eye(S_matrix.shape[0],S_matrix.shape[1])
            S_matrices = [S_matrix,S_matrix,S_matrix]
        else:
            print(f"Model variant {self.mode} not supported!")
           
        self.means = means
        self.S_matrices = S_matrices
        
    def compute_scores(self,DTE):
        
       llr = loglikelihoods(DTE,self.means,self.S_matrices, [1-self.eff_prior,self.eff_prior])
        
       return llr
        
def PCA(D,m):
    """PCA function

    Args:
        D (np.array): dataset
        m (int): directions

    Returns:
        DP, P
    """
    DC = centerData(D)
    C = createCenteredCov(DC)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    return DP,P

def logpdf_GAU_ND_fast(X, mu, C) -> np.array:
    """Generate the multivariate Gaussian Density for a Matrix of N samples

    Args:
        X (np.Array): (M,N) matrix of N samples of dimension M
        mu (np.Array): array of shape (M,1) representing the mean
        C (np.Array): array of shape (M,M) representig the covariance matrix
    """
    mu = mu.reshape(mu.shape[0], 1)
    XC = X - mu
    M = X.shape[0]
    invC = np.linalg.inv(C)
    _, logDetC = np.linalg.slogdet(C)
    v = (XC * np.dot(invC, XC)).sum(0)

    lpdf = -(M / 2) * np.log(2 * np.pi) - (1 / 2) * logDetC - (1 / 2) * v
    return lpdf

def confusion_matrix(pred,LTE):
    """confusion matrix

    Args:
        pred (np.array): prediction
        LTE (np.arraty): labels

    Returns:
        np.array : confusion matrix
    """
    nclasses = int(np.max(LTE))+1
    matrix = np.zeros((nclasses,nclasses))
    
    
    for i in range(len(pred)):
        matrix[pred[i],LTE[i]] += 1
        
    return matrix

def binary_posterior_prob(llr,prior,Cfn,Cfp):
    """binary posterior probability
    """
    new_llr = np.zeros(llr.shape)
    for i in range(len(llr)):
        new_llr[i] = llr[i] + np.log(prior*Cfn/((1-prior)*Cfp))
        
    return new_llr

def binary_DCFu(prior,Cfn,Cfp,cm):
    """binary DCFu
    """
    FNR = cm[0,1]/(cm[0,1]+cm[1,1])
    FPR = cm[1,0]/(cm[1,0]+cm[0,0])
    
    DCFu = prior*Cfn*FNR + (1-prior)*Cfp*FPR
    
    return DCFu

def DCF_min_impl(llr,label,prior,Cfp,Cfn):
    """DCF min implementation
    """
    post_prob = binary_posterior_prob(llr,prior,Cfn,Cfp)
    thresholds = np.sort(post_prob)
    DCF_tresh = []
    dummy_DCFu = min(prior*Cfn,(1-prior)*Cfp)
    for t in thresholds:
        pred = [1 if x >= t else 0 for x in post_prob]
        cm = confusion_matrix(pred, label)
        DCF_tresh.append(binary_DCFu(prior, Cfn, Cfp, cm)/dummy_DCFu)
        
    min_DCF = min(DCF_tresh)
    t_min = thresholds[np.argmin(DCF_tresh)]
    
    return min_DCF,t_min,thresholds

def loglikelihoods(DTE,means,S_matrices,prior):
    """calculate log likelyhoods
    """
    likelihoods=[]
    logSJoint=[]
    logSMarginal=0
    for i in range(2):
        mu=means[i]
        c=S_matrices[i]
        ll=logpdf_GAU_ND_fast(DTE, mu, c)
        likelihoods.append(ll)
        logSJoint.append(ll+np.log(prior[i]))
        
    logSMarginal = vrow(sc.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    llr = logSPost[1,:] - logSPost[0,:] - np.log(prior[1]/prior[0])
    
    return llr

def createCenteredCov(DC):
    C = 0
    for i in range(DC.shape[1]):
        C = C + np.dot(DC[:,i:i+1],(DC[:,i:i+1]).T)
    C = C/float(DC.shape[1])
    return C

def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)    #broadcasting applied
    return DC

def MVG_model(D,L):
    c0 = []
    c1 = []
    means = []
    S_matrices = []
    
    for i in range(D.shape[1]):
        if L[i] == 0:
            c0.append(D[:,i])
        elif L[i] == 1:
            c1.append(D[:,i])
    

    c0 = (np.array(c0)).T
    c1 = (np.array(c1)).T           
    
    c0_cent = centerData(c0)
    c1_cent = centerData(c1)
    
    S_matrices.append(createCenteredCov(c0_cent)) 
    S_matrices.append(createCenteredCov(c1_cent))        
    
    means.append(vcol(c0.mean(1)))
    means.append(vcol(c1.mean(1)))

    
    return means,S_matrices,(c0.shape[1],c1.shape[1])

def TCG_model(D,L):

    S_matrix = 0
    means,S_matrices,cN = MVG_model(D, L)
    
    cN = np.array(cN)
    
    S_matrices = np.array(S_matrices)
    
    D_cent = centerData(D)
    
    for i in range(cN.shape[0]):
        
        S_matrix += cN[i]*S_matrices[i]  
    
    S_matrix /=D.shape[1]
    
    return means,S_matrix

def KFCV(D, L, classifier ,point, args, logger):
    """K fold cross vaslidation

    Args:
        D (np.array): dataset
        L (np.array): labels
        classifier (callble): model
        point (tuple): working point

    Returns:
        min_DCF, scores, labels
    """
   
    pi,Cfn,Cfp = point
    
    samplesNumber = D.shape[1]
    N = int(samplesNumber / args.k)
    
    np.random.seed(seed=42)
    indexes = np.random.permutation(D.shape[1])
    
    scores = np.array([])
    labels = np.array([])
    
    
    for i in range(args.k):
        idxTest = indexes[i*N:(i+1)*N]
        
        idxTrainLeft = indexes[0:i*N]
        idxTrainRight = indexes[(i+1)*N:]
        idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
        
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]   
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        
        if znorm == True:
            DTR,DTE = znorm(DTR,DTE)
        
        if args.pca > 0: 
            DTR, P = PCA(DTR, args.pca)
            DTE = np.dot(P.T, DTE)
               
        classifier.train(DTR, LTR)
        
        scores_i = classifier.compute_scores(DTE)
        # logger.debug(f"Scores: {scores_i}")

        scores = np.append(scores, scores_i)
        labels = np.append(labels, LTE)
        
        
    labels = np.array(labels,dtype=int)
    min_DCF,_,_ = DCF_min_impl(scores, labels, pi, Cfp, Cfn)
    
    return min_DCF, scores, labels

def main(args,logger) -> None:
    D,L = load(logger)

    prior,Cfp,Cfn = (0.5,10,1)
    # prior,Cfp,Cfn = (1/11,1,1)

    MVG_obj = GaussClass("MVG",prior,Cfp,Cfn)
    NB_obj = GaussClass("NB",prior,Cfp,Cfn)
    TCG_obj = GaussClass("TCG",prior,Cfp,Cfn)
    TCGNB_obj = GaussClass("TCGNB",prior,Cfp,Cfn) 


    for model in [MVG_obj, NB_obj, TCG_obj, TCGNB_obj]:
        min_DCF,scores,labels = KFCV(D, L, model, (prior,Cfn,Cfp), args, logger)
        logger.info(f"{model.name()} min_DCF con K = {args.k} : {min_DCF} ")

        

