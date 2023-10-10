import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from itertools import permutations

# from PCA.main import PCA

PRJCT_ROOT = "/".join(os.getcwd().split("/")[:-1])

def plot_logreg(l,piT,lr,filename,args):
    plt.figure()
    plt.semilogx(l,lr[f"{6}"], label = "PCA 6")
    plt.semilogx(l,lr[f"{7}"], label = "PCA 7")
    # plt.semilogx(l,lr[f"{8}"], label = "PCA 8")
    plt.semilogx(l,lr[f"{0}"], label = "No PCA")
    
    plt.xlabel("Lambda")
    plt.ylabel("DCF_min")
    plt.legend()
    plt.title(f"piT = {piT} znorm = {args.znorm}")
    plt.savefig(PRJCT_ROOT+ "/plots/logreg/" + filename)
    # plt.show()

def randomize(D, L, seed=42):
    nTrain = int(D.shape[1])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]

    DTR = D[:, idxTrain]
    LTR = L[idxTrain]

    return DTR, LTR

def split_db_n(D,L,seed=42,n=2/3):
    nTrain = int(D.shape[1] * n)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return DTR, DTE, LTR, LTE

def split_db_kfold(D, L, k, args,logger):
    logger.debug("split db in K folds")
    samples = D.shape[1]
    N = samples // k
    
    indexes = np.random.permutation(samples)
    scores = np.zeros(k)
    labels = np.array([])
    DTR_folds = []
    DTE_folds = []
    LTR_folds = []
    LTE_folds = []

    for i in range(k):
        idxTest = indexes[i * N : (i + 1) * N]
        idxTrainLeft = indexes[0 : i * N]
        idxTrainRight = indexes[(i + 1) * N :]
        idxTrain = np.hstack([idxTrainLeft, idxTrainRight])

        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]

        if args.znorm:
            DTR, DTE = znorm(DTR, DTE)

        
        if args.pca > 0:
            logger.debug("PCA projection")
            DTR, DTE = PCA_projection(DTR, DTE, args.pca)

        DTR_folds.append(DTR)
        DTE_folds.append(DTE)
        LTR_folds.append(LTR)
        LTE_folds.append(LTE)

    return DTR_folds, DTE_folds, LTR_folds, LTE_folds
        
def PCA_projection(DTR, DTE, m):
    mu = vcol(np.mean(DTR, axis=1))
    center = DTR - mu
    cov = np.dot(center, center.T) / DTR.shape[1]
    eigvecs, _, _ = np.linalg.svd(cov)
    P = eigvecs[:, 0:m]
    DTR = np.dot(P.T, DTR)
    DTE = np.dot(P.T, DTE)
    return DTR, DTE

def znorm(DTR, DTE):
    mu = vcol(np.mean(DTR, axis=1))
    sigma = vcol(np.std(DTR, axis=1))

    ZD = DTR - mu
    ZD = ZD / sigma

    ZD2 = DTE - mu
    ZD2 = ZD2 / sigma

    return ZD, ZD2

def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    """Compute the minDCF"""
    t = np.array(scores)
    t.sort()
    np.concatenate([np.array([-np.inf]), t, np.array([np.inf])])
    dcfList = []
    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=_th))
    return np.array(dcfList).min()

def compute_binaryOptimalBayesDecisions(pi,Cfn,Cfp,classifier_CP, th = None):
    """Compute bayes optimal decision based on passed parameters.
       Class posterior probabilities (test scores) obtained from a binary classifier are passed with classifier_CP.
    """
    if th == None:
        th = -np.log((pi*Cfn)/((1-pi)*Cfp))
    OptDecisions = np.array([classifier_CP > th])
    return np.int32(OptDecisions)


def comp_confmat(labels, predicted):
    """Compute confusion matrix.
       Inputs are:
       - labels: labels of samples, each position is a sample, the content is the relative label
       - predicted: a np vector containing predictions of samples, structured as labels"""
    # extract the different classes
    classes = np.unique(labels)
    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))
    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):
           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((labels == classes[i]) & (predicted == classes[j]))
    return confmat.T.astype(int)

def compute_emp_Bayes_binary(M, pi, Cfn, Cfp):
    """Compute the empirical bayes risk, assuming thath M is a confusion matrix of a binary case"""
    FNR = M[0,1]/(M[0,1] + M[1,1])
    FPR = M[1,0]/(M[0,0] + M[1,0])
    return pi*Cfn*FNR + (1-pi)*Cfp*FPR # bayes risk's formula for binary case 


def compute_normalized_emp_Bayes_binary(M, pi, Cfn, Cfp):
    """Compute the normalized bayes risk, assuming thath M is a confusion matrix of a binary case"""
    empBayes = compute_emp_Bayes_binary(M,pi,Cfn,Cfp) 
    B_dummy = np.array([pi*Cfn, (1-pi)*Cfp]).min()
    return empBayes / B_dummy

def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    """Compute the actual DCF, which is basically the normalized bayes risk"""
    #compute opt bayes decisions
    Pred = compute_binaryOptimalBayesDecisions(pi, Cfn, Cfp, scores, th=th)
    #compute confusion matrix
    CM = comp_confmat(labels, Pred)
    #compute DCF and return it
    return compute_normalized_emp_Bayes_binary(CM, pi, Cfn, Cfp)

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2 / 3)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return DTR, DTE, LTR, LTE

def load(logger):
    logger.info("Loading the dataset")
    d = []
    c = []
    with open(PRJCT_ROOT+"/data/Train.txt", "r") as f:
        for line in f:
            d.append(
                np.array([float(_) for _ in line.strip().split(",")[:-1]]).reshape(10, 1)
            )
            c.append(int(line.strip().split(",")[-1]))
    dataset = np.hstack(d)
    classes = np.array(c)
    return dataset, classes

def plot(proj: np.array, c: np.array, filename: str) -> None:
    plt.figure()
    plt.scatter(proj[0, c == 0], proj[1, c == 0], label="Original")
    plt.scatter(proj[0, c == 1], proj[1, c == 1], label="Spoofed")
    plt.legend()
    plt.savefig(PRJCT_ROOT +"/" + filename)

def plot_hist(proj: np.array, c: np.array, filename: str) -> None:
    plt.figure()
    plt.hist(proj[0, c == 0], bins=100, density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black', label="Original")
    plt.hist(proj[0, c == 1], bins=100, density=True, alpha=0.4, linewidth=1.0, color='green', edgecolor='black', label="Spoofed")
    plt.legend()
    plt.savefig(PRJCT_ROOT +"/" + filename)

def plot_features_histograms(DTR, LTR, filename):
    plt.rcParams.update(plt.rcParamsDefault)
    for i in range(10):
        labels = ["spoofed fingerprint", "fingerprint"]
        title = filename + f"_{i:02d}"
        plt.figure()
        plt.title(title)

        y = DTR[:, LTR == 0][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black', label=labels[0])
        y = DTR[:, LTR == 1][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='green', edgecolor='black', label=labels[1])
        plt.legend()
        plt.savefig(PRJCT_ROOT+ "/plots/" + title + '.png')
        # plt.show()
        plt.close()

def plot_correlations(DTR, title, cmap="Greys"):
    corr = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = compute_correlation(X, Y)
            corr[x][y] = pearson_elem

    sns.set()
    heatmap = sns.heatmap(np.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False)
    fig = heatmap.get_figure()
    fig.savefig(PRJCT_ROOT + "/plots/" + title + ".png")
    plt.close()

def plot_explained_variance(logger,DTR,PCA):
    P,egnValues = PCA(logger,DTR, 10)
    # print(f"\n\n eigenvalues: {egnValues}\n\n")
    total_egnValues = sum(egnValues)
    var_exp = [(i / total_egnValues) for i in sorted(egnValues, reverse=True)]

    cum_sum_exp = np.cumsum(var_exp)

    x = range(1, len(cum_sum_exp)+1)
    y = cum_sum_exp
    plt.figure()
    plt.bar(x,y, color='blue',)
    plt.xlabel('Principal component index')
    plt.ylabel('Explained variance ratio')
    plt.title('Explained variance ratio vs. Principal component index')
    for i in range(len(x)):
        plt.annotate(f"{y[i]*100:.1f}", xy=(x[i]-0.05, y[i]),ha='center', va='bottom')
    plt.xticks(range(1,11))
    plt.savefig(PRJCT_ROOT+ "/plots/PCA_explainedVariance.png")
    plt.close()
    # plt.close()

    prop_var = egnValues / total_egnValues
    plt.figure(figsize=(14,10))
    plt.plot(np.arange(1, len(prop_var)+1), 
                    prop_var, marker='o')
    plt.xlabel('Principal Component',
            size = 20)
    plt.ylabel('Proportion of Variance Explained',
            size = 20)
    plt.title('Figure 1: Scree Plot for Proportion of Variance Explained',
            size = 25)
    plt.grid(True)
    plt.savefig(PRJCT_ROOT+ "/plots/PCA_screePlot.png")
    # plt.show()
    plt.close()

def plot_linear_svm(C_values,piT,svm,filename,args):
    plt.figure()
    plt.semilogx(C_values,svm[f"pca-6_z-False"], label = "PCA 6")
    plt.semilogx(C_values,svm[f"pca-6_z-True"], label = "PCA 6 - Z_NORM")
    #plt.semilogx(C_values,svm_pca8, label = "PCA 8")
    #plt.semilogx(C_values,svm_pca9, label = "PCA 9")
    #plt.semilogx(C_values,svm_pcaNone, label = "No PCA")
        
    plt.xlabel("C")
    plt.ylabel("DCF_min")
    plt.legend()
    # if piT == 0.1:
    #     path = "plots/svm/DCF_su_C_piT_min"
    # if piT == 0.33:
    #     path = "plots/svm/DCF_su_C_piT_033"
    # if piT == 0.5:
    #     path = "plots/svm/DCF_su_C_piT_medium"
    # if piT == 0.9:
    #     path = "plots/svm/DCF_su_C_piT_max"
        
        
    plt.title(f"piT = {piT}")
    plt.savefig(PRJCT_ROOT+"plots/svm/" + filename)
    # plt.show()

def compute_correlation(X, Y):

    x_sum,y_sum = np.sum(X), np.sum(Y)

    x2_sum, y2_sum = np.sum(X**2), np.sum(Y**2)

    sum_cross_prod = np.sum(X*Y.T)

    n = X.shape[0]
    num = n * sum_cross_prod - x_sum * y_sum
    den = np.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = num/ den
    return corr


def vcol(v) -> np.array:
    return v.reshape(v.size, 1)


def vrow(v) -> np.array:
    return v.reshape(1, v.size)


def write_data(data,filename):
    with open(PRJCT_ROOT+"/Results/"+filename, "w") as f:
        f.write(data)
        