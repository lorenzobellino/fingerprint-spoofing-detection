import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from itertools import permutations

# from PCA.main import PCA

PRJCT_ROOT = "/".join(os.getcwd().split("/"))


def plot_logreg(l, piT, lr, filename, args):
    plt.figure()
    plt.semilogx(l, lr[f"{6}"], label="PCA 6")
    plt.semilogx(l, lr[f"{7}"], label="PCA 7")
    # plt.semilogx(l,lr[f"{8}"], label = "PCA 8")
    plt.semilogx(l, lr[f"{0}"], label="No PCA")

    plt.xlabel("Lambda")
    plt.ylabel("DCF_min")
    plt.legend()
    plt.title(f"piT = {piT} znorm = {args.znorm}")
    plt.savefig(PRJCT_ROOT + "/plots/logreg/" + filename)
    # plt.show()


def randomize(D, L, seed=42):
    nTrain = int(D.shape[1])
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]

    DTR = D[:, idxTrain]
    LTR = L[idxTrain]

    return DTR, LTR


def split_db_n(D, L, seed=42, n=2 / 3):
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


def split_db_kfold(D, L, k, args, logger):
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


def compute_binaryOptimalBayesDecisions(pi, Cfn, Cfp, classifier_CP, th=None):
    """Compute bayes optimal decision based on passed parameters.
    Class posterior probabilities (test scores) obtained from a binary classifier are passed with classifier_CP.
    """
    if th == None:
        th = -np.log((pi * Cfn) / ((1 - pi) * Cfp))
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
    FNR = M[0, 1] / (M[0, 1] + M[1, 1])
    FPR = M[1, 0] / (M[0, 0] + M[1, 0])
    return pi * Cfn * FNR + (1 - pi) * Cfp * FPR  # bayes risk's formula for binary case


def compute_normalized_emp_Bayes_binary(M, pi, Cfn, Cfp):
    """Compute the normalized bayes risk, assuming thath M is a confusion matrix of a binary case"""
    empBayes = compute_emp_Bayes_binary(M, pi, Cfn, Cfp)
    B_dummy = np.array([pi * Cfn, (1 - pi) * Cfp]).min()
    return empBayes / B_dummy


def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    """Compute the actual DCF, which is basically the normalized bayes risk"""
    # compute opt bayes decisions
    Pred = compute_binaryOptimalBayesDecisions(pi, Cfn, Cfp, scores, th=th)
    # compute confusion matrix
    CM = comp_confmat(labels, Pred)
    # compute DCF and return it
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
    with open(PRJCT_ROOT + "/data/Train.txt", "r") as f:
        for line in f:
            d.append(
                np.array([float(_) for _ in line.strip().split(",")[:-1]]).reshape(
                    10, 1
                )
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
    plt.savefig(PRJCT_ROOT + "/" + filename)


def plot_hist(proj: np.array, c: np.array, filename: str) -> None:
    plt.figure()
    plt.hist(
        proj[0, c == 0],
        bins=100,
        density=True,
        alpha=0.4,
        linewidth=1.0,
        color="red",
        edgecolor="black",
        label="Original",
    )
    plt.hist(
        proj[0, c == 1],
        bins=100,
        density=True,
        alpha=0.4,
        linewidth=1.0,
        color="green",
        edgecolor="black",
        label="Spoofed",
    )
    plt.legend()
    plt.savefig(PRJCT_ROOT + "/" + filename)


def plot_features_histograms(DTR, LTR, filename):
    plt.rcParams.update(plt.rcParamsDefault)
    for i in range(10):
        labels = ["spoofed fingerprint", "fingerprint"]
        title = filename + f"_{i:02d}"
        plt.figure()
        plt.title(title)

        y = DTR[:, LTR == 0][i]
        plt.hist(
            y,
            bins=60,
            density=True,
            alpha=0.4,
            linewidth=1.0,
            color="red",
            edgecolor="black",
            label=labels[0],
        )
        y = DTR[:, LTR == 1][i]
        plt.hist(
            y,
            bins=60,
            density=True,
            alpha=0.4,
            linewidth=1.0,
            color="green",
            edgecolor="black",
            label=labels[1],
        )
        plt.legend()
        plt.savefig(PRJCT_ROOT + "/plots/" + title + ".png")
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
    heatmap = sns.heatmap(
        np.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False
    )
    fig = heatmap.get_figure()
    fig.savefig(PRJCT_ROOT + "/plots/" + title + ".png")
    plt.close()


def plot_explained_variance(logger, DTR, PCA):
    P, egnValues = PCA(logger, DTR, 10)
    # print(f"\n\n eigenvalues: {egnValues}\n\n")
    total_egnValues = sum(egnValues)
    var_exp = [(i / total_egnValues) for i in sorted(egnValues, reverse=True)]

    cum_sum_exp = np.cumsum(var_exp)

    x = range(1, len(cum_sum_exp) + 1)
    y = cum_sum_exp
    plt.figure()
    plt.bar(
        x,
        y,
        color="blue",
    )
    plt.xlabel("Principal component index")
    plt.ylabel("Explained variance ratio")
    plt.title("Explained variance ratio vs. Principal component index")
    for i in range(len(x)):
        plt.annotate(
            f"{y[i]*100:.1f}", xy=(x[i] - 0.05, y[i]), ha="center", va="bottom"
        )
    plt.xticks(range(1, 11))
    plt.savefig(PRJCT_ROOT + "/plots/PCA_explainedVariance.png")
    plt.close()
    # plt.close()

    prop_var = egnValues / total_egnValues
    plt.figure(figsize=(14, 10))
    plt.plot(np.arange(1, len(prop_var) + 1), prop_var, marker="o")
    plt.xlabel("Principal Component", size=20)
    plt.ylabel("Proportion of Variance Explained", size=20)
    plt.title("Figure 1: Scree Plot for Proportion of Variance Explained", size=25)
    plt.grid(True)
    plt.savefig(PRJCT_ROOT + "/plots/PCA_screePlot.png")
    # plt.show()
    plt.close()


def plot_linear_svm(C_values, piT, svm, filename, args):
    plt.figure()
    plt.semilogx(C_values, svm[f"pca-6_z-False"], label="PCA 6")
    plt.semilogx(C_values, svm[f"pca-6_z-True"], label="PCA 6 - Z_NORM")
    # plt.semilogx(C_values,svm_pca8, label = "PCA 8")
    # plt.semilogx(C_values,svm_pca9, label = "PCA 9")
    # plt.semilogx(C_values,svm_pcaNone, label = "No PCA")

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
    plt.savefig(PRJCT_ROOT + "plots/svm/" + filename)
    # plt.show()


def compute_correlation(X, Y):
    x_sum, y_sum = np.sum(X), np.sum(Y)

    x2_sum, y2_sum = np.sum(X**2), np.sum(Y**2)

    sum_cross_prod = np.sum(X * Y.T)

    n = X.shape[0]
    num = n * sum_cross_prod - x_sum * y_sum
    den = np.sqrt((n * x2_sum - x_sum**2) * (n * y2_sum - y_sum**2))

    corr = num / den
    return corr


def vcol(v) -> np.array:
    return v.reshape(v.size, 1)


def vrow(v) -> np.array:
    return v.reshape(1, v.size)


def write_data(data, filename):
    with open(PRJCT_ROOT + "/Results/" + filename, "w") as f:
        f.write(data)


def confusion_matrix(pred, LTE):
    """
    pred: list of predictions
    LTE: list of actual labels

    return: confusion matrix

    compute the confusion matrix starting from the predictions and the actual labels
    """
    nclasses = int(np.max(LTE)) + 1
    matrix = np.zeros((nclasses, nclasses))

    for i in range(len(pred)):
        matrix[pred[i], LTE[i]] += 1

    return matrix


def binary_posterior_prob(llr, prior, Cfn, Cfp):
    """

    llr: log likelihood ratio
    prior: prior probability
    Cfn: cost of false negative
    Cfp: cost of false positive

    return: posterior probabilities

    compute the posterior probabilities starting from the log likelihood ratio and the prior probability
    """
    new_llr = np.zeros(llr.shape)
    for i in range(len(llr)):
        # I just applied the llr formula in case of Cost addition
        # because of log properties I can rewrite it as sum between log_likelihood and prior and C log contribution
        new_llr[i] = llr[i] + np.log(prior * Cfn / ((1 - prior) * Cfp))

    return new_llr


# compute the un-normalized DCF
def binary_DCFu(prior, Cfn, Cfp, cm):
    """compute the un-normalized DCF

    Args:
        prior : prior probability
        Cfn : cost of false negative
        Cfp :   cost of false positive
        cm : confusion matrix

    Returns:
        DCFu : un-normalized DCF
    """
    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    FPR = cm[1, 0] / (cm[1, 0] + cm[0, 0])

    DCFu = prior * Cfn * FNR + (1 - prior) * Cfp * FPR

    return DCFu


# here we can plot the ROC plot which shows how the FPR and TPR change according to the current threshold value
def ROC_plot(thresholds, post_prob, LTE):
    """

    thresholds : list of thresholds
    post_prob : list of posterior probabilities
    LTE : list of actual labels

    Returns:
        FPR : list of False Positive Rates
        TPR : list of True Positive Rates

    plot the ROC plot which shows how the FPR and TPR change according to the current threshold value
    """
    FNR = []
    FPR = []
    for t in thresholds:
        # I consider t (and not prior) as threshold value for split up the dataset in the two classes' set
        pred = [1 if x >= t else 0 for x in post_prob]
        # I compute the confusion_matrix starting from these new results
        cm = confusion_matrix(pred, LTE)

        FNR.append(cm[0, 1] / (cm[0, 1] + cm[1, 1]))
        FPR.append(cm[1, 0] / (cm[1, 0] + cm[0, 0]))

    # plt.figure()
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")
    TPR = 1 - np.array(FNR)
    # plt.plot(FPR,TPR,scalex=False,scaley=False)
    # plt.show()

    return FPR, TPR


def Bayes_plot(llr, LTE):
    """
    llr : log likelihood ratio
    LTE : list of actual labels

    Returns:
        DCF_effPrior : dictionary of DCF values
        DCF_effPrior_min : dictionary of min DCF values

    compute the Bayes error plot
    """

    effPriorLogOdds = np.linspace(-3, 3, 21)
    DCF_effPrior = {}
    DCF_effPrior_min = {}
    print("In bayes_plot")
    # I try to compute DCF using several possible effPriorLogOdd values
    for p in effPriorLogOdds:

        # I compute the effPrior pi_tilde from the current effPriorLogOdds
        effPrior = 1 / (1 + np.exp(-p))

        # computation of the post_prob using pi_tilde as thresholds (not the optimal choice)
        post_prob = binary_posterior_prob(llr, effPrior, 1, 1)
        pred = [1 if x >= 0 else 0 for x in post_prob]
        cm = confusion_matrix(pred, LTE)

        # computation of the not optimal DCF value bound to the specific pi_tilde choice
        dummy = min(effPrior, (1 - effPrior))
        # now that we habe an effPrior which inglobes Cfn and Cfp contributions we can still recycle the old DCF formula but
        # passing (effPrior,1,1) as parameters
        DCF_effPrior[p] = binary_DCFu(effPrior, 1, 1, cm) / dummy

        # loop over all the possible thresholds values to find the optimal t
        thresholds = np.sort(post_prob)
        tmp_DCF = []
        for t in thresholds:
            # I consider t (so not prior neither 0!!!) as threshold value for splitting up the dataset in the two classes' set
            pred = [1 if x >= t else 0 for x in post_prob]
            # I compute the confusion_matrix starting from these new results
            cm = confusion_matrix(pred, LTE)
            # I save the normalized DCU computed for each t value
            tmp_DCF.append((binary_DCFu(effPrior, 1, 1, cm) / dummy))

        # computation of the min DCF bound to the specific t and the specific pi_tilde
        DCF_effPrior_min[p] = np.min(tmp_DCF)

    # log(pi/(1-pi)) on the x-axis

    plt.plot(effPriorLogOdds, DCF_effPrior.values(), label="DCF", color="r")
    plt.plot(effPriorLogOdds, DCF_effPrior_min.values(), label="min DCF", color="b")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.show()

    return DCF_effPrior, DCF_effPrior_min


def DCF_norm_impl(llr, label, prior, Cfp, Cfn):
    """

    llr : log likelihood ratio
    label : list of actual labels
    prior : prior probability
    Cfp : cost of false positive
    Cfn : cost of false negative

    Returns:
        DCF_norm : normalized DCF

    compute the normalized DCF NOT CALIBRATED
    """
    # optimal bayes decision for inf-par binary problem
    # infpar_llr = np.load("Data\commedia_llr_infpar_eps1.npy")
    # infpar_label =np.load("Data\commedia_labels_infpar_eps1.npy")
    # prior,Cfp,Cfn = (0.5,1,1)
    post_prob = binary_posterior_prob(llr, prior, Cfn, Cfp)
    pred = [1 if x > 0 else 0 for x in post_prob]
    cm = confusion_matrix(pred, label)
    # observe that when prior increase = class1 predicted more frequently by the classifier
    # when Cfn increases classifiers will make more FP errors and less FN errors -> the opposite for the opposite case

    # binary task evaluation
    # DCFu doesn't allow us to comparing different systems so it's only the first step to compute DCF
    DCFu = binary_DCFu(prior, Cfn, Cfp, cm)
    # let's compute DCF now, normalizing DCFu with a dummy system DCFu
    dummy_DCFu = min(prior * Cfn, (1 - prior) * Cfp)
    # print("dummy DCF: "+str(dummy_DCFu))
    # print("DCFu : "+str(DCFu));
    DCF_norm = DCFu / dummy_DCFu

    return DCF_norm


def DCF_min_impl(llr, label, prior, Cfp, Cfn):
    """
    llr : log likelihood ratio
    label : list of actual labels
    prior : prior probability
    Cfp : cost of false positive
    Cfn : cost of false negative

    Returns:
        min_DCF : minimum DCF
        t_min : minimum threshold
        thresholds : list of thresholds

    compute the minimum DCF
    """
    post_prob = binary_posterior_prob(llr, prior, Cfn, Cfp)
    thresholds = np.sort(post_prob)
    DCF_tresh = []
    dummy_DCFu = min(prior * Cfn, (1 - prior) * Cfp)
    # print("shape di post prob in DCF_min: "+str(post_prob.shape))

    # iteration over all the possible threshold values
    for t in thresholds:
        # I consider t (and not prior) as threshold value for split up the dataset in the two classes' set

        pred = [1 if x >= t else 0 for x in post_prob]
        # I compute the confusion_matrix starting from these new results
        cm = confusion_matrix(pred, label)
        # I save the normalized DCU computed for each t value
        DCF_tresh.append(binary_DCFu(prior, Cfn, Cfp, cm) / dummy_DCFu)

    # I choose the minimum DCF value and the relative threshold value
    min_DCF = min(DCF_tresh)
    t_min = thresholds[np.argmin(DCF_tresh)]

    # Observe how much loss due to poor calibration (the difference is clear if you compare DCF_norm with min_DCF)

    return min_DCF, t_min, thresholds


def PCA_impl(D, m):
    DC = centerData(D)
    C = createCenteredCov(DC)
    P = createP(C, m)
    DP = np.dot(P.T, D)

    return DP, P


def createCenteredCov(DC):
    C = 0
    for i in range(DC.shape[1]):
        C += np.dot(DC[:, i : i + 1], DC[:, i : i + 1].T)

    C /= float(DC.shape[1])
    return C


def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)
    return DC


def createP(C, m):
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]

    return P


def ML_estimate(X):
    mu_ml = dataset_mean(dataset=X)
    covariance_matrix_ml = (np.dot(((X - mu_ml)), (X - mu_ml).T)) / num_samples(X)

    return mu_ml, covariance_matrix_ml


def dataset_mean(dataset):
    return dataset.mean(1).reshape(
        (dataset.shape[0], 1)
    )  # dataset.shape[0] è numero features per rendere codice generico


def num_samples(dataset):
    return dataset.shape[1]


def logpdf_GAU_ND_1Sample(x, mu, C):
    # it seems that also for just one row at time we should use mu and C of the whole matrix
    xc = x - mu
    M = x.shape[0]
    logN = 0
    const = -0.5 * M * np.log(2 * np.pi)
    log_determ = np.linalg.slogdet(C)[1]
    lamb = np.linalg.inv(C)
    third_elem = np.dot(xc.T, np.dot(lamb, xc)).ravel()
    logN = const - 0.5 * log_determ - 0.5 * third_elem

    return logN


def logpdf_GAU_ND(X, mu, C):  # logpdf_GAU_ND algorithm for a 2-D matrix
    logN = []
    # print("Dim di X in logpdf_GAU_ND: "+str(X.shape))
    for i in range(X.shape[1]):
        # [:,i:i+1] notation allows us to take just the i-th column at time
        # remember that with this notation we mean [i,i+1) (left-extreme not included)
        logN.append(logpdf_GAU_ND_1Sample(X[:, i : i + 1], mu, C))
    return np.array(logN).ravel()


def logpdf_GMM_1Sample(x, mu, C, pi):
    # it seems that also for just one row at time we should use mu and C of the whole matrix
    xc = x - mu
    M = x.shape[0]
    logN = 0
    const = -0.5 * M * np.log(2 * np.pi)
    log_determ = np.linalg.slogdet(C)[1]
    lamb = np.linalg.inv(C)
    third_elem = np.dot(xc.T, np.dot(lamb, xc)).ravel()
    logN = const - 0.5 * log_determ - 0.5 * third_elem
    logN = np.log(pi) + logN
    return logN


def logpdf_GMM(X, mu, C, pi):
    logN = []
    for i in range(X.shape[1]):
        # [:,i:i+1] notation allows us to take just the i-th column at time
        # remember that with this notation we mean [i,i+1) (left-extreme not included)
        logN.append(logpdf_GMM_1Sample(X[:, i : i + 1], mu, C, pi))
    return np.array(logN).ravel()
