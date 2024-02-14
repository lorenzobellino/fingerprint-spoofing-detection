import np as np
import matplotlib.pyplot as plt
from Utils.utils import *
import scipy as sc


class quadLogRegClass:
    def __init__(self, l, piT):
        self.l = l

        # due of UNBALANCED classes (the spoofed has significantly more samples) we have to put a piT to apply this weigth
        self.piT = piT

    def gradient_test(self, DTR, LTR, l, pt, nt, nf):
        z = np.empty((LTR.shape[0]))
        z = 2 * LTR - 1

        def gradient(v):
            # print("in gradient")
            # grad = np.array( [derivative_w(v),derivative_b(v)], dtype = np.float64)
            # print("derivative w: ", derivative_w(v).size)        # print("derivative b: ", derivative_b(v).size)
            w, b = v[0:-1], v[-1]
            # print("w shape: ", w.size)        #print("w", w)
            # derivata rispetto a w        first_term = l*w

            second_term = 0
            third_term = 0

            # nt = self.nt
            # nf = self.nf
            # empirical prior        pt=nt/DTR.shape[1]
            first_term = l * w
            # print("size di DTR: "+str(DTR.shape))
            # print("size di z: "+str(z.shape))
            for i in range(DTR.shape[1]):  # S=DTR[:,i]
                S = np.dot(w.T, DTR[:, i]) + b
                ziSi = np.dot(z[i], S)
                if LTR[i] == 1:
                    internal_term = np.dot(
                        np.exp(-ziSi), (np.dot(-z[i], DTR[:, i]))
                    ) / (
                        1 + np.exp(-ziSi)
                    )  # print(1+np.exp(-ziSi))
                    second_term += internal_term
                else:
                    internal_term_2 = np.dot(
                        np.exp(-ziSi), (np.dot(-z[i], DTR[:, i]))
                    ) / (1 + np.exp(-ziSi))
                    third_term += internal_term_2
            # derivative_w= first_term + (pi/nt)*second_term + (1-pi)/(nf) * third_term
            derivative_w = (
                first_term + (pt / nt) * second_term + (1 - pt) / (nf) * third_term
            )
            # derivata rispetto a b
            first_term = 0
            second_term = 0

            for i in range(DTR.shape[1]):  # S=DTR[:,i]
                S = np.dot(w.T, DTR[:, i]) + b
                ziSi = np.dot(z[i], S)
                if LTR[i] == 1:
                    internal_term = (np.exp(-ziSi) * (-z[i])) / (1 + np.exp(-ziSi))
                    first_term += internal_term
                else:
                    internal_term_2 = (np.exp(-ziSi) * (-z[i])) / (1 + np.exp(-ziSi))
                    second_term += internal_term_2

            # derivative_b= (pi/nt)*first_term + (1-pi)/(nf) * second_term
            derivative_b = (pt / nt) * first_term + (1 - pt) / (nf) * second_term
            grad = np.hstack((derivative_w, derivative_b))
            return grad

        return gradient

    def quad_logreg_obj(self, v):
        # print("In quad_logreg_obj")
        loss = 0
        loss_c0 = 0
        loss_c1 = 0
        # for each possible v value in the current iteration (which corresponds to specific coord
        # obtained by the just tracked movement plotted from the actual Hessian and Gradient values and the previous calculated coord)
        # we extrapolate the w and b parameters to insert in the J loss-function

        w, b = v[0:-1], v[-1]
        w = vcol(w)
        n = self.fi_x.shape[1]

        regularization = (self.l / 2) * np.sum(w**2)

        # it's a sample way to apply the math transformation z = 2c - 1
        for i in range(n):

            if self.LTR[i : i + 1] == 1:
                zi = 1
                loss_c1 += np.logaddexp(
                    0, -zi * (np.dot(w.T, self.fi_x[:, i : i + 1]) + b)
                )
            else:
                zi = -1
                loss_c0 += np.logaddexp(
                    0, -zi * (np.dot(w.T, self.fi_x[:, i : i + 1]) + b)
                )

        J = (
            regularization
            + (self.piT / self.nT) * loss_c1
            + (1 - self.piT) / self.nF * loss_c0
        )
        grad = self.grad_funct(v)
        return J, grad

    def train(self, DTR, LTR):
        # print("In train")
        self.DTR = DTR
        self.LTR = LTR
        self.nt = DTR[:, LTR == 1].shape[1]
        self.nf = DTR.shape[1] - self.nt

        # print("nt: "+str(self.nt))
        # print("nf: "+str(self.nf))

        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size**2, order="F")
            return xxT

        n_features = DTR.shape[1]
        # Creare una matrice vuota per contenere le caratteristiche espans
        expanded_DTR = np.apply_along_axis(vecxxT, 0, DTR)
        # expanded_DTE = np.apply_along_axis(vecxxT, 0, Dte)
        self.fi_x = np.vstack([expanded_DTR, DTR])

        # phi_DTE = np.vstack([expanded_DTE, Dte])

        x0 = np.zeros(self.fi_x.shape[0] + 1)

        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])
        self.grad_funct = self.gradient_test(
            self.fi_x, self.LTR, self.l, self.piT, self.nt, self.nf
        )

        # print("sono dentro")

        # optimize.fmin_l_bfgs_b looks for secod order info to search direction pt and then find an acceptable step size at for pt
        # I set approx_grad=True so the function will generate an approximated gradient for each iteration
        params, f_min, _ = sc.optimize.fmin_l_bfgs_b(self.quad_logreg_obj, x0)
        # print("sono uscito")
        # the function found the coord for the minimal value of logreg_obj and they conrespond to w and b

        self.b = params[-1]

        self.w = np.array(params[0:-1])

        self.S = []

        return self.b, self.w

    def compute_scores(self, DTE):
        # print("In compute score")
        # I apply the model just trained to classify the test set samples
        S = self.S

        # def vecxxT(x):
        #     x = x[:, None]
        #     xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
        #     return xxT

        # expanded_DTE = np.apply_along_axis(vecxxT, 0, DTE)

        # phi_DTE = np.vstack([expanded_DTE, DTE])

        for i in range(DTE.shape[1]):
            x = vcol(DTE[:, i : i + 1])
            mat_x = np.dot(x, x.T)
            vec_x = vcol(np.hstack(mat_x))
            fi_x = np.vstack((vec_x, x))
            self.S.append(np.dot(self.w.T, fi_x) + self.b)

        pred = [
            1 if x > 0 else 0 for x in S
        ]  # I transform in 1 all the pos values and in 0 all the negative ones

        # acc = accuracy(S,LTE)

        # print(100-acc)

        return S


class LRCalibrClass:
    def __init__(self, l, piT):
        self.l = l
        self.DTR = []
        self.LTR = []
        # due of UNBALANCED classes (the spoofed has significantly more samples) we have to put a piT to apply this weigth
        self.piT = piT

    def logreg_obj(self, v):
        loss = 0
        loss_c0 = 0
        loss_c1 = 0
        # for each possible v value in the current iteration (which corresponds to specific coord
        # obtained by the just tracked movement plotted from the actual Hessian and Gradient values and the previous calculated coord)
        # we extrapolate the w and b parameters to insert in the J loss-function

        w, b = v[0:-1], v[-1]
        w = vcol(w)
        n = self.DTR.shape[1]

        regularization = (self.l / 2) * np.sum(w**2)

        # it's a sample way to apply the math transformation z = 2c - 1
        for i in range(n):

            if self.LTR[i : i + 1] == 1:
                zi = 1
                loss_c1 += np.logaddexp(
                    0, -zi * (np.dot(w.T, self.DTR[:, i : i + 1]) + b)
                )
            else:
                zi = -1
                loss_c0 += np.logaddexp(
                    0, -zi * (np.dot(w.T, self.DTR[:, i : i + 1]) + b)
                )

        J = (
            regularization
            + (self.piT / self.nT) * loss_c1
            + (1 - self.piT) / self.nF * loss_c0
        )

        return J

    def logistic_reg_calibration(self, DTR, LTR, DTE):
        self.DTR = DTR
        self.LTR = LTR
        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])
        _v, _J, _d = sc.optimize.fmin_l_bfgs_b(
            self.logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=True
        )
        _w = _v[0 : DTR.shape[0]]
        _b = _v[-1]
        calibration = np.log(self.piT / (1 - self.piT))
        self.b = _b
        self.w = _w
        STE = np.dot(_w.T, DTE) + _b - calibration
        return _w, _b

    def train(self, DTR, LTR):
        self.DTR = DTR
        self.LTR = LTR
        x0 = np.zeros(DTR.shape[0] + 1)

        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])

        # logRegObj = logRegClass(self.DTR, self.LTR, self.l) #I created an object logReg with logreg_obj inside

        # optimize.fmin_l_bfgs_b looks for secod order info to search direction pt and then find an acceptable step size at for pt
        # I set approx_grad=True so the function will generate an approximated gradient for each iteration
        params, f_min, _ = sc.optimize.fmin_l_bfgs_b(
            self.logreg_obj, x0, approx_grad=True
        )
        # the function found the coord for the minimal value of logreg_obj and they conrespond to w and b
        self.b = params[-1]

        self.w = np.array(params[0:-1])

        self.S = []

        return self.b, self.w

    def compute_scores(self, DTE):
        # I apply the model just trained to classify the test set samples
        S = self.S

        for i in range(DTE.shape[1]):
            x = DTE[:, i : i + 1]
            x = np.array(x)
            x = x.reshape((x.shape[0], 1))

            self.S.append(np.dot(self.w.T, x) + self.b)

        S = [
            1 if x > 0 else 0 for x in S
        ]  # I transform in 1 all the pos values and in 0 all the negative ones

        # acc = accuracy(S,LTE)

        # print(100-acc)
        llr = np.dot(self.w.T, DTE) + self.b

        return llr


def kfold_calib(D, L, classifier, options, calibrated=False):

    K = options["K"]
    pi = options["pi"]
    (cfn, cfp) = options["costs"]
    pca = options["pca"]
    znorm = options["znorm"]
    if calibrated == True:
        logObj = options["logCalibration"]

    samplesNumber = D.shape[1]
    N = int(samplesNumber / K)

    np.random.seed(seed=0)
    indexes = np.random.permutation(D.shape[1])
    actDCFtmp = 0
    scores = []
    labels = []
    scores_CAL = []

    for i in range(K):
        idxTest = indexes[i * N : (i + 1) * N]

        idxTrainLeft = indexes[0 : i * N]
        idxTrainRight = indexes[(i + 1) * N :]
        idxTrain = np.hstack([idxTrainLeft, idxTrainRight])

        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]

        if pca is not None:  # PCA needed
            DTR, P = PCA_impl(DTR, pca)
            DTE = np.dot(P.T, DTE)

        classifier.train(DTR, LTR)
        scores_i = classifier.compute_scores(DTE)

        scores = np.append(scores, scores_i)
        labels = np.append(labels, LTE)

    # plot the Bayes error BEFORE calibration
    labels = np.array(labels, dtype=int)
    DCF_effPrior, DCF_effPrior_min = Bayes_plot(scores, labels)

    # plot the ROC BEFORE calibration
    # post_prob = binary_posterior_prob(scores,pi,cfn,cfp)
    # thresholds = np.sort(post_prob)
    # ROC_plot(thresholds,post_prob,labels)

    DTRc = scores[: int(len(scores) * 0.7)]
    DTEc = scores[int(len(scores) * 0.7) :]
    LTRc = labels[: int(len(labels) * 0.7)]
    LTEc = labels[int(len(labels) * 0.7) :]
    estimated_w, estimated_b = logObj.logistic_reg_calibration(
        np.array([DTRc]), LTRc, np.array([DTEc])
    )
    print("estimated_w: " + str(estimated_w))
    print("estimated_b: " + str(estimated_b))
    # (DTRc,LTRc), (DTEc,LTEc)= split_db_2to1(scores, labels)
    # estimated_b, estimated_w = logObj.train(DTRc,LTRc)

    scores_append = scores.reshape((1, scores.size))
    final_score = np.dot(estimated_w.T, scores_append) + estimated_b
    print("final score: ")
    print(final_score)
    DCF_effPrior, DCF_effPrior_min = Bayes_plot(final_score, labels)
    return DCF_effPrior, DCF_effPrior_min, scores, labels


def best_logreg(logger):
    DTR, LTR = load(logger)
    prior, Cfp, Cfn = (0.5, 10, 1)
    l = 1e-2
    pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
    QuadLogReg = quadLogRegClass(l, pi_tilde)

    LogObj = LRCalibrClass(1e-2, 0.5)

    options = {
        "K": 5,
        "pi": 0.5,
        "pca": 6,
        "costs": (1, 10),
        "logCalibration": LogObj,
        "znorm": False,
    }

    DCF_effPrior, DCF_effPrior_min, lr_not_calibr_scores, lr_labels = kfold_calib(
        DTR, LTR, QuadLogReg, options, True
    )

    post_prob = binary_posterior_prob(scores, prior, Cfn, Cfp)
    thresholds = np.sort(post_prob)


class SVMClass:

    def __init__(self, K, C, piT):
        self.K = K
        self.C = C
        self.piT = piT
        self.Z = []
        self.D_ = []
        self.w_hat_star = []
        self.f = []

    def compute_lagrangian_wrapper(self, H):
        def compute_lagrangian(alpha):
            alpha = alpha.reshape(-1, 1)
            Ld_alpha = 0.5 * alpha.T @ H @ alpha - np.sum(alpha)
            gradient = H @ alpha - 1
            return Ld_alpha.item(), gradient.flatten()

        return compute_lagrangian

    def compute_H(self, D_, LTR):
        Z = np.where(LTR == 0, -1, 1).reshape(-1, 1)
        Gb = D_.T @ D_
        Hc = Z @ Z.T * Gb
        return Z, Hc

    def primal_solution(self, alpha, Z, x):
        return alpha * Z * x.T

    def compute_primal_objective(self, w_star, C, Z, D_):
        w_star = vcol(w_star)
        Z = vrow(Z)
        # print("w size: "+str(w_star.shape))
        # print("D size: "+str(D_.shape))
        # print("Z size: "+str(Z.shape))
        fun1 = 0.5 * (w_star * w_star).sum()
        fun2 = Z * np.dot(w_star.T, D_)
        fun3 = 1 - fun2
        zeros = np.zeros(fun3.shape)
        sommatoria = np.maximum(zeros, fun3)
        fun4 = np.sum(sommatoria)
        fun5 = C * fun4
        ris = fun1 + fun5
        return ris

    def train(self, DTR, LTR):
        # print("dentro train")
        nf = DTR[:, LTR == 0].shape[1]
        nt = DTR[:, LTR == 1].shape[1]
        emp_prior_f = nf / DTR.shape[1]
        emp_prior_t = nt / DTR.shape[1]
        Cf = self.C * self.piT / emp_prior_f
        Ct = self.C * self.piT / emp_prior_t

        K_row = np.ones((1, DTR.shape[1])) * self.K
        D_ = np.vstack((DTR, K_row))
        self.D_ = D_
        # print(D_)
        self.Z, H_ = self.compute_H(D_, LTR)
        compute_lag = self.compute_lagrangian_wrapper(H_)
        bound_list = [(-1, -1)] * DTR.shape[1]

        for i in range(DTR.shape[1]):
            if LTR[i] == 0:
                bound_list[i] = (0, Cf)
            else:
                bound_list[i] = (0, Ct)

        # factr=1 requires more iteration but returns more accurate results
        (alfa, self.f, d) = sc.optimize.fmin_l_bfgs_b(
            compute_lag,
            x0=np.zeros(LTR.size),
            approx_grad=False,
            factr=1.0,
            bounds=bound_list,
        )
        w_hat_star = (vcol(alfa) * self.Z * D_.T).sum(axis=0)
        self.w_hat_star = w_hat_star

    def compute_scores(self, DTE):
        # print("dentro comp_score")
        K_row2 = np.ones((1, DTE.shape[1])) * self.K
        D2_ = np.vstack((DTE, K_row2))
        score = np.dot(self.w_hat_star, D2_)

        predicted_labels = np.where(score > 0, 1, 0)
        # error = (predicted_labels != DTE).mean()
        # accuracy,error=compute_accuracy_error(predicted_labels,mRow(LTE))
        primal_obj = self.compute_primal_objective(
            self.w_hat_star, self.C, self.Z, self.D_
        )
        dual_gap = primal_obj + self.f
        # print("K:",self.K)
        # print("C:",self.C)
        # print("primal loss: ",primal_obj)
        # print("dual loss: ", -self.f)
        # print("dual gap: ",dual_gap)
        # print(f'Error rate: {error * 100:.1f}%')
        return score


def best_SVM(logger):
    DTR, LTR = load(logger)
    prior, Cfp, Cfn = (0.5, 10, 1)
    K_svm = 0
    C = 10
    mode = "rbf"
    gamma = 1e-3
    pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
    SVMObj = SVMClass(K_svm, C, pi_tilde, mode, gamma)

    LogObj = LRCalibrClass(1e-2, 0.5)

    options = {
        "K": 5,
        "pi": 0.5,
        "pca": 6,
        "costs": (1, 10),
        "logCalibration": LogObj,
        "znorm": False,
    }

    DCF_effPrior, DCF_effPrior_min, svm_not_calibr_scores, svm_labels = kfold_calib(
        DTR, LTR, SVMObj, options, True
    )
    post_prob = binary_posterior_prob(svm_not_calibr_scores, prior, Cfn, Cfp)
    thresholds = np.sort(post_prob)
    svm_FPR, svm_TPR = ROC_plot(thresholds, post_prob, svm_labels)


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


class GMMClass:

    # METTI TUTTI STI VALORI IN UN OPTIONS CHE E' MEGLIO
    def __init__(
        self,
        target_max_comp,
        not_target_max_comp,
        mode_target,
        mode_not_target,
        psi,
        alpha,
        prior,
        Cfp,
        Cfn,
    ):
        self.not_target_max_comp = not_target_max_comp  # number max components in LBG (after all the splits), the number of components will be n
        self.target_max_comp = target_max_comp  # number max components in LBG (after all the splits), the number of components will be n
        self.gmm0 = None
        self.gmm1 = None
        self.model = None
        self.mode_target = mode_target
        self.mode_not_target = mode_not_target
        self.psi = psi  # psi (lower bound for eigenvalues of the cov matrices in LBG)
        self.alpha = alpha  # alfa LBG factor
        # self.tiedness = tiedness        #boolean

        self.eff_prior = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
        # print("eff prior: "+str(self.eff_prior))

    # we train two different GMM (1 for the target class and one for the non target class)
    def train(self, DTR, LTR):
        D0 = DTR[:, LTR == 0]
        D1 = DTR[:, LTR == 1]
        model = []

        if self.mode_not_target == "full":
            self.gmm0 = self.LBG(D0, self.psi, self.alpha, self.not_target_max_comp)
        elif self.mode_not_target == "diag":
            self.gmm0 = self.Diag_LBG(
                D0, self.psi, self.alpha, self.not_target_max_comp
            )
        elif self.mode_not_target == "tied":
            self.gmm0 = self.TiedCov_LBG(
                D0, self.psi, self.alpha, self.not_target_max_comp
            )
        else:
            print("Error for non target mode")

        model.append(self.gmm0)

        if self.mode_target == "full":
            self.gmm1 = self.LBG(D1, self.psi, self.alpha, self.target_max_comp)
        elif self.mode_target == "diag":
            self.gmm1 = self.Diag_LBG(D1, self.psi, self.alpha, self.target_max_comp)
        elif self.mode_target == "tied":
            self.gmm1 = self.TiedCov_LBG(D1, self.psi, self.alpha, self.target_max_comp)
        else:
            print("Error for target mode")

        model.append(self.gmm1)

        self.model = model

        return model

    def compute_scores(self, DTE):
        llr = self.predict_log(self.model, DTE, [1 - self.eff_prior, self.eff_prior])
        return llr

    def EM_GMM(self, X, gmm, psi):  # gmm list of tuple
        while True:
            num_components = len(gmm)
            logS = np.empty((num_components, X.shape[1]))
            # ---START of E step---
            for component_index in range(num_components):
                logS[component_index, :] = logpdf_GAU_ND(
                    X, gmm[component_index][1], gmm[component_index][2]
                )
                logS[component_index, :] += np.log(
                    gmm[component_index][0]
                )  # computing the joint loglikelihood densities
            logSMarginal = vrow(sc.special.logsumexp(logS, axis=0))
            logSPost = logS - logSMarginal
            SPost = np.exp(logSPost)

            old_loss = logSMarginal
            # ---END of E step---
            # ---START of M step---
            Z = np.sum(SPost, axis=1)

            F = np.zeros((X.shape[0], num_components))
            for component_index in range(num_components):
                F[:, component_index] = np.sum(SPost[component_index, :] * X, axis=1)

            S = np.zeros((X.shape[0], X.shape[0], num_components))
            for component_index in range(num_components):
                S[:, :, component_index] = np.dot(SPost[component_index, :] * X, X.T)

            mu = F / Z
            cov_mat = S / Z

            for component_index in range(num_components):
                cov_mat[:, :, component_index] -= np.dot(
                    vcol(mu[:, component_index]), vrow(mu[:, component_index])
                )
                U, s, _ = np.linalg.svd(
                    cov_mat[:, :, component_index]
                )  # avoiding degenerate solutions
                s[s < psi] = psi
                cov_mat[:, :, component_index] = np.dot(U, vcol(s) * U.T)
            w = Z / np.sum(Z)

            gmm = [
                (
                    w[index_component],
                    vcol(mu[:, index_component]),
                    cov_mat[:, :, index_component],
                )
                for index_component in range(num_components)
            ]
            new_loss = logpdf_GMM(X, gmm)
            # ---END of M step---
            # if np.mean(new_loss)-np.mean(old_loss)>0:
            #     print("Bene")
            # if np.mean(new_loss)-np.mean(old_loss)<0:
            #     print("Male")

            if np.mean(new_loss) - np.mean(old_loss) < 1e-6:
                # print("EM terminato")
                break
        # print(np.mean(new_loss))
        return gmm

    def EM_Diag_GMM(self, X, gmm, psi):  # gmm list of tuple
        while True:
            num_components = len(gmm)
            logS = np.empty((num_components, X.shape[1]))
            # ---START of E step---
            for component_index in range(num_components):
                logS[component_index, :] = logpdf_GAU_ND(
                    X, gmm[component_index][1], gmm[component_index][2]
                )
                logS[component_index, :] += np.log(
                    gmm[component_index][0]
                )  # computing the joint loglikelihood densities
            logSMarginal = vrow(sc.special.logsumexp(logS, axis=0))
            logSPost = logS - logSMarginal
            SPost = np.exp(logSPost)

            old_loss = logSMarginal
            # ---END of E step---
            # ---START of M step---
            Z = np.sum(SPost, axis=1)

            F = np.zeros((X.shape[0], num_components))
            for component_index in range(num_components):
                F[:, component_index] = np.sum(SPost[component_index, :] * X, axis=1)

            S = np.zeros((X.shape[0], X.shape[0], num_components))
            for component_index in range(num_components):
                S[:, :, component_index] = np.dot(SPost[component_index, :] * X, X.T)

            mu = F / Z
            cov_mat = S / Z

            for component_index in range(num_components):
                cov_mat[:, :, component_index] -= np.dot(
                    vcol(mu[:, component_index]), vrow(mu[:, component_index])
                )

            w = Z / np.sum(Z)

            for index_component in range(num_components):
                cov_mat[:, :, index_component] = np.diag(
                    np.diag(cov_mat[:, :, index_component])
                )  # diag cov mat
                U, s, _ = np.linalg.svd(
                    cov_mat[:, :, index_component]
                )  # avoiding degenerate solutions
                s[s < psi] = psi
                cov_mat[:, :, index_component] = np.dot(U, vcol(s) * U.T)

            gmm = [
                (
                    w[index_component],
                    vcol(mu[:, index_component]),
                    cov_mat[:, :, index_component],
                )
                for index_component in range(num_components)
            ]
            new_loss = logpdf_GMM(X, gmm)
            # ---END of M step---
            # if np.mean(new_loss)-np.mean(old_loss)>0:
            #     print("BENE")
            # if np.mean(new_loss)-np.mean(old_loss)<0:
            #     print("MALE")
            if np.mean(new_loss) - np.mean(old_loss) < 1e-6:
                # print("EM terminato")
                break
        # print(np.mean(new_loss))
        return gmm

    def EM_TiedCov_GMM(self, X, gmm, psi):  # gmm list of tuple
        while True:
            num_components = len(gmm)
            logS = np.empty((num_components, X.shape[1]))
            # ---START of E step---
            for component_index in range(num_components):
                logS[component_index, :] = logpdf_GAU_ND(
                    X, gmm[component_index][1], gmm[component_index][2]
                )
                logS[component_index, :] += np.log(
                    gmm[component_index][0]
                )  # computing the joint loglikelihood densities
            logSMarginal = vrow(sc.special.logsumexp(logS, axis=0))
            logSPost = logS - logSMarginal
            SPost = np.exp(logSPost)

            old_loss = logSMarginal
            # ---END of E step---
            # ---START of M step---
            Z = np.sum(SPost, axis=1)

            F = np.zeros((X.shape[0], num_components))
            for component_index in range(num_components):
                F[:, component_index] = np.sum(SPost[component_index, :] * X, axis=1)

            S = np.zeros((X.shape[0], X.shape[0], num_components))
            for component_index in range(num_components):
                S[:, :, component_index] = np.dot(SPost[component_index, :] * X, X.T)

            mu = F / Z
            cov_mat = S / Z

            for component_index in range(num_components):
                cov_mat[:, :, component_index] -= np.dot(
                    vcol(mu[:, component_index]), vrow(mu[:, component_index])
                )
            w = Z / np.sum(Z)

            cov_updated = np.zeros(cov_mat[:, :, 0].shape)
            for component_index in range(num_components):
                cov_updated += (1 / num_samples(X)) * (
                    Z[component_index] * cov_mat[:, :, component_index]
                )

            U, s, _ = np.linalg.svd(cov_updated)  # avoiding degenerate solutions
            s[s < psi] = psi
            cov_updated = np.dot(U, vcol(s) * U.T)
            gmm = [
                (w[index_component], vcol(mu[:, index_component]), cov_updated)
                for index_component in range(num_components)
            ]

            new_loss = logpdf_GMM(X, gmm)
            # ---END of M step---
            # if np.mean(new_loss)-np.mean(old_loss)>0:
            #     # print("BENE")
            # if np.mean(new_loss)-np.mean(old_loss)<0:
            #     # print("MALE")
            if np.mean(new_loss) - np.mean(old_loss) < 1e-6:
                # print("EM terminato")
                break
        # print(np.mean(new_loss))
        return gmm

    def LBG(self, X, psi, alpha=0.1, num_components_max=4):
        w_start = 1
        mu_start, C_start = ML_estimate(X)

        U, s, _ = np.linalg.svd(C_start)  # avoiding degenerate solutions
        s[s < psi] = psi
        C_start = np.dot(U, vcol(s) * U.T)

        gmm_start = []
        if num_components_max == 1:
            gmm_start.append((w_start, vcol(mu_start), C_start))
            return gmm_start
        new_w = w_start / 2
        U, s, Vh = np.linalg.svd(C_start)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        mu_new_1 = mu_start + d
        mu_new_2 = mu_start - d
        gmm_start.append((new_w, vcol(mu_new_1), C_start))
        gmm_start.append((new_w, vcol(mu_new_2), C_start))

        while True:
            gmm_start = self.EM_GMM(X, gmm_start, psi)
            num_components = len(gmm_start)

            if num_components == num_components_max:
                break
            new_gmm = []
            for index_component in range(num_components):
                new_w = gmm_start[index_component][0] / 2
                U, s, Vh = np.linalg.svd(gmm_start[index_component][2])
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                mu_new_1 = gmm_start[index_component][1] + d
                mu_new_2 = gmm_start[index_component][1] - d
                new_gmm.append((new_w, vcol(mu_new_1), gmm_start[index_component][2]))
                new_gmm.append((new_w, vcol(mu_new_2), gmm_start[index_component][2]))
            gmm_start = new_gmm
        return gmm_start

    def Diag_LBG(self, X, psi, alpha=0.1, num_components_max=4):
        w_start = 1
        mu_start, C_start = ML_estimate(X)

        C_start = np.diag(np.diag(C_start))

        U, s, _ = np.linalg.svd(C_start)  # avoiding degenerate solutions
        s[s < psi] = psi
        C_start = np.dot(U, vcol(s) * U.T)

        gmm_start = []

        if num_components_max == 1:
            gmm_start.append((w_start, vcol(mu_start), C_start))
            return gmm_start

        new_w = w_start / 2
        U, s, Vh = np.linalg.svd(C_start)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        mu_new_1 = mu_start + d
        mu_new_2 = mu_start - d
        gmm_start.append((new_w, vcol(mu_new_1), C_start))
        gmm_start.append((new_w, vcol(mu_new_2), C_start))

        while True:
            gmm_start = self.EM_Diag_GMM(X, gmm_start, psi)
            num_components = len(gmm_start)
            if num_components == num_components_max:
                break
            new_gmm = []
            for index_component in range(num_components):
                new_w = gmm_start[index_component][0] / 2
                U, s, Vh = np.linalg.svd(gmm_start[index_component][2])
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                mu_new_1 = gmm_start[index_component][1] + d
                mu_new_2 = gmm_start[index_component][1] - d
                new_gmm.append((new_w, vcol(mu_new_1), gmm_start[index_component][2]))
                new_gmm.append((new_w, vcol(mu_new_2), gmm_start[index_component][2]))
            gmm_start = new_gmm
            # print("GMMMM START....")
            # print(len(gmm_start))
        return gmm_start

    def TiedCov_LBG(self, X, psi, alpha=0.1, num_components_max=4):
        w_start = 1
        mu_start, C_start = ML_estimate(X)

        U, s, _ = np.linalg.svd(C_start)  # avoiding degenerate solutions
        s[s < psi] = psi
        C_start = np.dot(U, vcol(s) * U.T)

        gmm_start = []

        if num_components_max == 1:
            gmm_start.append((w_start, vcol(mu_start), C_start))
            return gmm_start

        new_w = w_start / 2
        U, s, Vh = np.linalg.svd(C_start)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        mu_new_1 = mu_start + d
        mu_new_2 = mu_start - d
        gmm_start.append((new_w, vcol(mu_new_1), C_start))
        gmm_start.append((new_w, vcol(mu_new_2), C_start))
        while True:
            gmm_start = self.EM_TiedCov_GMM(X, gmm_start, psi)
            num_components = len(gmm_start)

            if num_components == num_components_max:
                break
            new_gmm = []
            for index_component in range(num_components):
                new_w = gmm_start[index_component][0] / 2
                U, s, Vh = np.linalg.svd(gmm_start[index_component][2])
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                mu_new_1 = gmm_start[index_component][1] + d
                mu_new_2 = gmm_start[index_component][1] - d
                new_gmm.append((new_w, vcol(mu_new_1), gmm_start[index_component][2]))
                new_gmm.append((new_w, vcol(mu_new_2), gmm_start[index_component][2]))
            gmm_start = new_gmm
        return gmm_start

    def predict_log(
        self, model, test_samples, prior
    ):  # use this function to avoid numerical problems
        num_classes = len(model)
        # print("Len model: "+str(len(model)))
        likelihoods = []
        logSJoint = []
        # class_posterior_probs = []
        for index_class in range(num_classes):
            gmm_c = model[index_class]
            likelihood_c = logpdf_GMM(test_samples, gmm_c)
            likelihoods.append(likelihood_c)
            logSJoint.append(likelihood_c + np.log(prior[index_class]))

        logSMarginal = vrow(sc.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        llr = logSPost[1, :] - logSPost[0, :] - np.log(prior[1] / prior[0])
        # SPost = np.exp(logSPost)

        # predicted_label = np.argmax(np.array(SPost),axis=0)

        return llr  # ravel per appiattire il vettore e farlo combaciare con labels per calcolo accuracy


def best_GMM(logger):
    prior, Cfp, Cfn = (0.5, 10, 1)
    target_max_comp = 2
    not_target_max_comp = 8
    mode_target = "diag"
    mode_not_target = "diag"
    psi = 0.01
    alpha = 0.1
    pca = None
    pi_tilde = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)

    GMMObj = GMMClass(
        target_max_comp,
        not_target_max_comp,
        mode_target,
        mode_not_target,
        psi,
        alpha,
        prior,
        Cfp,
        Cfn,
    )

    LogObj = LRCalibrClass(1e-2, 0.5)

    options = {
        "K": 5,
        "pi": 0.5,
        "pca": pca,
        "costs": (1, 10),
        "logCalibration": LogObj,
        "znorm": False,
    }

    DCF_effPrior, DCF_effPrior_min, gmm_not_calibr_scores, gmm_labels = kfold_calib(
        DTR, LTR, GMMObj, options, True
    )

    post_prob = binary_posterior_prob(gmm_not_calibr_scores, prior, Cfn, Cfp)
    thresholds = np.sort(post_prob)
    gmm_FPR, gmm_TPR = ROC_plot(thresholds, post_prob, gmm_labels)


def main(args, logger):
    logger.info("starting on Evaluation")
    logger.info("-----------------------------------------------")
    logger.info("Starting Evaluation for Logistic Regression")
    best_logreg(logger)
    logger.info("-----------------------------------------------")
    logger.info("Starting Evaluation for SVM")
    best_SVM(logger)
    logger.info("-----------------------------------------------")
    logger.info("Starting Evaluation for GMM")
    best_GMM(logger)
    logger.info("-----------------------------------------------")
