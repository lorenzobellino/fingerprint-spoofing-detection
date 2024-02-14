import numpy as np
import scipy as sc
import scipy.optimize as opt
from prettytable import PrettyTable
from Utils.utils import (
    load,
    PCA_projection,
    compute_min_DCF,
    write_data,
    PRJCT_ROOT,
    znorm,
)
from matplotlib import pyplot as plt
from GMM.models import *


def main(args, logger):
    logger.info("starting on Gaussian Mixture Model")
    D, L = load(logger)

    K = 5
    prior = 0.5
    Cfp = 10
    Cfn = 1
    gmm_pca6_glob = {}
    gmm_pca7_glob = {}
    gmm_pca8_glob = {}
    gmm_pca9_glob = {}
    gmm_pcaNone_glob = {}

    for mode_target in ["diag", "tied"]:
        for mode_not_target in ["full", "diag", "tied"]:
            gmm_pca6 = []
            gmm_pca7 = []
            gmm_pca8 = []
            gmm_pca9 = []
            gmm_pcaNone = []
            for pca in [6, 8, None]:
                for t_max_n in [1, 2]:
                    gmm_tmp = []
                    for nt_max_n in [2, 4, 8]:
                        for znorm in [False]:
                            alfa = 0.1
                            psi = 0.01
                            options = {
                                "K": 5,
                                "pca": pca,
                                "pi": 0.5,
                                "costs": (1, 10),
                                "znorm": znorm,
                            }

                            GMMObj = GMMClass(
                                t_max_n,
                                nt_max_n,
                                mode_target,
                                mode_not_target,
                                psi,
                                alfa,
                                prior,
                                Cfp,
                                Cfn,
                            )
                            min_DCF, scores, labels = kfold(D, L, GMMObj, options)
                            # if min_DCF > 1:
                            #     min_DCF = 1

                            print(
                                f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}"
                            )

                            # un vettore che si annulla ad ogni nuovo t_max_n
                            gmm_tmp.append(min_DCF)
                            if pca == 6:
                                # if znorm==True:

                                gmm_pca6.append(min_DCF)
                                gmm_pca6_glob.setdefault(
                                    (
                                        f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",
                                        min_DCF,
                                    )
                                )

                                # else:
                                #     gmm_pca6_noznorm.append(min_DCF)

                            # if pca == 7:
                            #     gmm_pca7.append(min_DCF)
                            #     gmm_pca7_glob.setdefault((f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",min_DCF))

                            if pca == 8:
                                gmm_pca8.append(min_DCF)
                                gmm_pca8_glob.setdefault(
                                    (
                                        f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",
                                        min_DCF,
                                    )
                                )

                            #     if kernel=="poly" :
                            #         poly_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
                            #     else:
                            #         rbf_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
                            # if pca == 9:
                            #     gmm_pca9.append(min_DCF)
                            #     gmm_pca9_glob.setdefault((f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",min_DCF))

                            if pca == None:
                                # if znorm==True:
                                gmm_pcaNone.append(min_DCF)
                                gmm_pcaNone_glob.setdefault(
                                    (
                                        f"GMM min_DCF mode_target={mode_target} e mode_not_target={mode_not_target} con K = {K} , nt_max_n={nt_max_n} t_max_n={t_max_n} pca = {pca}: {min_DCF} znorm: {znorm}",
                                        min_DCF,
                                    )
                                )

                                # else:
                                #     gmm_pcaNone_noznorm.append(min_DCF)

                    fig = plt.figure()
                    plt.plot([2, 4, 8], gmm_tmp)
                    plt.xlabel("nt_max_n")
                    plt.ylabel("DCF_min")
                    titolo = f"mode_target: {mode_target}, mode_non_target:{mode_not_target} PCA: {pca}, t_max_n: {t_max_n}"
                    plt.title(titolo)
                    plt.show()

                # Creazione del grafico
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

                # Aggiunta dei dati al grafico
                ax.scatter(t_max_n, min_DCF, nt_max_n)

                # Impostazione delle etichette degli assi
                ax.set_xlabel("t_max_n")
                ax.set_ylabel("DCF_min")
                ax.set_zlabel("nt_max_n")
                name_graph = "GMM " + mode_target + " " + mode_not_target
                plt.title(name_graph)
                plt.show()
                # plt.semilogx(C_values,svm_pca6, label = "PCA 6")
                # #plt.semilogx(C_values,svm_pca7, label = "PCA 7")
                # plt.semilogx(C_values,svm_pca6_noznorm, label = "PCA 6 No Znorm")
                # #plt.semilogx(C_values,svm_pca9, label = "PCA 9")
                # plt.semilogx(C_values,svm_pcaNone, label = "No PCA")
                # plt.semilogx(C_values,svm_pcaNone_noznorm, label = "No PCA No Znorm")

                # plt.xlabel("C")
                # plt.ylabel("DCF_min")
                # plt.legend()
                # # if piT == 0.1:
                # #     path = "plots/svm/DCF_su_C_piT_min"
                # # if piT == 0.33:
                # #     path = "plots/svm/DCF_su_C_piT_033"
                # # if piT == 0.5:
                # #     path = "plots/svm/DCF_su_C_piT_medium"
                # # if piT == 0.9:
                # #     path = "plots/svm/DCF_su_C_piT_max"
                # title=str(piT)+" "+str(kernel)+" "+str(gamma)

                # # plt.savefig(path)
                # plt.show()
