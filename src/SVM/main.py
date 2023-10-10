import numpy as np
import scipy as sc
import scipy.optimize as opt
from prettytable import PrettyTable
from Utils.utils import load, PCA_projection,compute_min_DCF,write_data,PRJCT_ROOT,znorm
from matplotlib import pyplot as plt
from SVM.model import SVM_linear, SVM_Poly,SVM_RBF


def poly_SVM(D,L, args,logger):
    logger.info(f"plynomial SVM")
    # folds = 5 # can't use K for K-fold because in SVM K is already used
    N = int(D.shape[1]/args.k)
    PCA = [6,7,0] #number of dimension to keep in PCA
    zlist = [False]
    np.random.seed(0)
    indexes = np.random.permutation(D.shape[1])  

    C_list = np.logspace(-4,1,6).tolist() # from 10^-4 to 10
    K_list = np.logspace(-1,1,3).tolist() # from 10^-1 to 10
    hyper_list = [(2,0), (2,1)] # tuples of hyperparameters (d,c)

    # working points
    Cfn = 1
    Cfp = 1
    pi_list = [0.1, 0.5]




    # produce a graph and a result table for each K,c,d: on x plot different C used for training, on Y plot relative Cprim obtained
    for K_num, K in enumerate(K_list):
        
        for d,c in hyper_list:
            # st = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            # print(f"### {st}: Starting SVM Poly with K = {K}, d={d},c={c}") #feedback print
            
            #set table
            results = PrettyTable()
            results.align = "c"
            results.field_names = ["K", "C", "Kernel", "PCA", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]
            
            #set graph
            fig, ax = plt.subplots() 
            ax.set_xscale('log')
            ax.set(xlabel='C', ylabel='Cprim', title=f'SVM K={K} - Polynomial Kernel (d={d},c={c})')
            plt.grid(True)
            plt.xticks(C_list)

            for zn in zlist:
                for PCA_m in PCA:
                    Cprim_list = np.array([])
                    for C in C_list:
                    #for each C compute minDCF after K-fold
                                
                        scores_pool = np.array([])
                        labels_pool = np.array([])
                        
                        for i in range(args.k):
                        
                            idxTest = indexes[i*N:(i+1)*N]
                        
                            if i > 0:
                                idxTrainLeft = indexes[0:i*N]
                            elif (i+1) < args.k:
                                idxTrainRight = indexes[(i+1)*N:]
                        
                            if i == 0:
                                idxTrain = idxTrainRight
                            elif i == args.k-1:
                                idxTrain = idxTrainLeft
                            else:
                                idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
                        
                            
                            DTR = D[:, idxTrain]
                            DTE = D[:,idxTest]
                            if zn == True:
                                DTR,DTE = znorm(DTR,DTE)
                            if PCA_m > 0:
                                DTR,DTE = PCA_projection(D[:, idxTrain],D[:, idxTest],PCA_m)
                            
                            # if PCA_m != None:
                            #     DTR,P = PCA(DTR,PCA_m) # fit PCA to training set
                            LTR = L[idxTrain]
                            # if PCA_m != None:
                            #     DTE = np.dot(P.T,D[:, idxTest]) # transform test samples according to P from PCA on dataset
                            # else:
                            #     DTE = D[:,idxTest]
                            LTE = L[idxTest]
                            
                            #pool test scores and test labels in order to compute the minDCF on complete pool set
                            labels_pool = np.hstack((labels_pool,LTE))
                            scores_pool = np.hstack((scores_pool,SVM_Poly(DTR, LTR, DTE, C, K, d, c)))
                        
                        #compute minDCF for the current SVM with/without PCA for the 2 working points  
                        minDCF = np.zeros(2)
                        for i, pi in enumerate(pi_list):
                            minDCF[i] = compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
                        #compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
                        Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
                        Cprim_list = np.hstack((Cprim_list,Cprim))
                        # add current result to table
                        results.add_row([K, C, f"Poly(d={d},c={c})", PCA_m, np.round(minDCF[0],3), np.round(minDCF[1],3), Cprim])
                        logger.info(f"\t...computed C={C}, PCA={PCA_m}") #feedback print
                        
                    #plot the graph
                    ax.plot(C_list, Cprim_list, label =f'PCA-{PCA_m}')
                    logger.info(f"\tCprim values for PCA={PCA_m}: {Cprim_list}") #feedback print         
            
            logger.info(f'Completed SVM Poly with K = {K} and d={d},c={c} ###') #feedback print
            plt.legend()
            fig.savefig(f"{PRJCT_ROOT}/plots/SVM/SVM_Poly_K{K_num}d{d}_c{c}.png", dpi=200)
            logger.info("Plot saved!")
            plt.show()
            
            # print and save as txt the results table for each K,c,d combination
            logger.info(results)
            data = results.get_string()
            write_data(data, f'SVM_Poly_K{K_num}d{d}c{c}_ResultsTable.txt')
            # with open(f'Results/minDCF_SVM_Poly_results/SVM_Poly_K{K_num}d{d}c{c}_ResultsTable.txt', 'w') as file:
            #     file.write(data)


def RBF_SVM(D,L, args,logger):
    logger.info(f"RBF SVM") 
    N = int(D.shape[1]/args.k)
    PCA = [6,7,0] #number of dimension to keep in PCA
    zlist = [False]

    np.random.seed(0)
    indexes = np.random.permutation(D.shape[1])  

    C_list = np.logspace(-4,1,6).tolist() 
    K_list = np.logspace(-2,1,4).tolist() 
    gamma_list = np.logspace(-3,0,4).tolist() 

    K_list = [1,10]

    # working points
    Cfn = 1
    Cfp = 1
    pi_list = [0.1, 0.5]

    results = PrettyTable()
    results.align = "c"
    results.field_names = ["K", "C", "PCA", "Kernel", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]

    for K_num, K in enumerate(K_list):
        K_num += 2
        for zn in zlist:
            for PCA_m in PCA:
                fig, ax = plt.subplots() 
                ax.set_xscale('log')
                ax.set(xlabel='C', ylabel='Cprim', title=f'SVM K={K} - RBF Kernel')
                plt.grid(True)
                plt.xticks(C_list)
            
                for gamma in gamma_list:    
                    Cprim_list = np.array([])
                    for C in C_list:
                        scores_pool = np.array([])
                        labels_pool = np.array([])
                        
                        for i in range(args.k):
                        
                            idxTest = indexes[i*N:(i+1)*N]
                        
                            if i > 0:
                                idxTrainLeft = indexes[0:i*N]
                            elif (i+1) < args.k:
                                idxTrainRight = indexes[(i+1)*N:]
                        
                            if i == 0:
                                idxTrain = idxTrainRight
                            elif i == args.k-1:
                                idxTrain = idxTrainLeft
                            else:
                                idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
                        
                            DTR = D[:, idxTrain]
                            DTE = D[:,idxTest]

                            if zn == True:
                                DTR,DTE = znorm(DTR,DTE)
                            if PCA_m > 0:
                                DTR,DTE = PCA_projection(D[:, idxTrain],D[:, idxTest],PCA_m)
                            

                            LTR = L[idxTrain]
                            LTE = L[idxTest]
                            
                            #pool test scores and test labels in order to compute the minDCF on complete pool set
                            labels_pool = np.hstack((labels_pool,LTE))
                            scores_pool = np.hstack((scores_pool,SVM_RBF(DTR, LTR, DTE, C, K, gamma)))
                        
                        #compute minDCF for the current SVM with/without PCA for the 2 working points  
                        minDCF = np.zeros(2)
                        for i, pi in enumerate(pi_list):
                            minDCF[i] = compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
                        #compute Cprim (the 2 working points average minDCF) and save it in a list for plotting after
                        Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
                        Cprim_list = np.hstack((Cprim_list,Cprim))
                        # add current result to table
                        results.add_row([K, C, PCA_m, f"RBF(γ = {gamma}) ", np.round(minDCF[0],3), np.round(minDCF[1],3), Cprim])
                        logger.info(f"\t...computed C={C}, γ={gamma}") #feedback print
                        
                    #plot the graph
                    ax.plot(C_list, Cprim_list, label =f'log(gamma)={np.log10(gamma)}')
                    logger.info(f"\tCprim values for γ={gamma}: {Cprim_list}") #feedback print         
            
                logger.info(f'Completed SVM RBF with K = {K} and PCA = {PCA_m} ###') #feedback print
                plt.legend()
                fig.savefig(PRJCT_ROOT+f"/plots/SVM/SVM_RBF_K{K_num}_PCA{PCA_m}.png", dpi=200)
                # plt.show()
            
    # print and save as txt the final results table
    logger.info(results)
    data = results.get_string()

    # with open('Results/SVM_RBF_ResultsTable.txt', 'w') as file:
    #     file.write(data)   
    write_data(data,f"SVM_RBF_ResultsTable.txt")
    logger.info(f"{data}")   


def linear_SVM(D,L, args,logger):
    N = int(D.shape[1]/args.k)
    # PCA = [6,7,0] 
    PCA = [6,0]

    z_list = [True]

    np.random.seed(42)
    indexes = np.random.permutation(D.shape[1])  

    C_list = np.logspace(-3,2,6).tolist() 
    # K_list = np.logspace(-2,1,4).tolist()
    K_list = [0.1,1]

    Cfn = 1
    Cfp = 1
    pi_list = [0.1, 0.5]

    results = PrettyTable()
    results.align = "c"
    results.field_names = ["K", "C", "PCA", "minDCF (pi = 0.1)", "minDCF (pi = 0.5)", "Cprim"]


    for K_num, K in enumerate(K_list):
        fig, ax = plt.subplots() 
        ax.set_xscale('log')
        ax.set(xlabel='C', ylabel='Cprim', title=f'Linear SVM K={K}')
        plt.grid(True)
        plt.xticks(C_list)
        
        for zn in z_list:
            for PCA_m in PCA:
                Cprim_list = np.array([])
                for C in C_list:
                    scores_pool = np.array([])
                    labels_pool = np.array([])
                    
                    for i in range(args.k):
                    
                        idxTest = indexes[i*N:(i+1)*N]
                    
                        if i > 0:
                            idxTrainLeft = indexes[0:i*N]
                        elif (i+1) < args.k:
                            idxTrainRight = indexes[(i+1)*N:]
                    
                        if i == 0:
                            idxTrain = idxTrainRight
                        elif i == args.k-1:
                            idxTrain = idxTrainLeft
                        else:
                            idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
                    
                        DTR = D[:, idxTrain]
                        DTE = D[:,idxTest]

                        if zn == True:
                            DTR,DTE = znorm(DTR,DTE)
                        if PCA_m > 0:
                            DTR,DTE = PCA_projection(D[:, idxTrain],D[:, idxTest],PCA_m)
                        
                        LTR = L[idxTrain]
                        LTE = L[idxTest]
                        
                        labels_pool = np.hstack((labels_pool,LTE))
                        scores_pool = np.hstack((scores_pool,SVM_linear(DTR, LTR, DTE, C, K)))
                    
                    minDCF = np.zeros(2)
                    for i, pi in enumerate(pi_list):
                        minDCF[i] = compute_min_DCF(scores_pool, labels_pool, pi, Cfn, Cfp)
                    Cprim = np.round((minDCF[0] + minDCF[1])/ 2 , 3)
                    Cprim_list = np.hstack((Cprim_list,Cprim))
                    results.add_row([K, C, PCA_m, np.round(minDCF[0],3), np.round(minDCF[1],3), Cprim])
                    logger.info(f"\t...computed C={C}, PCA={PCA_m}") #feedback print
                ax.plot(C_list, Cprim_list, label =f'PCA-{PCA_m}')
                logger.info(f"\tCprim values for PCA={PCA_m}: {Cprim_list}") #feedback print         
            
        logger.info(f"Completed SVM linear with K = {K}") 
        plt.legend()
        fig.savefig(f"{PRJCT_ROOT}/plots/SVM/SVM_linear_{K_num}.png", dpi=200)
        # plt.show()

    logger.info(results)
    data = results.get_string()
    write_data(data,f"SVM_linear_{K_num}_result.txt")
    logger.info(f"{data}")

def main(args,logger):
    logger.info("starting on Support Vector Machine")
    D,L = load(logger)
    # linear_SVM(D,L,args,logger)
    logger.info(f"_"*60)
    # poly_SVM(D,L,args,logger)
    logger.info(f"_"*60)
    RBF_SVM(D,L,args,logger)
    # logger.info(f"_"*60)


# def quadratic_SVM(DTR,LTR, args,logger):
#     logger.info(f"quadratic SVM")
#     K=  5
#     piT = 0.1
#     poly_svm_pca6={}
#     poly_svm_pca8={}
#     poly_svm_pcaNone={}
#     rbf_svm_pca6 = {}
#     rbf_svm_pca8 = {}
#     rbf_svm_pcaNone = {}
#     for piT in [0.1]:
#         for kernel in ["poly"]:
#             if kernel=="poly":
#                 ci=[0,1]
#                 string="d=2 c= "
#             else:
#                 ci=[0.01,0.001,0.0001]
#                 string="gamma= "
#             for value in ci:  
#                 svm_pca6 = []
#                 svm_pca6_noznorm = []
#                 svm_pcaNone = []
#                 svm_pcaNone_noznorm = []
#                 #svm_pcaNone = []
#                 C_values = np.logspace(-3, -1, num=3)
#                 for C in np.logspace(-3, -1, num=3):
#                     for K_svm in [1]:
#                     #we saw that piT=0.1 is the best value
#                         for pca in [6,None]:
#                             for znorm in [False,True]:
#                                 options={"K":5,
#                                           "pca":pca,
#                                           "pi":0.5,
#                                           "costs":(1,10),
#                                           "znorm":znorm}
#                                 SVMObj = SVMClass(K_svm, C, piT,kernel,value)
#                                 min_DCF, scores, labels = kfold(DTR, LTR,SVMObj,options)
#                                 if min_DCF > 1: 
#                                     min_DCF = 1
                                
#                                 print(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} znorm: {znorm}")
                                
#                                 if pca == 6:
#                                     if znorm==True:
#                                         svm_pca6.append(min_DCF)
#                                         if kernel=="poly" :
#                                             poly_svm_pca6.setdefault(f"SVM min_DCF kernel={kernel} ({string} {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} Znorm",min_DCF)
#                                 #     else:
#                                 #         rbf_svm_pca6.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} no Znorm",min_DCF)
#                                     else:
#                                         svm_pca6_noznorm.append(min_DCF)
#                                         if kernel=="poly" :
#                                             poly_svm_pca6.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} no Znorm",min_DCF)
#                                     # else:
#                                     #     rbf_svm_pca6.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} Znorm",min_DCF)
                                    
#                                 # if pca == 7: 
#                                 #     svm_pca7.append(min_DCF)
#                                 # if pca == 8:
#                                 #     svm_pca8.append(min_DCF)
#                                 #     if kernel=="poly" :
#                                 #         poly_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
#                                 #     else:
#                                 #         rbf_svm_pca8.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
#                                 # # if pca == 9:
#                                 # #     svm_pca9.append(min_DCF)
#                                 if pca == None:
#                                     if znorm==True:
#                                         svm_pcaNone.append(min_DCF)
#                                         if kernel=="poly" :
#                                             poly_svm_pcaNone.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} Znorm",min_DCF)
#                                         # else:
#                                         #     rbf_svm_pcaNone.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ",min_DCF)
#                                     else:
#                                         svm_pcaNone_noznorm.append(min_DCF)
#                                         if kernel=="poly" :
#                                             poly_svm_pcaNone.setdefault(f"SVM min_DCF kernel={kernel} ({string } {value}) con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} no Znorm",min_DCF)
                                        
#                 plt.semilogx(C_values,svm_pca6, label = "PCA 6")
#                 #plt.semilogx(C_values,svm_pca7, label = "PCA 7")
#                 plt.semilogx(C_values,svm_pca6_noznorm, label = "PCA 6 No Znorm")
#                 #plt.semilogx(C_values,svm_pca9, label = "PCA 9")
#                 plt.semilogx(C_values,svm_pcaNone, label = "No PCA")
#                 plt.semilogx(C_values,svm_pcaNone_noznorm, label = "No PCA No Znorm")
                    
#                 plt.xlabel("C")
#                 plt.ylabel("DCF_min")
#                 plt.legend()
#                 # if piT == 0.1:
#                 #     path = "plots/svm/DCF_su_C_piT_min"
#                 # if piT == 0.33:
#                 #     path = "plots/svm/DCF_su_C_piT_033"
#                 # if piT == 0.5:
#                 #     path = "plots/svm/DCF_su_C_piT_medium"
#                 # if piT == 0.9:
#                 #     path = "plots/svm/DCF_su_C_piT_max"
#                 if kernel=="rbf":
#                     gamma=" gamma : "+str(value)
#                 else:
#                     gamma=" ci: " +str(value)
                    
#                 title=str(piT)+" "+str(kernel)+" "+str(gamma)
#                 plt.title(title)
#                 # plt.savefig(path)
#                 plt.show()# K=  5


# def linear_SVM(DTR,LTR, args,logger):
#     logger.info(f"linear SVM")
#     K=  5
#     piT = 0.1
#     for piT in [0.1]:
#         C_values = np.logspace(-5, 2, num=8)
#         point = (0.5,1,10)
#         svm = dict()
#         for C in np.logspace(-5, 2, num=8):
#             for K_svm in [1]:
#                 for zscore in [True,False]:
#                     for pca in [6]:
#                         args.znorm = zscore
#                         args.pca = pca
#                         SVMObj = SVMClass(K_svm, C, piT)
#                         min_DCF, scores, labels = KFCV(DTR, LTR,SVMObj,point,args,logger)
#                         if min_DCF > 1: 
#                             min_DCF = 1
#                         logger.info(f"SVM min_DCF con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}, pi={pi}, znorm={zscore}: {min_DCF} ")
#                         try:
#                             svm[f"pca-{pca}_z-{zscore}"].append(min_DCF)
#                         except:
#                             svm[f"pca-{pca}_z-{zscore}"] = [min_DCF]
                        
#         logger.info(f"plotting")
#         plot_linear_svm(C_values,piT,svm,f"SVM_linear_DCF_C_piT_{piT}.png",args)                
