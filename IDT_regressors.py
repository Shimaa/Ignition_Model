############################
# @author Shimaa Gamil
# KAUST CCRC 
################################

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from scipy.spatial import distance 
import random
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import time
import csv

#from sklearn import cross_validation

# KNN
def runKNN (X_train, y_train, X_test, y_test, K):
    k_best = K[1]
    mse_best = 10000
    r2_best = 100
    
    for k in K:
        knn = KNeighborsRegressor (n_neighbors=k, weights='distance')

        knn.fit (X_train, y_train)

        mse = mean_squared_error(y_test, knn.predict (X_test))
        r2 = r2_score(y_test, knn.predict (X_test))
        print ('\nKNN with k=:', k)
        print ('MSE = ' , mse)
        print('R^2: ' , r2)

        if mse < mse_best:
            k_best = k
            mse_best = mse
            r2_best = r2

    return knn, mse_best, r2_best, k_best


# Ensebmle
def runEnsemble_GB (X_train, y_train, X_test, y_test, n_estimators):
    n_best = n_estimators[0]
    mse_best = 10
    r2_best = 100
    best_model = 0

    original_params = {'n_estimators': 20000, 'max_leaf_nodes': 10, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 100}


    for setting in [{'learning_rate': 1.0, 'subsample': 1.0},
                  #  {'learning_rate': 0.1, 'subsample': 1.0},
                  #  {'learning_rate': 0.01, 'subsample': 1.0},
                   # {'learning_rate': 0.2, 'subsample': 1.0},
                    #{'learning_rate': 1.0, 'subsample': 0.5},
                   # {'learning_rate': 0.1, 'subsample': 0.5},
                    {'learning_rate': 1.0, 'max_features': 1, 'subsample': 1.0}]:
        
        params = dict(original_params)
        params.update(setting)

        '''
        for n in n_estimators:
            params = {'n_estimators': n, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        ''' 
        en = ensemble.GradientBoostingRegressor(**params)
        en.fit(X_train, y_train)

        mse = mean_squared_error(y_test, en.predict(X_test))
        r2 = r2_score(y_test, en.predict(X_test))
        #print ('\nEnsemble with n_estimators=', n)
        print ('\nEnsemble with setting=', setting)
        print ('MSE = ' , mse)
        print('R^2: ' , r2)

        if mse < mse_best:
            #n_best = n
            mse_best = mse
            r2_best = r2
            best_model = en

    return best_model, mse_best, r2_best, n_best


# Gradient
def runGB (X_train, y_train, X_test, y_test):
    #params = {'n_estimators': 5, 'learning_rate': 0.1,
    # 'max_depth': 1, 'random_state': 0, 'loss': 'ls'}

    est = GradientBoostingRegressor()
    est.fit(X_train, y_train)
    #predictions = est.predict (X_test)
    mse = mean_squared_error(y_test, est.predict(X_test))
    r2 = r2_score(y_test, est.predict(X_test))
    print ('\nGradient:')
    print ('MSE = ' , mse)
    print('R^2: ' , r2)

    return est, mse, r2


# MLP
def run_MLP (X_train, y_train, X_test, y_test, hl_sizes=(7,)):
    MLP = MLPRegressor(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=hl_sizes, learning_rate='constant',
           learning_rate_init=0.01, max_iter=200, momentum=0.8,
           nesterovs_momentum=True, power_t=0.5, random_state=7, shuffle=True,
           solver='adam', tol=0.000001, validation_fraction=0.1, verbose=True,
           warm_start=False)
    MLP.fit(X_train, y_train);

    mse = mean_squared_error(y_test, MLP.predict(X_test))
    r2 = r2_score(y_test, MLP.predict(X_test))
    print ('\nMLP:')
    print ('MSE = ' , mse)
    print('R^2: ' , r2)

    return MLP, mse, r2


# Decision Tree
def runDT (X_train, y_train, X_test, y_test, depth_list):
    depth_best = depth_list[0]
    mse_best = 10000
    r2_best = 100
    best_model = 0
    
    for depth in depth_list:
        DT = tree.DecisionTreeRegressor(max_depth=depth)
        DT.fit(X_train, y_train)

        mse = mean_squared_error(y_test, DT.predict(X_test))
        r2 = r2_score(y_test, DT.predict(X_test))
        print ('\nDT with max depth = ', depth)
        print ('MSE = ' , mse)
        print('R^2: ' , r2)

        if mse < mse_best:
            depth_best = depth
            mse_best = mse
            r2_best = r2
            best_model = DT

    return best_model, mse_best, r2_best, depth_best

def runAdaBoostDT (X_train, y_train, X_test, y_test):

    depth_list = [5, 10, 15, 50, 100]

    for depth in depth_list:
    
        #DT = DecisionTreeRegressor(max_depth=depth)

        adaDT = AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth),
                              n_estimators=3000, random_state=1)

        adaDT.fit(X_train, y_train)
        mse = mean_squared_error(y_test, adaDT.predict(X_test))
        r2 = r2_score(y_test, adaDT.predict(X_test))

        print ('\nAdaBoost DT with max_depth=', depth)
        print ('MSE = ' , mse)
        print('R^2: ' , r2)
        


# Random Forests
def runRF (X_train, y_train, X_test, y_test, n_estimators):
    n_best = n_estimators[0]
    mse_best = 10000
    r2_best = 100
    best_model = 0
    
    for n in n_estimators:
        RF = RandomForestRegressor(n_estimators=n)
        RF.fit(X_train, y_train)
        mse = mean_squared_error(y_test, RF.predict(X_test))
        r2 = r2_score(y_test, RF.predict(X_test))
        print ('\nRandom Forest with n_estimators=', n)
        print ('MSE = ' , mse)
        print('R^2: ' , r2)

        if mse < mse_best:
            n_best = n
            mse_best = mse
            r2_best = r2
            best_model = RF

    return best_model, mse_best, r2_best, n_best
        

# SVM
def runSVM (X_train, y_train, X_test, y_test):
    sv = svm.SVR(kernel='poly', C=1e3, degree=2)
    #sv = svm.SVR()
    sv.fit(X_train, y_train)

    mse = mean_squared_error(y_test, sv.predict(X_test))
    r2 = r2_score(y_test, sv.predict(X_test))
    print ('\nSVM:')
    print ('MSE = ' , mse)
    print('R^2: ' , r2)

    return sv, mse, r2

def runSVR_poly (X_train, y_train, X_test, y_test):
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)

    lw = 2
    plt.scatter(T, y, color='darkorange', label='data')
    plt.hold('on')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

# ensemble bagging
def runEnsembleBaggingDT (X_train, y_train, X_test, y_test):
    ensemble = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                                max_features=1.0,
                                bootstrap_features=False,
                                random_state=7).fit(X_train, y_train)

    mse = mean_squared_error(y_test, ensemble.predict(X_test))
    r2 = r2_score(y_test, ensemble.predict(X_test))
    print ('\nEnsemble Bagging using DT:')
    print ('MSE = ' , mse)
    print('R^2: ' , r2)

    return ensemble, mse, r2



# ensemble bagging
def runEnsembleBaggingLR (X_train, y_train, X_test, y_test):
    # Trained on sparse format
    bag_knn = BaggingRegressor(
        base_estimator=KNeighborsRegressor(),
        random_state=1,
    ).fit(X_train, y_train)

    mse = mean_squared_error(y_test, bag_knn.predict(X_test))
    r2 = r2_score(y_test, bag_knn.predict(X_test))
    print ('\nEnsemble Bagging using KNN Regression:')
    print ('MSE = ' , mse)
    print('R^2: ' , r2)

    return bag_knn, mse, r2


# best model after trying different regressors
def runBestModel (X, y, X_train, y_train, X_test, y_test):
    seed = 9
    best_model = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=1.0, loss='ls', max_depth=None,
             max_features=None, max_leaf_nodes=4, min_impurity_split=1e-07,
             min_samples_leaf=1, min_samples_split=5,
             min_weight_fraction_leaf=0.0, n_estimators=20000,
             presort='auto', random_state=2, subsample=1.0, verbose=0,
             warm_start=False)

    best_model.fit(X_train, y_train)

    mse = mean_squared_error(y_test, best_model.predict(X_test))
    r2 = r2_score(y_test, best_model.predict(X_test))
    print ('\nBest Model:')
    print ('MSE = ' , mse)
    print('R^2: ' , r2)

    # cross validation to the best model
    
    # ran it and mean is .9959
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    results = model_selection.cross_val_score(best_model, X, y, cv=kfold)
    print ('Cross Validation Results:', results)
    print('Cross Validation Mean:', results.mean())
    
    return best_model

# save the model into disk
def saveModel (name, model):
    name = name + '.pkl'
    joblib.dump(model, name)

# load a model from disk
def loadModel (name):
    name = name + '.pkl'
    model = joblib.load(name)
    return model

# predict IDT value
def predict (model, params):
    IDT = model.predict (params)
    return IDT

# predict IDT value of a file input
def predict_file (model, file_in, file_out): 
    f_out = open (file_out, 'wt', newline='')
    writer = csv.writer(f_out)
    writer.writerow(['temp', 'press', 'phi', 'OrigIDT', 'PredIDT'])

    with open(file_in, newline='') as f_in:
        reader = csv.reader(f_in)
        for row in reader:
            temp = row [0]
            press = row [1]
            phi = row [2]
          #  OrigIDT_10 = row [3]
            OrigIDT = row [4]
            params = np.array([temp, press, phi])
            params = params.reshape(1, -1)
            PredIDT = model.predict (params)
            PredIDT = PredIDT.item(0)
         #   PredIDT = '%.1f' % round(PredIDT/10, 4)
            
            writer.writerow ([temp, press, phi, OrigIDT, PredIDT/10])

    f_out.close ()

def main():
    # load dataset
    filename = '\\ML\\Ignition_Model\\datasets\\CH4_IDT_dataset_12727_no_header.csv'
    raw_data = open(filename, 'r')
    dataset = np.loadtxt(raw_data, delimiter=',')

    seed = 9
    start_time = time.time()
    
    # separate the data from the target attributes
    X = dataset[:,0:3]
    y = dataset[:,3] 

    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state=0)

    raw_data.close()

    '''
    best_model = 0
    best_mse = -1
    best_r2 = -1
    
    # call KNN using different K values
    K = [1, 2, 3, 4, 5, 10]
    KNN_model, KNN_mse, Knn_r2, k = runKNN (X_train, y_train, X_test, y_test, K)

    #initialize best model with KNN results
    best_model = KNN_model
    best_mse = KNN_mse
    best_r2 = Knn_r2

    # call Decision Tree using different depth
    depth_list = [5, 10, 40, 70, 100, 150, 180, 230]
    DT_model, DT_mse, DT_r2, depth = runDT (X_train, y_train, X_test, y_test, depth_list)

    # check if it is a better model
    if (DT_r2 > best_r2):
        best_model = DT_model
        best_mse = DT_mse
        best_r2 = DT_r2

    # call Random Forest using different # of estimators
    n_estimators_list = [3, 5, 10, 30, 50, 100, 150, 300]
    RF_model, RF_mse, RF_r2, RF_nEst = runRF (X_train, y_train, X_test, y_test, n_estimators_list)

    # check if it is a better model
    if (RF_r2 > best_r2):
        best_model = RF_model
        best_mse = RF_mse
        best_r2 = RF_r2

    '''
    # call SVM 
    SVM_model, SVM_mse, SVM_r2 = runSVM (X_train, y_train, X_test, y_test)

    '''
    # check if it is a better model
    if (SVM_r2 > best_r2):
        best_model = SVM_model
        best_mse = SVM_mse
        best_r2 = SVM_r2

    # call Gradient
    GB_model, GB_mse, GB_r2 = runGB (X_train, y_train, X_test, y_test)

    
    # check if it is a better model
    if (GB_r2 > best_r2):
        best_model = GB_model
        best_mse = GB_mse
        best_r2 = GB_r2

    
    # call Ensemble with Gradient
    n_estimators_list = [50, 70, 100, 150, 300, 1000, 5000, 10000]
    En_GB_model, En_GB_mse, En_GB_r2, En_GB_nEst = runEnsemble_GB (X_train, y_train, X_test, y_test, n_estimators_list)

    # predict IDT value
    params = [1000, 40, 1] # temp, press, phi
    IDT = predict (En_GB_model, params)
    print ('predicted IDT for params:', params, ' = ', IDT/10)

    
    # check if it is a better model
    if (En_GB_r2 > best_r2):
        best_model = En_GB_model
        best_mse = En_GB_mse
        best_r2 = En_GB_r2

    
    # call Ensemble Bagging
    EnBag_model, EnBag_mse, EnBag_r2 = runEnsembleBaggingDT (X_train, y_train, X_test, y_test)

    # check if it is a better model
    if (EnBag_r2 > best_r2):
        best_model = EnBag_model
        best_mse = EnBag_mse
        best_r2 = EnBag_r2        
    

    # Best Model
    print ('\n\nBest Model:', best_model)
    print ('Best MSE:', best_mse)
    print ('Best R^2:', best_r2)

    
    # cross validation of Random Forest
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    results = model_selection.cross_val_score(RF_model, X, y, cv=kfold)
    print ('Cross Validation Results:', results)
    print('Cross Validation Mean:', results.mean())

    # predict IDT value
    params = [1200, 41, 0.4] # temp, press, phi
    IDT = predict (RF_model, params)
    print ('predicted IDT for params:', params, ' = ', IDT/10)

    params = [1200, 41, 1] # temp, press, phi
    IDT = predict (RF_model, params)
    print ('predicted IDT for params:', params, ' = ', IDT/10)

    params = [1200, 41, 2] # temp, press, phi
    IDT = predict (RF_model, params)
    print ('predicted IDT for params:', params, ' = ', IDT/10)
    

    #predict_file (RF_model, filename, 'IDT_RF_model.csv')
    #predict_file (En_GB_model, filename, 'IDT_En-GB_model.csv')
    #predict_file (GB_model, filename, 'IDT_GB_model.csv')  

    #runAdaBoostDT (X_train, y_train, X_test, y_test)
    
 
    # run the best model after trying different regressors
    best_model = runBestModel (X, y, X_train, y_train, X_test, y_test)
    #predict_file (best_model, filename, 'IDT_best_model.csv')

    # Save the model
    #saveModel ('IDT_best_model', best_model)

    # predict IDT value
    params = [1000, 40, 1] # temp, press, phi
    IDT = predict (best_model, params)
    print ('predicted IDT for params:', params, ' = ', IDT/10)

    # load the model from disk
    #IDT_best_model = loadModel ('IDT_best_model')
    '''
    

    '''
    # predict IDT value
    params = [1200, 41, 0.4] # temp, press, phi
    IDT = predict (IDT_best_model, params)
    print ('predicted IDT for params:', params, ' = ', IDT)

    params = [1200, 41, 1] # temp, press, phi
    IDT = predict (IDT_best_model, params)
    print ('predicted IDT for params:', params, ' = ', IDT)

    params = [1200, 41, 2] # temp, press, phi
    IDT = predict (IDT_best_model, params)
    print ('predicted IDT for params:', params, ' = ', IDT)

    
    
    print("--- Processing time %s seconds ---" % (time.time() - start_time))

    
    ## PLOT the best model
    plt.figure(figsize=(12, 6))
    plt.title('RF Test Data')
    plt.plot(list(range (len(y_test))), RF.predict(X_test), 'b-',
             label='Predicted')
    plt.plot(list(range (len(y_test))), y_test, 'r-',
             label='original')
    plt.xlabel('Sample#')
    plt.ylabel('IDT')
    plt.legend()
    plt.show()
    '''

    '''
    plt.figure(figsize=(12, 6))
    plt.title('RF All DataSet')
    plt.plot(list(range (len(y))), RF.predict(X), 'b-',
             label='Predicted')
    plt.plot(list(range (len(y))), y, 'r-',
             label='original')
    plt.xlabel('Sample#')
    plt.ylabel('IDT')
    plt.legend()
    plt.show()
    '''

if __name__ == '__main__':
    main()
