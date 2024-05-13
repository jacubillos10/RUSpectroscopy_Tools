from sklearn import linear_model, metrics, model_selection, preprocessing, multioutput, svm, ensemble
import csv 

#Regressors (individual) 
RidgeReg = linear_model.Ridge()
svmReg = svm.SVR()
RandomFReg = ensemble.RandomForestRegressor()
GradBoostReg = ensemble.GradientBoostingRegressor()

#Multioutput Regressors
RidgeRegM = multioutput.MultiOutputRegressor(RidgeReg)
svmRegM = multioutput.MultiOutputRegressor(svmReg)
RandomFRegM = multioutput.MultiOutputRegressor(RandomFReg)
GradBoostRegM = multioutput.MultiOutputRegressor(GradBoostReg)

#Parameter selection
paramRR = {'alpha': [0.5, 1, 5, 10]}
clfRR = model_selection.GridSearchCV(RidgeRegM,paramRR)

paramSVM = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clfSVMR = model_selection.GridSearchCV(svmRegM,paramSVM)

paramRF = {'n_estimators':[50, 100, 500]}
clfRFR = model_selection.GridSearchCV(RandomFRegM,paramRF)

paramGB = {'n_estimators':[50, 100, 500], 'learning_rate':[0.1, 50, 100]}
clfGBR = model_selection.GridSearchCV(GradBoostRegM,paramGB)

