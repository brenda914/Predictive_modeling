
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import random
from datetime import datetime
from datetime import date
from datetime import time
import math
import statistics

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
from xgboost import plot_importance
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score,roc_auc_score,auc,roc_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb


# ## Model Selection

# In[2]:


train=pd.read_csv('train_final.csv')
test=pd.read_csv('test_final.csv')

variable_names = list(train)
do_not_use_for_training = ['index','claim_date',
                         'claim_number',
                         'fraud',
                         'Zipcode',
                         'state',
                         'City',
                         'EstimatedPopulation',
                         'TotalWages',
                         'TaxReturnsFiled',
                          'claim/income/liab']

feature_names = [f for f in variable_names if f not in do_not_use_for_training]
print('features selected are:',feature_names)

X_train1 = train[feature_names]
Y_train1 = train['fraud']

X_train,X_val,Y_train,Y_val = train_test_split(X_train1, Y_train1,test_size = 0.2,random_state=10)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_val_std=sc.transform(X_val)

X_test = test[feature_names]
X_test_std=sc.transform(X_test)


# ## Xgboost

# In[ ]:


print('***************************Start model*****************************')
print('***************************use grid search to choose best parameters for each model *****************************')
print('Model 1: Xgboost')

clf=xgb.XGBClassifier(n_estimators=100,
                            learning_rate =0.1,
                            max_depth=3,
                            min_child_weight=1,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.75,
                            objective= 'binary:logistic',
                            nthread=12,
                            scale_pos_weight=1,
                            reg_alpha=0.01,
                            seed=27)

xgb_model = clf.fit(X_train_std, Y_train)

fpr,tpr,thresholds=roc_curve(Y_val,xgb_model.predict_proba(X_val_std)[:,1])
print(auc(fpr,tpr))


# ## GradientBoosting

# In[ ]:


print('Model 2: Gradient Boosting')

clf=GradientBoostingClassifier(n_estimators=100,
                               learning_rate=0.1,
                               max_depth=3,
                               max_features='sqrt',
                               min_samples_split=300,
                               min_samples_leaf=40,
                               subsample=0.8,
                               random_state=10)

gbdt=clf.fit(X_train_std,Y_train)
fpr,tpr,thresholds=roc_curve(Y_val,gbdt.predict_proba(X_val_std)[:,1])
print(auc(fpr,tpr))


# ## Lightgbm

# In[ ]:


print('Model 3: Lightgbm')

clf=lgb.LGBMClassifier(application='binary',objective='binary',metric='auc',is_unbalance=True,boosting='gbdt',
                           num_leaves=3,learning_rate=0.2,verbose=0)

lgb_model = clf.fit(X_train_std, Y_train)
fpr,tpr,thresholds=roc_curve(Y_val,lgb_model.predict_proba(X_val_std)[:,1])
print(auc(fpr,tpr))


# ## Logistic Regression

# In[ ]:


print('Model 4: Logistic Regression')
def random_list(start,stop,length):
    if length>=0:
        length=int(length)
        start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
        random_list = []
    for i in range(length):
        random_list.append(random.uniform(start, stop))
    return(random_list)

params = {'C':random_list(0,1000,100),
         'penalty':['l1', 'l2'],
         'solver':['liblinear']}
                   
lr = LogisticRegression()

clf = GridSearchCV(lr,params,cv=5,scoring='roc_auc')
best_lr=clf.fit(X_train,Y_train)
print('Best Penalty:', best_lr.best_estimator_.get_params()['penalty'])
print('Best C:', best_lr.best_estimator_.get_params()['C'])
print('Best score:', best_lr.best_score_)


# ## NAIVE BAYES CLASSIFICATION

# In[ ]:


print('Model 4: Naive Bayes Classification')
nb = GaussianNB()
clf= GridSearchCV(nb,params,cv=5,scoring='roc_auc')
best_nb=clf.fit(X_train_std,Y_train)
print('Best Penalty:', best_lr.best_estimator_.get_params()['penalty'])
print('Best C:', best_lr.best_estimator_.get_params()['C'])
print('Best score:', best_lr.best_score_)


# ## Random Forest

# In[ ]:


print('Model 5: Random Forest')
param_test = {'n_estimators':[100,1000,1100]}
clf = GridSearchCV(RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt' 
                                          ,random_state=10),param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

best_rf = clf.fit(X_train_std, Y_train)
print('Best estimator:', best_rf.best_estimator_.get_params()['n_estimators'])
print('Best score:', best_rf.best_score_)


# ## Easy ensemble classifier

# In[ ]:


from imblearn.ensemble import EasyEnsembleClassifier
print("Model 6: Balanced Random Forest")
eec = EasyEnsembleClassifier(n_estimators=100, 
                             base_estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
                                                               n_estimators=20,learning_rate=0.5),
                             warm_start=False, sampling_strategy='auto', 
                             replacement=False, random_state=0)

eec.fit(X_train_std, Y_train)

clf = eec
y_train_pred = clf.predict(X_train_std)
y_pred = clf.predict(X_val_std)
print("Training Accuracy : {:.2%}".format(accuracy_score(y_train_pred, Y_train)))
print("Balanced Training Accuracy : {:.2%}".format(balanced_accuracy_score(y_train_pred, Y_train)))
print("Testing Accuracy : {:.2%}".format(accuracy_score(y_pred, Y_val)))
print("Balanced Testing Accuracy : {:.2%}".format(balanced_accuracy_score(y_pred, Y_val)))
print("Confusion Matrix:")
print(confusion_matrix(Y_val, y_pred))
print("Classification Report:")
print(classification_report(Y_val, y_pred))


# ## AdaBoost

# In[ ]:


adaboost = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=20,
    learning_rate=0.5)

adaboost.fit(X_train_std, y_train)

clf = adaboost
y_train_pred = clf.predict(X_train_std)
y_pred = clf.predict(X_val_std)
print("Training Accuracy : {:.2%}".format(accuracy_score(y_train_pred, Y_train)))
print("Balanced Training Accuracy : {:.2%}".format(balanced_accuracy_score(y_train_pred, Y_train)))
print("Testing Accuracy : {:.2%}".format(accuracy_score(y_pred, Y_val)))
print("Balanced Testing Accuracy : {:.2%}".format(balanced_accuracy_score(y_pred, Y_val)))
print("Confusion Matrix:")
print(confusion_matrix(Y_val, y_pred))
print("Classification Report:")
print(classification_report(y_pred, Y_val))

y_pred_probs = clf.predict_proba(X_val_std)
fpr, tpr, thresholds = roc_curve(Y_val, y_pred_probs[:, 1])
print(auc(fpr, tpr))


# ## Ensemble models utilizing Stacking method

# In[ ]:


eec_probs = eec.predict_proba(X_val_std)
gbdt_probs = gbdt.predict_proba(X_val_std)
xgb_probs = xgb_model.predict_proba(X_val_std)

best_auc = 0
for a in np.arange(0.1, 1.0, 0.1):
    for b in np.arange(0.1, 1.0 - a, 0.1):
        c = 1 - a - b
        stacked_probs = a * eec_probs + b * gbdt_probs + c * xgb_probs
        fpr, tpr, thresholds = roc_curve(Y_val, stacked_probs[:, 1])
        new_auc = auc(fpr, tpr)
        if new_auc > best_auc:
            best_auc = new_auc
            best = (a, b, c)
print(best, best_auc)

best_auc = 0
for a in np.arange(0.1, 1.0, 0.1):
    b = 1 - a
    stacked_probs = a * eec_probs + b * gbdt_probs
    fpr, tpr, thresholds = roc_curve(Y_val, stacked_probs[:, 1])
    new_auc = auc(fpr, tpr)
    if new_auc > best_auc:
        best_auc = new_auc
        best = (a, b)
print(best, best_auc)

# stacked_probs = 0.7 * eec_probs + 0.2 * gbdt_probs + 0.1 * xgb_probs
stacked_probs = 0.9 * eec_probs + 0.1 * gbdt_probs
fpr, tpr, thresholds = roc_curve(Y_val, stacked_probs[:, 1])
auc(fpr, tpr)


# ## Stacking 2

# In[ ]:


from mlxtend.classifier import EnsembleVoteClassifier

eclf = EnsembleVoteClassifier(clfs=base_models,
                              weights=[1,1,1,1,1,1,5], voting='soft')

eclf.fit(X_train_std, Y_train)

clf = eclf
y_pred_probs = clf.predict_proba(X_val_std)
fpr, tpr, thresholds = roc_curve(Y_val, y_pred_probs[:, 1])
print(auc(fpr, tpr))

print("               ===== Balanced Random Forest =====")
clf = eclf
Y_train_pred = clf.predict(X_train_std)
y_pred = clf.predict(X_val_std)
print("Training Accuracy : {:.2%}".format(accuracy_score(Y_train_pred, Y_train)))
print("Balanced Training Accuracy : {:.2%}".format(balanced_accuracy_score(Y_train_pred, Y_train)))
print("Testing Accuracy : {:.2%}".format(accuracy_score(y_pred, Y_val)))
print("Balanced Testing Accuracy : {:.2%}".format(balanced_accuracy_score(y_pred, Y_val)))
print("Confusion Matrix:")
print(confusion_matrix(Y_val, y_pred))
print("Classification Report:")
print(classification_report(Y_val, y_pred))

from mlxtend.classifier import StackingClassifier
sclf = StackingClassifier(classifiers=base_models, 
                          meta_classifier=LogisticRegression(penalty='l2',
                                                             solver='liblinear', 
                                                             class_weight='balanced'), 
                          use_probas=True)

sclf.fit(X_train_std, Y_train)

clf = sclf
y_pred_probs = clf.predict_proba(X_val_std)
fpr, tpr, thresholds = roc_curve(Y_val, y_pred_probs[:, 1])
print(auc(fpr, tpr))

print("               ===== Balanced Random Forest =====")
clf = sclf
Y_train_pred = clf.predict(X_train_std)
y_pred = clf.predict(X_val_std)
print("Training Accuracy : {:.2%}".format(accuracy_score(Y_train_pred, Y_train)))
print("Balanced Training Accuracy : {:.2%}".format(balanced_accuracy_score(Y_train_pred, Y_train)))
print("Testing Accuracy : {:.2%}".format(accuracy_score(y_pred, Y_val)))
print("Balanced Testing Accuracy : {:.2%}".format(balanced_accuracy_score(y_pred, Y_val)))
print("Confusion Matrix:")
print(confusion_matrix(Y_val, y_pred))
print("Classification Report:")
print(classification_report(Y_val, y_pred))

from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import MultiLayerPerceptron as MLP

nn1 = MLP(hidden_layers=[50], 
          l2=0.00, 
          l1=0.0, 
          epochs=20, 
          eta=0.05, 
          momentum=0.1,
          decrease_const=0.0,
          minibatches=1, 
          random_seed=1,
          print_progress=3)

sclf = StackingClassifier(classifiers=base_models, 
                          meta_classifier=nn1, use_probas=True)

sclf.fit(X_train_std, Y_train)

clf = sclf
y_pred_probs = clf.predict_proba(X_val_std)
fpr, tpr, thresholds = roc_curve(Y_val, y_pred_probs[:, 1])
print(auc(fpr, tpr))

print("               ===== Balanced Random Forest =====")
clf = sclf
Y_train_pred = clf.predict(X_train_std)
y_pred = clf.predict(X_val_std)
print("Training Accuracy : {:.2%}".format(accuracy_score(Y_train_pred, Y_train)))
print("Balanced Training Accuracy : {:.2%}".format(balanced_accuracy_score(Y_train_pred, Y_train)))
print("Testing Accuracy : {:.2%}".format(accuracy_score(y_pred, Y_val)))
print("Balanced Testing Accuracy : {:.2%}".format(balanced_accuracy_score(y_pred, Y_val)))
print("Confusion Matrix:")
print(confusion_matrix(Y_val, y_pred))
print("Classification Report:")
print(classification_report(Y_val, y_pred))


# ## Stacking3

# In[ ]:


lr = LogisticRegression(solver='liblinear', class_weight='balanced')
rf = RandomForestClassifier(n_estimators=1000, max_depth=10)
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),
                                n_estimators = 100,
                                sampling_strategy = 1.0,
                                random_state=0)
eec = EasyEnsembleClassifier(n_estimators=100, 
                             base_estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
                                                               n_estimators=20,learning_rate=0.5),
                             warm_start=False, sampling_strategy='auto', 
                             replacement=False, random_state=0)
adaboost = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=20,
    learning_rate=0.5)
gbdt_params = {'n_estimators': 100, 'max_depth': 2, 'min_samples_split': 2,
          'learning_rate': 0.1}
gbdt = GradientBoostingClassifier(**gbdt_params)
param = {}
param['max_depth'] = 6
param['learning_rate'] = 0.2
param['subsample'] = 0.9
param['colsample_bytree'] = 0.7
param['min_split_loss'] = 15
param['min_child_weight'] = 8
param['scale_pos_weight'] = 0.8
param['objective'] = 'binary:logistic'
param['eval_metric'] = 'auc'
param['silent'] = 1
xgboost = xgb.XGBClassifier()
xgb_model = xgboost.fit(X_train_std, Y_train,
                       eval_set=[(X_val_std, y_test)], 
                        eval_metric='auc')


base_models = [lr, rf, bbc, eec, adaboost, gbdt, xgb_model]

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y) 

predictions = np.zeros((len(X), len(base_models)))

for train_index, test_index in skf.split(X, y):
    for i, m in enumerate(base_models):
        m.fit(X[train_index,:], y[train_index])
        predictions[test_index, i] = m.predict_proba(X[test_index,:])[:,1]
        
X_train_std_meta, X_val_std_meta, Y_train_meta, y_test_meta = train_test_split(predictions[:,3:-1], y, test_size=0.2, random_state = 15)

meta_clf=LogisticRegression(penalty='l2',solver='liblinear', 
                            class_weight='balanced')
meta_clf.fit(X_train_std_meta, Y_train_meta)

clf = meta_clf
y_pred_probs = clf.predict_proba(X_val_std_meta)
fpr, tpr, thresholds = roc_curve(y_test_meta, y_pred_probs[:, 1])
print(auc(fpr, tpr))

meta_clf=RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced')
meta_clf.fit(X_train_std_meta, Y_train_meta)

print("               ===== Balanced Random Forest =====")
clf = meta_clf
Y_train_pred = clf.predict(X_train_std_meta)
y_pred = clf.predict(X_val_std_meta)
print("Training Accuracy : {:.2%}".format(accuracy_score(Y_train_pred, Y_train_meta)))
print("Balanced Training Accuracy : {:.2%}".format(balanced_accuracy_score(Y_train_pred, Y_train_meta)))
print("Testing Accuracy : {:.2%}".format(accuracy_score(y_pred, y_test_meta)))
print("Balanced Testing Accuracy : {:.2%}".format(balanced_accuracy_score(y_pred, y_test_meta)))
print("Confusion Matrix:")
print(confusion_matrix(y_test_meta, y_pred))
print("Classification Report:")
print(classification_report(y_test_meta, y_pred))

clf = meta_clf
y_pred_probs = clf.predict_proba(X_val_std_meta)
fpr, tpr, thresholds = roc_curve(y_test_meta, y_pred_probs[:, 1])
print(auc(fpr, tpr))

meta_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
                                   n_estimators = 100, learning_rate = 0.5)

meta_clf.fit(X_train_std_meta, Y_train_meta)

clf = meta_clf
y_pred_probs = clf.predict_proba(X_val_std_meta)
fpr, tpr, thresholds = roc_curve(y_test_meta, y_pred_probs[:, 1])
print(auc(fpr, tpr))

meta_clf = GradientBoostingClassifier(n_estimators=300, max_depth=2, learning_rate=0.01)
meta_clf.fit(X_train_std_meta, Y_train_meta)
clf = meta_clf
y_pred_probs = clf.predict_proba(X_val_std_meta)
fpr, tpr, thresholds = roc_curve(y_test_meta, y_pred_probs[:, 1])
print(auc(fpr, tpr))


# ## Feature selection

# In[ ]:


# # Drop Features

xgb_selector = SelectFromModel(xgb_model)
xgb_selector.fit(X_train_std, Y_train)
xgb_support = xgb_selector.get_support()
embeded_xgb_feature = X_train.loc[:,xgb_support].columns.tolist()
print(str(len(embeded_xgb_feature)), 'selected features')

embeded_gb_selector = SelectFromModel(gbdt)
embeded_gb_selector.fit(X_train_std, Y_train)
embeded_gb_support = embeded_gb_selector.get_support()
embeded_gb_feature = X_train.loc[:,embeded_gb_support].columns.tolist()
print(str(len(embeded_gb_feature)), 'selected features')

embeded_gbm_selector = SelectFromModel(lgb_model)
embeded_gbm_selector.fit(X_train_std, Y_train)
embeded_gbm_support = embeded_gbm_selector.get_support()
embeded_gbm_feature = X_train.loc[:,embeded_gbm_support].columns.tolist()
print(str(len(embeded_gbm_feature)), 'selected features')

embeded_rf_selector = SelectFromModel(best_rf)
embeded_rf_selector.fit(X_train_std, Y_train)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X_train.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


features = list(X_train1.columns.values)
features = set(features)

a = set(embeded_xgb_feature)
b = set(embeded_gbm_feature)
c = set(embeded_rf_feature)
d = set(embeded_gb_feature)

to_drops = list(features-a-b-c-d)

features = list(X_train1.columns.values)
Y_train = train['fraud']
feature_selection = {}
vc_score_flag = 0.7271459262650759
init_vc_score = 0.7241006595750108
vc_score = init_vc_score
count = 0

#change the num, vc_score with the best
#drop features and try to increase the score

while vc_score <= 0.7271459262650759:
    
    to_remove = to_drops[np.random.randint(0, len(to_drops))]
    #print(to_remove)
    features.remove(to_remove)
    X_train = train[features]
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    x_train,x_test,y_train,y_test = train_test_split(X_train_std, Y_train,test_size = 0.2,random_state = 10)
    #xgb
    best_xgb1 = xgb.XGBClassifier(n_estimators=100,
                     max_depth=3,
                     min_child_weight=1,
                     learning_rate =0.08,
                     gamma=0,
                     scale_pos_weight=1,
                     subsample=0.8,
                     colsample_bytree=0.75,
                     reg_alpha=0.01,
                     objective= 'binary:logistic',
                     silent=1,
                     booster='gbtree', 
                     nthread=4,
                     reg_lambda=1,
                     seed=27)  
    best_xgb1.fit(x_train, y_train)
    fpr,tpr,thresholds=roc_curve(y_test,best_xgb1.predict_proba(x_test)[:,1])
    xgb_score = auc(fpr,tpr)
    xgb_selector = SelectFromModel(best_xgb1)
    xgb_selector.fit(X_train_std, Y_train)
    xgb_support = xgb_selector.get_support()
    embeded_xgb_feature = X_train.loc[:,xgb_support].columns.tolist()
    #lgbm:
    lgb_model = lgb.LGBMClassifier(learning_rate = 0.15,application='binary',objective='binary',metric='auc',is_unbalance=True,boosting='gbdt',
                           num_leaves=3,c=0.1,verbose=0)
    lgb_model.fit(x_train,y_train)
    fpr,tpr,thresholds=roc_curve(y_test,lgb_model.predict_proba(x_test)[:,1])
    lgbm_score = auc(fpr,tpr)
    embeded_gbm_selector = SelectFromModel(lgb_model)
    embeded_gbm_selector.fit(X_train_std, Y_train)
    embeded_gbm_support = embeded_gbm_selector.get_support()
    embeded_gbm_feature = X_train.loc[:,embeded_gbm_support].columns.tolist()
    #gbc
    gbdt = GradientBoostingClassifier(n_estimators=140,
                               learning_rate=0.08,
                               max_depth=3,
                               max_features='sqrt',
                               min_samples_split=300,
                               min_samples_leaf=40,
                               subsample=0.8,
                               random_state=10)
    gbdt.fit(x_train,y_train)
    fpr,tpr,thresholds=roc_curve(y_test,gbdt.predict_proba(x_test)[:,1])
    gbc_score = auc(fpr,tpr)
    embeded_gb_selector = SelectFromModel(gbdt)
    embeded_gb_selector.fit(X_train_std, Y_train)
    embeded_gb_support = embeded_gb_selector.get_support()
    embeded_gb_feature = X_train.loc[:,embeded_gb_support].columns.tolist()
    #rf
    best_rf = RandomForestClassifier(n_estimators=100,
    max_depth=11,
    max_features='sqrt',
    min_samples_split=70,
    min_samples_leaf=20,
    oob_score=True,
    random_state=10)
    best_rf.fit(x_train,y_train)
    fpr,tpr,thresholds=roc_curve(y_test,best_rf.predict_proba(x_test)[:,1])
    rf_score = auc(fpr,tpr)
    embeded_rf_selector = SelectFromModel(best_rf)
    embeded_rf_selector.fit(X_train_std, Y_train)
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X_train.loc[:,embeded_rf_support].columns.tolist()
    #get list of all var without importance to models
    # put all selection together
    features = set(features)
    a = set(embeded_xgb_feature)
    b = set(embeded_gbm_feature)
    c = set(embeded_rf_feature)
    d = set(embeded_gb_feature)

    to_drops = list(features-a-b-c-d)
    features = list(features)
    #print(to_drops)
    classifiers = [('xgb', best_xgb1), ('gbc', gbdt), ('lgb', lgb_model),('rf', best_rf)]

    # Instantiate a VotingClassifier vc
    vc = VotingClassifier(estimators=classifiers, voting='soft')     
    vc.fit(x_train, y_train)   
    y_pred = vc.predict_proba(x_test)
    fpr,tpr,thresholds=roc_curve(y_test,vc.predict_proba(x_test)[:,1])
    vc_score = auc(fpr,tpr)
    feature_selection[count] = {'vc_score':vc_score, 'xgb_score':xgb_score, 'lgbm_score':lgbm_score, 'rf_score':rf_score, 'gbc_score':gbc_score,'features':features}
    count = count+ 1
#print(feature_selection)
print(vc_score)


print(feature_selection)
print(vc_score)
features = list(features)

