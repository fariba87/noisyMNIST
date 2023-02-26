
# try Random forest as traditional Machine leanring for small data set and Imbalanced dataset


import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data_load import image_label_data , class_weight
#%%
#load data
train_data , train_label,val_data, val_label =image_label_data(only_resize=False,
                                                               only_resize_and_balance=False, 
                                                               preprocess_and_resize=True,
                                                               preprocess_and_resize_and_balanced=False,
                                                               augment_class_wise_=False,
                                                               augment_class_wise_preprocess=False)



Xtt=np.reshape(train_data,(len (train_data),-1))
Xttest=np.reshape(val_data,(len (val_data),-1))
#%% try Random Forest on train data and evaluate on test data

clf = RandomForestClassifier(n_estimators=800, max_depth=5, random_state=0, class_weight='balanced' )
clf.fit(Xtt,  np.argmax(train_label, axis=1))
scores_test=clf.score(Xttest,  np.argmax(val_label, axis=1))
scores_train=clf.score(Xtt,  np.argmax(train_label, axis=1))
#score_train =0.83 val_score=0.35

# we see overfitting --> try Kfold cross validation or tuning hyperparameter of Random Forest

#%%
 #  Kfold cross validation


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
kfold = StratifiedKFold(n_splits=10, shuffle=True)
train_concat=np.concatenate((train_data, val_data), axis=0)
#train_concat=train_data
#label_concat=train_label
label_concat=np.concatenate((train_label, val_label), axis=0)

train_concat=np.reshape(train_concat, (len (train_concat),-1))
val_data_RF=np.reshape(val_data,(len (val_data),-1)) 


#.split(train_concat, label_concat)
scores = []
acc_per_fold = []
loss_per_fold = []
fold_no=1
for train_ix, test_ix in kfold.split(train_concat, np.argmax(label_concat,axis=1)):
    train_X , test_X= train_concat[train_ix] , train_concat[test_ix]
    train_y , test_y= label_concat[train_ix],label_concat[test_ix]
 
    clf.fit(train_X ,np.argmax(train_y,axis=1))

    score_train=clf.score(train_concat,  np.argmax(label_concat, axis=1))

    score_val=clf.score(val_data_RF,  np.argmax(val_label, axis=1))

    print(f'train Score for fold {fold_no}: {score_train}')
    
    print(f'test Score for fold {fold_no}: {score_val}')

    fold_no = fold_no + 1
    
# train_Score= 0.94 val_score=0.64