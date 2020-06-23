import keras as ks
from keras import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense,Dropout,BatchNormalization
from sklearn.model_selection import KFold
from keras import regularizers
from keras.optimizers import SGD
from keras.constraints import UnitNorm
import os
import numpy as np
def get_threshold_metrics(y_true, y_pred, drop_intermediate=False,
                          disease='all'):
    """
    Retrieve true/false positive rates and auroc/aupr for class predictions

    Arguments:
    y_true - an array of gold standard mutation status
    y_pred - an array of predicted mutation status
    disease - a string that includes the corresponding TCGA study acronym

    Output:
    dict of AUROC, AUPR, pandas dataframes of ROC and PR data, and cancer-type
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import precision_recall_curve, average_precision_score

    roc_columns = ['fpr', 'tpr', 'threshold']
    pr_columns = ['precision', 'recall', 'threshold']

    if drop_intermediate:
        roc_items = zip(roc_columns,
                        roc_curve(y_true, y_pred, drop_intermediate=False))
    else:
        roc_items = zip(roc_columns, roc_curve(y_true, y_pred))

    roc_df = pd.DataFrame.from_dict(dict(roc_items))

    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    pr_df = pd.DataFrame.from_records([prec, rec]).T
    pr_df = pd.concat([pr_df, pd.Series(thresh)], ignore_index=True, axis=1)
    pr_df.columns = pr_columns

    auroc = roc_auc_score(y_true, y_pred, average='weighted')
    aupr = average_precision_score(y_true, y_pred, average='weighted')

    return {'auroc': auroc, 'aupr': aupr, 'roc_df': roc_df,
            'pr_df': pr_df, 'disease': disease}

gpu_id = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')

ks.backend.clear_session()
print('reading')

x_np=np.array(pd.read_csv('genefile.csv',index_col=0,header=0))
y_df=np.array(pd.read_csv('statefile.csv',index_col=0,header=0))
strat=pd.read_csv('stratfile.csv',index_col=0,header=0)

print('read down')
input_dim=2048



x_train_all,x_test_all, y_train_all, y_test_all =train_test_split(x_df.iloc[:,:input_dim],y_df,test_size=0.1, random_state=0,stratify=strat)#,stratify=strat)
input = Input(shape=(input_dim,))
encoded=Dense(128,activation='relu',name='encoded_hidden1')(input)
encoder_output=Dense(64,activation='relu',name='encoded_hidden2')(encoded)
LR=Dense(32,activation='sigmoid',name='LR')(encoder_output)

decoded=Dense(64,activation='relu',name='decoded_hidden2')(encoder_output)
decoded=Dense(128,activation='relu',name='decoded_hidden3')(decoded)
decoded=Dense(2000,activation='tanh',name='decoded_output')(decoded)
final_1 = Dense(1, activation='sigmoid')(LR)

autoencoder=Model(inputs=input,outputs=decoded)

autoencoder.compile(optimizer='sgd',
              loss='mean_squared_error',#loss='binary_crossentropy',
              metrics=['accuracy'])
encoder=Model(inputs=input,outputs=final_1)
encoder.compile(optimizer='sgd',
              loss='binary_crossentropy',#loss='binary_crossentropy',
              metrics=['accuracy'])
seed = 1
np.random.seed(seed)
all_specificity = list()
all_precision = list()
all_sensitivity = list()
accurancy_sum = 0
loss_sum = 0
k_fold = KFold(5,True, random_state=1)
index = k_fold.split(X=x_train_all, y=y_train_all)

index_train_all=np.array(x_train_all.index)
cv_results_df=list()
for train_index, test_index in index:
        X_train = np.array(x_train_all.iloc[train_index,:])
        X_cv = np.array(x_train_all.iloc[test_index,:])
        y_train = np.array(y_train_all.iloc[train_index,:])
        y_cv = np.array(y_train_all.iloc[test_index,:])
        test_index=np.array(index_train_all[test_index])
        print("Training --------------------")
        learning_rate_reduction = ks.callbacks.ReduceLROnPlateau(monitor='loss', patience=5, verbose=1,
                                                                 epsilon=0.0003, factor=0.5, min_lr=1e-5)#epsilon=0.0003, factor=0.9, min_lr=0.00001)
        t = autoencoder.fit(X_train, X_train, batch_size=128, epochs=50, verbose=1,
                      callbacks=[learning_rate_reduction], validation_data=(X_cv,X_cv))  # batch_size=128ds
        print("\nTesting --------------------")
        cv_pred=np.concatenate((encoder.predict(X_cv),y_cv),axis=1)
        cv_df=pd.DataFrame(data=cv_pred, columns=['dignosis','total_status'],index=test_index)
        cv_results_df.append(cv_df)

cv_df=pd.concat(cv_results_df,axis=0)
cv_df.to_csv('SAEcv_results.csv')

train_pred=np.concatenate((encoder.predict(np.array(x_train_all.loc[:,:])),np.array(y_train_all.loc[:,:])),axis=1)
train_df=pd.DataFrame(data=train_pred, columns=['dignosis','total_status'],index=x_train_all.index)
train_df.to_csv('SAEtrain_results.csv')

test_pred=np.concatenate((encoder.predict(np.array(x_test_all.loc[:,:])),np.array(y_test_all.loc[:,:])),axis=1)
test_df=pd.DataFrame(data=test_pred, columns=['dignosis','total_status'],index=x_test_all.index)
test_df.to_csv('SAEtest_results.csv')

