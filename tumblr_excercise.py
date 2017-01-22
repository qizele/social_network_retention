import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

#Read one textfile with given seperator
def raw_readin(fileloc, dlm, start):
    datafile=pd.read_csv(fileloc, sep=dlm, header=start)
    return datafile

user_act=raw_readin('E:\Careers\Tumblr\user_act.csv', ',' , 0)
user_act.columns=['act_ts','act_uid','act_device']
user_act.groupby(by='act_device').act_uid.nunique()
#de-dupe user active data, keep user_id and device type
user_act_unique=pd.pivot_table(user_act, index='act_uid',columns='act_device', values='act_ts', aggfunc='count', fill_value=0).reset_index()
#merge two datasets
user_regi=raw_readin('E:\Careers\Tumblr\user_regi.csv', ',' , 0)
user_fnl=pd.merge(left=user_regi, right=user_act_unique, how='left', left_on='user_id',right_on='act_uid')
user_fnl['is_active']=0
user_fnl.ix[~user_fnl.act_uid.isnull(),'is_active']=1
user_fnl.describe()

#outlier removal for attribute analysis (using Q3+3*IQR)
def outlier_removal(data, variable, outlier_indicator):
    data[outlier_indicator]=0
    var_q1=data.loc[data[variable]>0,variable].quantile(0.25)
    var_q3=data.loc[data[variable]>0,variable].quantile(0.75)
    var_iqr=var_q3-var_q1
    var_outlier=var_q3+3*var_iqr
    data.ix[data[variable]>var_outlier, outlier_indicator]=1

#outlier removal for attribute analysis (using Q99.5)
def outlier_removal_2(data, variable, outlier_indicator):
    data[outlier_indicator]=0
    var_outlier=data.loc[data[variable]>0,variable].quantile(0.995)
    data.ix[data[variable]>var_outlier, outlier_indicator]=1

#outlier_removal(user_fnl, 'pageviews', 'pageviews_outlier')
#outlier_removal(user_fnl, 'follows', 'follows_outlier')
#outlier_removal(user_fnl, 'likes', 'likes_outlier')
#outlier_removal(user_fnl, 'reblogs', 'reblogs_outlier')
#outlier_removal(user_fnl, 'original_posts', 'original_posts_outlier')
#outlier_removal(user_fnl, 'searches', 'searches_outlier')
#outlier_removal(user_fnl, 'received_engagments', 'received_engagments_outlier')

#user_fnl.loc[(user_fnl.pageviews_outlier==1)&(user_fnl.follows_outlier==1)&(user_fnl.likes_outlier==1)
 #            &(user_fnl.reblogs_outlier==1)&(user_fnl.original_posts_outlier==1)
 #            &(user_fnl.searches_outlier==1)&(user_fnl.received_engagments_outlier)==1,:]

#no oulier using Q3+3*IQR
#start attribute analysis
user_fnl.describe()
pageviews_q3=user_fnl.loc[user_fnl['pageviews']>0,'pageviews'].quantile(0.5)
follows_q3=user_fnl.loc[user_fnl['follows']>0,'follows'].quantile(0.5)
likes_q3=user_fnl.loc[user_fnl['likes']>0,'likes'].quantile(0.5)
reblogs_q3=user_fnl.loc[user_fnl['reblogs']>0,'reblogs'].quantile(0.5)
original_posts_q3=user_fnl.loc[user_fnl['original_posts']>0,'original_posts'].quantile(0.5)
searches_q3=user_fnl.loc[user_fnl['searches']>0,'searches'].quantile(0.5)
unfollows_q3=user_fnl.loc[user_fnl['unfollows']>0,'unfollows'].quantile(0.5)
received_engagments_q3=user_fnl.loc[user_fnl['received_engagments']>0,'received_engagments'].quantile(0.5)

pageviews_act=user_fnl.loc[user_fnl['pageviews']>0,'pageviews'].count()
follows_act=user_fnl.loc[user_fnl['follows']>0,'follows'].count()
likes_act=user_fnl.loc[user_fnl['likes']>0,'likes'].count()
reblogs_act=user_fnl.loc[user_fnl['reblogs']>0,'reblogs'].count()
original_posts_act=user_fnl.loc[user_fnl['original_posts']>0,'original_posts'].count()
searches_act=user_fnl.loc[user_fnl['searches']>0,'searches'].count()
unfollows_act=user_fnl.loc[user_fnl['unfollows']>0,'unfollows'].count()
received_engagments_act=user_fnl.loc[user_fnl['received_engagments']>0,'received_engagments'].count()

user_fnl.groupby('regi_device').count().user_id
user_fnl.groupby('regi_source').count().user_id


user_fnl['datetime'] = pd.to_datetime(user_fnl['regi_ts'],unit='s')
datatime = pd.DatetimeIndex(user_fnl.datetime)
user_fnl['day'] = datatime.day
user_fnl['hour'] = datatime.hour
user_fnl.groupby('day').count().user_id
user_fnl.groupby('hour').count().user_id
user_fnl['regi_timeperiod']='night'
user_fnl.ix[(user_fnl.hour>=5)&(user_fnl.hour<13),'regi_timeperiod']='morning'
user_fnl.ix[(user_fnl.hour>=13)&(user_fnl.hour<18),'regi_timeperiod']='afternoon'
user_fnl.ix[(user_fnl.hour>=0)&(user_fnl.hour<5),'regi_timeperiod']='late_night'
user_fnl.groupby('regi_timeperiod').count().user_id

user_fnl_active=user_fnl.loc[user_fnl['is_active']==1,:]
user_fnl_active.describe()


user_fnl_inactive=user_fnl.loc[user_fnl['is_active']==0,:]
user_fnl_inactive.describe()

##################var important############################
user_model_data=user_fnl.loc[:,['regi_ts','user_id','is_verified','pageviews','follows','likes','reblogs','original_posts',
                                'searches','unfollows','received_engagments','regi_device','regi_source','is_active']]

user_model_data['datetime'] = pd.to_datetime(user_model_data['regi_ts'],unit='s')
datatime = pd.DatetimeIndex(user_model_data.datetime)
user_model_data['day'] = datatime.day
user_model_data['hour'] = datatime.hour
user_model_data['regi_device']=user_model_data['regi_device'].astype('category')
user_model_data['regi_source']=user_model_data['regi_source'].astype('category')

#greate dummy variable for categorical variables
user_model_data=pd.get_dummies(user_model_data,columns=['regi_device','regi_source']) 

#training testing seperation
import random
training_size=int(0.8*len(user_model_data))
training_set=user_model_data.loc[random.sample(user_model_data.index, training_size),:]
validation_set=user_model_data.loc[~user_model_data.index.isin(training_set.index),:]

#variable importance
from sklearn.ensemble import *
from sklearn.metrics import *
train_target=training_set['is_active']
train_data_var=training_set.drop(['is_active','user_id','regi_ts', 'datetime'],axis=1)
val_target=validation_set['is_active']
val_data_var=validation_set.drop(['is_active','user_id','regi_ts', 'datetime'],axis=1)

rf=RandomForestClassifier(n_estimators=1000,max_features='sqrt',class_weight='balanced').fit(train_data_var,train_target)
imp=rf.feature_importances_
imp=pd.Series(imp,name='importance')
feature=pd.Series(train_data_var.columns, name='feature')
var_imp=pd.concat([feature,imp], axis=1)
var_imp.sort_values(by='importance', ascending=False,inplace=True)

from sklearn.metrics import roc_auc_score
pred_full=rf.predict_proba(val_data_var)
roc_auc_score(val_target, pred_full[:,1])

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100)
gbc.fit(train_data_var,train_target)
imp2=gbc.feature_importances_
imp2=pd.Series(imp2,name='importance')
feature=pd.Series(train_data_var.columns, name='feature')
var_imp2=pd.concat([feature,imp2], axis=1)
var_imp2.sort_values(by='importance', ascending=False,inplace=True)

from sklearn.metrics import roc_auc_score
pred_full=gbc.predict_proba(val_data_var)
roc_auc_score(val_target, pred_full[:,1])

#check association
correlation_matrix=training_set.corr(method='spearman', min_periods=1)
correlation_matrix=training_set.corr(method='kendall', min_periods=1)
association_vector=pd.DataFrame(correlation_matrix.loc[:,'is_active'])
association_vector.sort_values(by='is_active')
#core models with only high corrlation var
train_data_var_1=training_set.loc[:,['hour','pageviews','follows', 'searches']]

rf_core=RandomForestClassifier(n_estimators=1000,max_features='sqrt',class_weight='balanced').fit(train_data_var_1,train_target)
val_data_var_1=validation_set.loc[:,['hour','pageviews','follows', 'searches']]
pred=rf_core.predict(val_data_var_1)

pred_1=rf_core.predict_proba(val_data_var_1)
roc_auc_score(val_target, pred_1[:,1])

val_tar=validation_set['is_active']
from sklearn.metrics import f1_score
f1_score(val_tar, pred) 
from sklearn.metrics import roc_auc_score
roc_auc_score(val_tar, pred)

#check other method to see if this low AUC is due to random forest or not?
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100)
gbc.fit(train_data_var_1,train_target)
y_pred_gb = gbc.predict_proba(val_data_var_1)
roc_auc_score(val_tar, y_pred_gb[:,1])

from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression()
lrc.fit(train_data_var_1,train_target)
y_pred_lr = lrc.predict(val_data_var_1)
roc_auc_score(val_tar, y_pred_lr)

#extended models with high/medium correlation var
train_data_var_2=training_set.loc[:,['hour','pageviews','follows', 'searches','likes','reblogs','original_posts','unfollows','is_verified'
                                     ,'received_engagments']]
#extended models with high/medium correlation var
rf_extend=RandomForestClassifier(n_estimators=1000,max_features='sqrt',class_weight='balanced').fit(train_data_var_2,train_target)

val_data_var_2=validation_set.loc[:,['hour','pageviews','follows', 'searches','likes','reblogs','original_posts','unfollows','is_verified'
                                     ,'received_engagments']]

pred2=rf_extend.predict(val_data_var_2)

pred_2=rf_extend.predict_proba(val_data_var_2)
roc_auc_score(val_target, pred_2[:,1])


val_tar=validation_set['is_active']
from sklearn.metrics import f1_score
f1_score(val_tar, pred2) 
from sklearn.metrics import roc_auc_score
roc_auc_score(val_tar, pred2)

#check other method to see if this low AUC is due to random forest or not?
from sklearn.ensemble import GradientBoostingClassifier
gbc2 = GradientBoostingClassifier(n_estimators=100)
gbc2.fit(train_data_var_2,train_target)
y_pred_gb2 = gbc2.predict(val_data_var_2)
roc_auc_score(val_tar, y_pred_gb2)

y_pred_gb = gbc2.predict_proba(val_data_var_2)
roc_auc_score(val_tar, y_pred_gb[:,1])

#CART indicator
from sklearn import tree
tree_model=tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=2, min_samples_split=2)
tree_model=tree_model.fit(train_data_var_1, train_target)


#######Codes from Stackoverflow and worked#############
def get_code(tree, feature_names):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node):
                if (threshold[node] != -2):
                        print ("if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node])
                        print ("} else {")
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node])
                        print ("}")
                else:
                        print ("return " + str(value[node]))

        recurse(left, right, threshold, features, 0)

get_code(tree_model, train_data_var_1.columns)

#validation of cart model
low_data=validation_set[(validation_set.pageviews<5)&(validation_set.follows<=3)] 
med_data=validation_set[((validation_set.pageviews<5)&(validation_set.follows>3))|((validation_set.pageviews>=5)&(validation_set.pageviews<=10))] 
high_data=validation_set[validation_set.pageviews>10] 

sum(validation_set['is_active'])
sum(low_data['is_active'])
sum(med_data['is_active'])
sum(high_data['is_active'])

#gain chart (with 10 deciles)
def gain_chart(data_in, actual_vec, pred_vec):
    data_in.sort_values(by=pred_vec, ascending=False, inplace=True)
    data_in.reset_index(inplace=True)
    data_in.drop('index', axis=1,inplace=True)
    cut=int(len(data_in)/10)
    total_event=sum(data_in[actual_vec])
    gain_vec=[]
    for i in range(0,9):
        start=cut*i
        end=cut*(i+1)-1
        cut_event=sum(data_in.loc[start:end, actual_vec])
        cut_event_pct=cut_event/(total_event+0.0)
        gain_vec.append(cut_event_pct)
        
    cut_event=sum(data_in.loc[(9*cut):len(data_in)-1, actual_vec])
    cut_event_pct=cut_event/(total_event+0.0)
    gain_vec.append(cut_event_pct)
    gain_vec2=np.cumsum(gain_vec)
    return gain_vec2

#Plot Gain Chart
pred=pd.DataFrame(pred_1[:,1])
pred.columns=['pred']
actual=pd.DataFrame(val_target).reset_index()['is_active']
pred_result=pd.concat([actual, pred], axis=1)
pred_result.columns=['actual','pred']        
gain_vec=gain_chart(pred_result,'actual','pred')

pred=pd.DataFrame(y_pred_gb[:,1])
actual=pd.DataFrame(val_target).reset_index()['is_active']
pred_result=pd.concat([actual, pred], axis=1)
pred_result.columns=['actual','pred']        
gain_vec=gain_chart(pred_result,'actual','pred')


pred=pd.DataFrame(pred_2[:,1])
pred.columns=['pred']
actual=pd.DataFrame(val_target).reset_index()['is_active']
pred_result=pd.concat([actual, pred], axis=1)
pred_result.columns=['actual','pred']        
gain_vec=gain_chart(pred_result,'actual','pred')

pred=pd.DataFrame(y_pred_gb[:,1])
actual=pd.DataFrame(val_target).reset_index()['is_active']
pred_result=pd.concat([actual, pred], axis=1)
pred_result.columns=['actual','pred']        
gain_vec=gain_chart(pred_result,'actual','pred')

#appendix (univariate plot)
def univariate_plot(data, n_bin, response, var, alpha=0.5):
    #Step 1 - Sort the data
    var_data=data.loc[:,[response,var]]
    var_data.sort_values(by=var,ascending=True, inplace=True)
    var_data.reset_index(inplace=True, drop=True)
    #Step 2 - Create a bin for missing value if any
    missing_len=len(data.loc[pd.isnull(data.loc[:,var])])
    bin_var=[]
    bin_response=[]
    bin_count=[]
    if missing_len>0:
        bin_count.append(missing_len)
        bin_var.append(-999999) 
        sum_response=sum(data.loc[pd.isnull(data.loc[:,var]), response])
        bin_response.append(np.log((sum_response+alpha)/(missing_len-sum_response+alpha)))
    
    #Step 3 - Create bins, counts, mean_var, and elogit
    var_data.dropna(axis=0, how='any', inplace=True)
    length=len(var_data)
    n_piece=length/n_bin
    for i in range(0,n_bin,1):
        if i<n_bin-1:
            bin_count.append(n_piece)
            bin_var.append(sum(var_data.loc[i*n_piece:(i+1)*n_piece-1,var])/n_piece)
            sum_response=sum(var_data.loc[i*n_piece:(i+1)*n_piece-1,response])
            bin_response.append(np.log((sum_response+alpha)/(n_piece-sum_response+alpha)))
        else:
            n_final=length-i*n_piece
            bin_count.append(n_final)
            bin_var.append(sum(var_data.loc[i*n_piece:length-1,var])/n_final)
            sum_response=sum(var_data.loc[i*n_piece:length-1,response])
            bin_response.append(np.log((sum_response+alpha)/(n_final-sum_response+alpha)))
    #Step 4: Create and return output
    output=pd.DataFrame({'counts' :bin_count,'mean_var' :bin_var,'mean_resp':bin_response})
    return output

output_pageviews=univariate_plot(training_set, 20, 'is_active', 'pageviews')
output_follows=univariate_plot(training_set, 20, 'is_active', 'follows')
output_searches=univariate_plot(training_set, 20, 'is_active', 'searches')
output_hour=univariate_plot(training_set, 20, 'is_active', 'hour')
