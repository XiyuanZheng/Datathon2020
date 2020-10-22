#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:59:25 2020

@author: stacy
"""


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.cluster import AffinityPropagation
#Load data
providers_df = pd.read_csv('providers.csv')
beneficiary_df = pd.read_csv('clean_beneficiary.csv')
# insurance_df.head()

#Merge data
inpatients_df = pd.read_csv('newInpatients.csv')
inpatients_df = pd.merge(inpatients_df, providers_df[['PID','Fraud']],
                        on=['PID'])
inpatients_df = pd.merge(inpatients_df,beneficiary_df,on=['BID'])
dropcol = ['Unnamed: 0', 'BID']
col = inpatients_df.columns.difference(dropcol)

inpatients_df = inpatients_df[col]
inpatients_df.columns


#features selection
col= [ 'CID','PID','AmtReimbursed',
        'Chronic_Alzheimer','Chronic_Cancer', 'Chronic_Depression', 'Chronic_Diabetes',
         'Chronic_Heartfailure', 'Chronic_IschemicHeart','Chronic_KidneyDisease', 
         'Chronic_ObstrPulmonary','Chronic_Osteoporasis', 'Chronic_rheumatoidarthritis', 
         'Chronic_stroke','Gender',
        'ClaimDuration', 'DOD', 'DeductibleAmt', 'age',
        #'AdmitDiagnosisCode',
       # 'DiagnosisCode_1','DiagnosisCode_10', 'DiagnosisCode_2', 'DiagnosisCode_3',
       # 'DiagnosisCode_4', 'DiagnosisCode_5', 'DiagnosisCode_6',
       # 'DiagnosisCode_7', 'DiagnosisCode_8', 'DiagnosisCode_9', 
       'Fraud', 'HospitalStay', 
        'InpatientAnnualDeductibleAmt','InpatientAnnualReimbursementAmt', 
        'NumOfMonths_PartACov', 'NumOfMonths_PartBCov', 
        'OutpatientAnnualDeductibleAmt','OutpatientAnnualReimbursementAmt', 
       
       # 'ProcedureCode_1','ProcedureCode_2', 'ProcedureCode_3', 'ProcedureCode_4',
       # 'ProcedureCode_5', 'ProcedureCode_6', 
        'RenalDisease', 'numDiagnosisCode', 'numPhysicians', 'numProcedureCode',
        'existFrequentFraudCode']

X = inpatients_df.copy()
X=X[col]

# Define the binary data columns
binary_feature_columns = ['CID','PID','Chronic_Alzheimer',
       'Chronic_Cancer', 'Chronic_Depression', 'Chronic_Diabetes',
       'Chronic_Heartfailure', 'Chronic_IschemicHeart',
       'Chronic_KidneyDisease', 'Chronic_ObstrPulmonary',
       'Chronic_Osteoporasis', 'Chronic_rheumatoidarthritis', 'Chronic_stroke',
       'ClaimDuration', 'DOD','Gender','RenalDisease','Fraud','existFrequentFraudCode']
non_binary_columns = X.columns.difference(binary_feature_columns)


#scaling
def normalize(X, binary_columns):
    # Function that normalizes the non binary data so that across all data 
    # Define scaling function
    X_scaled = X.copy()
    scaler = StandardScaler()
    
    # Define columns that are non-binary
    non_binary_columns = X.columns.difference(binary_columns)
  
    # Scale the X features      
    X_scaled[non_binary_columns] = scaler.fit_transform(X_scaled[non_binary_columns])
              
    return X_scaled, scaler

X_scaled,scaler = normalize(X,binary_feature_columns)

#drop fraud col
col = X_scaled.columns.difference(['Fraud','CID','PID'])

#Model
def kmeansModel(df,n,col):
    X=df[col].copy()
    kmeans = KMeans(n_clusters=n, random_state = 1)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    SSE = kmeans.inertia_
    print('SSE = ' + str(SSE))
    
    dists = euclidean_distances(kmeans.cluster_centers_)
    print(dists)
    centers_df = pd.DataFrame(kmeans.cluster_centers_)
    centers_df.columns = col
    
    return y_kmeans,centers_df

# def affinityModel(df,n,col):
#     X=df[col].copy()
#     mdl = AffinityPropagation().fit(X)
#     y_kmeans = mdl.predict(X)
#     SSE = mdl.inertia_
#     print('SSE = ' + str(SSE))
    
#     dists = euclidean_distances(mdl.cluster_centers_)
#     print(dists)
#     return y_kmeans

#elbow inpatient
number_of_clusters = list(range(1,8))
average_distance = []
for k in number_of_clusters: 
    mdk_k_means = KMeans(n_init = 1,  # number of different centroid seed initializations (number of times algorithm is run)
                   n_clusters=k,  # number of clusters (k)
                   random_state = 1)  # random seed for k-means algorithm
    mdk_k_means.fit(X_scaled[col])
    clK = mdk_k_means.labels_
    distances_from_clusters = mdk_k_means.transform(X_scaled[col]).min(axis=1)
    average_distance.append(distances_from_clusters.mean())

plt.figure() #figsize=(4,6)
plt.scatter(number_of_clusters,average_distance)
plt.ylabel('Average centroid to point euclidien distances')
plt.xlabel('Number of clusters')
plt.title('Elbow plot')
plt.show()
     


k = 4
y_kmeans,centers_df = kmeansModel(X_scaled,k,col)
# y_kmeans = affinityModel(X_scaled,k,col)
centers_df[non_binary_columns] = pd.DataFrame(scaler.inverse_transform(centers_df[non_binary_columns].values), 
                     columns=non_binary_columns)
# centers_df.to_csv('inpatient_centers.csv')



newX = X_scaled.copy()
newX[non_binary_columns] = pd.DataFrame(scaler.inverse_transform(X_scaled[non_binary_columns].values), 
                     columns=non_binary_columns)
#results
newX['kmeans'] = y_kmeans
newX.columns
for i in range(0,k):
    tmpX = newX.loc[newX['kmeans']==i]
    TotalAmt = tmpX['AmtReimbursed'].sum()
    FraudAmt = tmpX.loc[tmpX['Fraud']=='Yes']['AmtReimbursed'].sum()
    avgFraudAmt = tmpX.loc[tmpX['Fraud']=='Yes']['AmtReimbursed'].mean()
    FraudAmtPercentage = FraudAmt/TotalAmt
    
    noFraud = tmpX.groupby('Fraud').size()[0]
    Fraud = tmpX.groupby('Fraud').size()[1]
    FraudPercent = Fraud/(noFraud+Fraud)
    numClaims = tmpX.index.size
    print('K = '+str(i)+', FraudPercentage: '+ str(FraudPercent) 
          + ', FraudAmtPercentage: '+ str(FraudAmtPercentage) + ', avgFraudAmt: '
          + str(avgFraudAmt) + ', num of Claims in this group: '+ str(numClaims))

# tmp = newX.loc[newX['kmeans']==3]
# tmp['numProcedureCode'].mean()

outputCol= ['CID','PID','AmtReimbursed', 'Chronic_Alzheimer', 'Chronic_Cancer',
       'Chronic_Depression', 'Chronic_Diabetes', 'Chronic_Heartfailure',
       'Chronic_IschemicHeart', 'Chronic_KidneyDisease',
       'Chronic_ObstrPulmonary', 'Chronic_Osteoporasis',
       'Chronic_rheumatoidarthritis', 'Chronic_stroke', 'ClaimDuration', 'DOD',
       'DeductibleAmt', 'Gender', 'HospitalStay',
       'InpatientAnnualDeductibleAmt', 'InpatientAnnualReimbursementAmt',
       'NumOfMonths_PartACov', 'NumOfMonths_PartBCov',
       'OutpatientAnnualDeductibleAmt', 'OutpatientAnnualReimbursementAmt',
       'RenalDisease', 'age', 'existFrequentFraudCode', 'numDiagnosisCode',
       'numPhysicians', 'numProcedureCode','Fraud','kmeans']
newX[outputCol].to_csv('inpatientClusters.csv')





#outpatient
outpatients_df = pd.read_csv('NewOutpatients.csv')

#Merge data
outpatients_df = pd.merge(outpatients_df, providers_df[['PID','Fraud']],
                        on=['PID'])
outpatients_df = pd.merge(outpatients_df,beneficiary_df,on=['BID'])
dropcol = ['Unnamed: 0', 'BID']
col = outpatients_df.columns.difference(dropcol)


outpatients_df = outpatients_df[col]
outpatients_df.columns


#features selection
outcol= [ 'CID','PID', 'AmtReimbursed',
       'Chronic_Alzheimer','Chronic_Cancer', 'Chronic_Depression', 'Chronic_Diabetes',
        'Chronic_Heartfailure', 'Chronic_IschemicHeart','Chronic_KidneyDisease', 
        'Chronic_ObstrPulmonary','Chronic_Osteoporasis', 'Chronic_rheumatoidarthritis', 
        'Chronic_stroke','ClaimDuration', 'DOD', 'DeductibleAmt', 
        #'AdmitDiagnosisCode',
       # 'DiagnosisCode_1','DiagnosisCode_10', 'DiagnosisCode_2', 'DiagnosisCode_3',
       # 'DiagnosisCode_4', 'DiagnosisCode_5', 'DiagnosisCode_6',
       # 'DiagnosisCode_7', 'DiagnosisCode_8', 'DiagnosisCode_9', 
       'Fraud','Gender', 'InpatientAnnualDeductibleAmt',
       'InpatientAnnualReimbursementAmt', 'NumOfMonths_PartACov',
       'NumOfMonths_PartBCov', 'OutpatientAnnualDeductibleAmt',
       'OutpatientAnnualReimbursementAmt', 
       # 'ProcedureCode_1','ProcedureCode_2', 'ProcedureCode_3', 'ProcedureCode_4',
       # 'ProcedureCode_5', 'ProcedureCode_6', 
       'RenalDisease', 'age','numDiagnosisCode', 'numPhysicians', 'numProcedureCode',
       'existFrequentFraudCode']

X = outpatients_df.copy()
X=X[outcol]

# Define the binary data columns
binary_feature_columns = ['CID','PID','Chronic_Alzheimer',
       'Chronic_Cancer', 'Chronic_Depression', 'Chronic_Diabetes',
       'Chronic_Heartfailure', 'Chronic_IschemicHeart',
       'Chronic_KidneyDisease', 'Chronic_ObstrPulmonary',
       'Chronic_Osteoporasis', 'Chronic_rheumatoidarthritis', 'Chronic_stroke',
       'ClaimDuration', 'DOD','Gender','RenalDisease','Fraud','existFrequentFraudCode']
non_binary_columns = X.columns.difference(binary_feature_columns)
X_scaled,scaler = normalize(X,binary_feature_columns)

#drop fraud col
col = X_scaled.columns.difference(['Fraud','CID','PID'])

#elbow inpatient
number_of_clusters = list(range(1,10))
average_distance = []
for k in number_of_clusters: 
    mdk_k_means = KMeans(n_init = 1,  # number of different centroid seed initializations (number of times algorithm is run)
                   n_clusters=k,  # number of clusters (k)
                   random_state = 1)  # random seed for k-means algorithm
    mdk_k_means.fit(X_scaled[col])
    clK = mdk_k_means.labels_
    distances_from_clusters = mdk_k_means.transform(X_scaled[col]).min(axis=1)
    average_distance.append(distances_from_clusters.mean())

plt.figure() #figsize=(4,6)
plt.scatter(number_of_clusters,average_distance)
plt.ylabel('Average centroid to point euclidien distances')
plt.xlabel('Number of clusters')
plt.title('Elbow plot')
plt.show()




k = 7
y_kmeans,centers_df = kmeansModel(X_scaled,k,col)

newX = X_scaled.copy()
newX[non_binary_columns] = pd.DataFrame(scaler.inverse_transform(X_scaled[non_binary_columns].values), 
                     columns=non_binary_columns)
#results
newX['kmeans'] = y_kmeans
newX.columns
for i in range(0,k):
    tmpX = newX.loc[newX['kmeans']==i]
    TotalAmt = tmpX['AmtReimbursed'].sum()
    FraudAmt = tmpX.loc[tmpX['Fraud']=='Yes']['AmtReimbursed'].sum()
    avgFraudAmt = tmpX.loc[tmpX['Fraud']=='Yes']['AmtReimbursed'].mean()
    FraudAmtPercentage = FraudAmt/TotalAmt
    
    noFraud = tmpX.groupby('Fraud').size()[0]
    Fraud = tmpX.groupby('Fraud').size()[1]
    FraudPercent = Fraud/(noFraud+Fraud)
    numClaims = tmpX.index.size
    print('K = '+str(i)+', FraudPercentage: '+ str(FraudPercent) 
          + ', FraudAmtPercentage: '+ str(FraudAmtPercentage) + ', avgFraudAmt: '
          + str(avgFraudAmt) + ', num of Claims in this group: '+ str(numClaims))

centers_df[non_binary_columns] = pd.DataFrame(scaler.inverse_transform(centers_df[non_binary_columns].values), 
                     columns=non_binary_columns)
# centers_df.to_csv('outpatient_centers.csv')

outputCol= ['CID','PID','AmtReimbursed', 'Chronic_Alzheimer', 'Chronic_Cancer',
       'Chronic_Depression', 'Chronic_Diabetes', 'Chronic_Heartfailure',
       'Chronic_IschemicHeart', 'Chronic_KidneyDisease',
       'Chronic_ObstrPulmonary', 'Chronic_Osteoporasis',
       'Chronic_rheumatoidarthritis', 'Chronic_stroke', 'ClaimDuration', 'DOD',
       'DeductibleAmt', 'Gender', 
       'InpatientAnnualDeductibleAmt', 'InpatientAnnualReimbursementAmt',
       'NumOfMonths_PartACov', 'NumOfMonths_PartBCov',
       'OutpatientAnnualDeductibleAmt', 'OutpatientAnnualReimbursementAmt',
       'RenalDisease', 'age', 'existFrequentFraudCode', 'numDiagnosisCode',
       'numPhysicians', 'numProcedureCode','Fraud','kmeans']
newX[outputCol].to_csv('outpatientClusters.csv')

# tmp = newX.loc[newX['kmeans']==6]
# AA = pd.DataFrame()
# AA['PID'] = tmp.PID.unique()
# AA = pd.merge(AA, providers_df[['PID','Fraud']],
#                         on=['PID'])
# AA.groupby('Fraud').size()

tmpX = newX.loc[newX['kmeans']==1]
data_fraud = tmpX.loc[tmpX['Fraud']=='Yes']
data_nofraud = tmpX.loc[tmpX['Fraud']=='No']
sns.distplot(data_fraud.HospitalStay.dropna(),kde=False,label='Fraud')
sns.distplot(data_nofraud.HospitalStay.dropna(),kde=False,label='NonFraud')
plt.legend()
plt.title('HospitalStay histogram')
plt.ylabel('frequency')
plt.show()
