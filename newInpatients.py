#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:26:48 2020

@author: stacy
"""


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import KMeans
import datetime as dt
from sklearn import preprocessing
outpatients_df = pd.read_csv('outpatients.csv')
inpatients_df = pd.read_csv('inpatients.csv')
providers_df = pd.read_csv('providers.csv')
beneficiary_df = pd.read_csv('beneficiary.csv')


# providers_df.groupby('Fraud').size()

tmp_inpatients_df = pd.read_csv('inpatients.csv')
# inpatients_df['StartDt'].dtypes
inpatients_df['StartDt']= pd.to_datetime(inpatients_df['StartDt'])
inpatients_df['EndDt']= pd.to_datetime(inpatients_df['EndDt'])

inpatients_df['ClaimDuration'] = inpatients_df['EndDt'] - inpatients_df['StartDt']
inpatients_df['ClaimDuration'] =inpatients_df['ClaimDuration'].dt.components.days


inpatients_df['AdmissionDt']= pd.to_datetime(inpatients_df['AdmissionDt'])
inpatients_df['DischargeDt']= pd.to_datetime(inpatients_df['DischargeDt'])

inpatients_df['HospitalStay']= inpatients_df['DischargeDt'] - inpatients_df['AdmissionDt']
inpatients_df['HospitalStay'] =inpatients_df['HospitalStay'].dt.components.days

#Add a total diagnosis code column
inpatients_df[['AttendingPhysician', 'OperatingPhysician', 
              'OtherPhysician']] = np.where(inpatients_df[['AttendingPhysician', 
                                                          'OperatingPhysician', 
                                                          'OtherPhysician',]].isnull(), 0, 1)                                                                                               
inpatients_df['numPhysicians'] = inpatients_df['AttendingPhysician'] + inpatients_df['OperatingPhysician']+inpatients_df['OtherPhysician']

inpatients_df=inpatients_df.fillna(0)
inpatients_df.isna().any()
#Add a total diagnosis code column
tmp_inpatients_df[['DiagnosisCode_1', 'DiagnosisCode_2', 'DiagnosisCode_3',
       'DiagnosisCode_4', 'DiagnosisCode_5', 'DiagnosisCode_6',
       'DiagnosisCode_7', 'DiagnosisCode_8', 'DiagnosisCode_9',
       'DiagnosisCode_10',]] = np.where(tmp_inpatients_df[['DiagnosisCode_1', 'DiagnosisCode_2', 'DiagnosisCode_3',
       'DiagnosisCode_4', 'DiagnosisCode_5', 'DiagnosisCode_6',
       'DiagnosisCode_7', 'DiagnosisCode_8', 'DiagnosisCode_9',
       'DiagnosisCode_10',]].isnull(), 0, 1)                                                                                                 
tmp_inpatients_df['numDiagnosisCode'] = tmp_inpatients_df['DiagnosisCode_1'] + tmp_inpatients_df['DiagnosisCode_2']+tmp_inpatients_df['DiagnosisCode_3'] + tmp_inpatients_df['DiagnosisCode_4']+tmp_inpatients_df['DiagnosisCode_5'] + tmp_inpatients_df['DiagnosisCode_6']+tmp_inpatients_df['DiagnosisCode_6'] + tmp_inpatients_df['DiagnosisCode_8']+tmp_inpatients_df['DiagnosisCode_9'] + tmp_inpatients_df['DiagnosisCode_10']

inpatients_df['numDiagnosisCode'] = tmp_inpatients_df['numDiagnosisCode']



tmp_inpatients_df[['ProcedureCode_1',
       'ProcedureCode_2', 'ProcedureCode_3', 'ProcedureCode_4',
       'ProcedureCode_5', 'ProcedureCode_6',]] = np.where(tmp_inpatients_df[['ProcedureCode_1',
       'ProcedureCode_2', 'ProcedureCode_3', 'ProcedureCode_4',
       'ProcedureCode_5', 'ProcedureCode_6',]].isnull(), 0, 1)                                                                                                 
tmp_inpatients_df['numProcedureCode'] = tmp_inpatients_df['ProcedureCode_1'] + tmp_inpatients_df['ProcedureCode_2']+tmp_inpatients_df['ProcedureCode_3'] + tmp_inpatients_df['ProcedureCode_4']+tmp_inpatients_df['ProcedureCode_5'] + tmp_inpatients_df['ProcedureCode_6']
inpatients_df['numProcedureCode'] = tmp_inpatients_df['numProcedureCode']



# inpatients_df['DiagnosisCode_10']=inpatients_df['DiagnosisCode_10'].astype(str)
# inpatients_df['AdmitDiagnosisCode']=inpatients_df['AdmitDiagnosisCode'].astype(str)
# codelist=inpatients_df['DiagnosisCode_10']
# codelist = codelist.append(inpatients_df['AdmitDiagnosisCode'])
# for i in range(1,10):
#     inpatients_df['DiagnosisCode_'+str(i)]=inpatients_df['DiagnosisCode_'+str(i)].astype(str)
#     tmplist = inpatients_df['DiagnosisCode_'+str(i)]
#     codelist = codelist.append(tmplist)

# le = preprocessing.LabelEncoder()
# le.fit(codelist)
# le.classes_
# inpatients_df['AdmitDiagnosisCode'] = le.transform(inpatients_df['AdmitDiagnosisCode'])

# for i in range(1,11):
#     inpatients_df['DiagnosisCode_'+str(i)]=le.transform(inpatients_df['DiagnosisCode_'+str(i)])



# inpatients_df['ProcedureCode_6']=inpatients_df['ProcedureCode_6'].astype(str)
# codelist=inpatients_df['ProcedureCode_6']
# for i in range(1,6):
#     inpatients_df['ProcedureCode_'+str(i)]=inpatients_df['ProcedureCode_'+str(i)].astype(str)
#     tmplist = inpatients_df['ProcedureCode_'+str(i)]
#     codelist = codelist.append(tmplist)

# le = preprocessing.LabelEncoder()
# le.fit(codelist)
# le.classes_

# for i in range(1,7):
#     inpatients_df['ProcedureCode_'+str(i)]=le.transform(inpatients_df['ProcedureCode_'+str(i)])


# inpatients_df.isna().any()
# inpatients_df.columns
col = ['BID','PID','CID', 'AmtReimbursed',
       'AdmitDiagnosisCode', 'DeductibleAmt',
       # 'DiagnosisCode_1', 'DiagnosisCode_2',
       # 'DiagnosisCode_3', 'DiagnosisCode_4', 'DiagnosisCode_5',
       # 'DiagnosisCode_6', 'DiagnosisCode_7', 'DiagnosisCode_8',
       # 'DiagnosisCode_9', 'DiagnosisCode_10', 'ProcedureCode_1',
       # 'ProcedureCode_2', 'ProcedureCode_3', 'ProcedureCode_4',
       # 'ProcedureCode_5', 
       'ClaimDuration', 'HospitalStay',
       'numPhysicians', 'numDiagnosisCode','numProcedureCode',
       'existFrequentFraudCode']

#Add highly fraudulent diagnosis code

# df_patients = inpatients_df.copy()
# df_codeCounts = pd.DataFrame()
# for i in range(1,11):
#     df_codeCounts= pd.concat([df_codeCounts, df_patients['DiagnosisCode_'+str(i)].value_counts()],
#                               )
# df_codeCounts=df_codeCounts.reset_index()
# df_codeCounts = df_codeCounts.groupby(['index']).agg({0: 'sum'})
# df_codeCounts=df_codeCounts.reset_index()
# df_codeCounts.columns= ['code','totalCount']
# df_codeCounts.loc[0, 'totalCount'] = 0

# inpatients_df['codeCount']=0
# for i in range(1,11):
#     df_codeCounts.columns= ['DiagnosisCode_'+str(i),'totalCount_' +str(i)]
#     inpatients_df = pd.merge(inpatients_df, df_codeCounts[['DiagnosisCode_'+str(i),'totalCount_'+str(i)]],
#                         on=['DiagnosisCode_'+str(i)])
    
    
# inpatients_df['codeCount'] =  inpatients_df['totalCount_1'] + inpatients_df['totalCount_2']+ inpatients_df['totalCount_3'] +inpatients_df['totalCount_4'] + inpatients_df['totalCount_5'] + inpatients_df['totalCount_6'] +   inpatients_df['totalCount_7'] + inpatients_df['totalCount_8'] + inpatients_df['totalCount_9'] + inpatients_df['totalCount_10'] 




# df_patients = pd.merge(df_patients, providers_df[['PID','Fraud']],
#                         on=['PID'])
# print(df_patients.groupby('Fraud').size())

# tmp = df_patients[df_patients['Fraud']=='Yes']

# df_codeCounts_fraud = pd.DataFrame()
# for i in range(1,11):
#     df_codeCounts_fraud= pd.concat([df_codeCounts_fraud, tmp['DiagnosisCode_'+str(i)].value_counts()])
# df_codeCounts_fraud=df_codeCounts_fraud.reset_index()
# df_codeCounts_fraud = df_codeCounts_fraud.groupby(['index']).agg({0: 'sum'})
# df_codeCounts_fraud=df_codeCounts_fraud.reset_index()        
# df_codeCounts_fraud.columns= ['code','fraudCount']


# codeCount_df = pd.merge(df_codeCounts_fraud, df_codeCounts[['code','totalCount']],
#                         on=['code'])
# codeCount_df['Fraud%']=codeCount_df['fraudCount']/codeCount_df['totalCount']

# xx = codeCount_df.loc[codeCount_df['Fraud%']>=0.8]


frequentFraudCode = ['73381','73319','73024','30430','25081',
                     '6253','4610','3180','2865','2825'] #over 80%

for i in range(1,11):
    inpatients_df['DiagnosisCode_'+str(i)]=inpatients_df['DiagnosisCode_'+str(i)].astype(str)


# inpatients_df['existFrequentFraudCode']=inpatients_df['AdmitDiagnosisCode'].apply(lambda x: 1 if x=='73381' else 0)

inpatients_df['existFrequentFraudCode']=0
for i in range(1,11):
    inpatients_df['existFrequentFraudCode_'+str(i)]=inpatients_df['DiagnosisCode_'+str(i)].apply(lambda x: 1 
                                                                               if (x in frequentFraudCode) else 0)

inpatients_df['existFrequentFraudCode'] =  inpatients_df['existFrequentFraudCode_1'] + inpatients_df['existFrequentFraudCode_2']+ inpatients_df['existFrequentFraudCode_3'] +inpatients_df['existFrequentFraudCode_4'] + inpatients_df['existFrequentFraudCode_5'] + inpatients_df['existFrequentFraudCode_6'] +   inpatients_df['existFrequentFraudCode_7'] + inpatients_df['existFrequentFraudCode_8'] + inpatients_df['existFrequentFraudCode_9'] + inpatients_df['existFrequentFraudCode_10'] 
inpatients_df['existFrequentFraudCode']= inpatients_df['existFrequentFraudCode'].apply(lambda x: 1 if x >= 1 else 0)
 

inpatients_df[col].to_csv('NewInpatients.csv')
