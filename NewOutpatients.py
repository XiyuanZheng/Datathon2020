#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:23:14 2020

@author: stacy
"""


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

tmp_outpatients_df = pd.read_csv('outpatients.csv')
# inpatients_df['StartDt'].dtypes
outpatients_df['StartDt']= pd.to_datetime(outpatients_df['StartDt'])
outpatients_df['EndDt']= pd.to_datetime(outpatients_df['EndDt'])

outpatients_df['ClaimDuration'] = outpatients_df['EndDt'] - outpatients_df['StartDt']
outpatients_df['ClaimDuration'] =outpatients_df['ClaimDuration'].dt.components.days
outpatients_df.columns


#Add a total diagnosis code column
outpatients_df[['AttendingPhysician', 'OperatingPhysician', 
              'OtherPhysician']] = np.where(outpatients_df[['AttendingPhysician', 
                                                          'OperatingPhysician', 
                                                          'OtherPhysician',]].isnull(), 0, 1)                                                                                               
outpatients_df['numPhysicians'] = outpatients_df['AttendingPhysician'] + outpatients_df['OperatingPhysician']+outpatients_df['OtherPhysician']

outpatients_df=outpatients_df.fillna(0)
outpatients_df.isna().any()
#Add a total diagnosis code column
tmp_outpatients_df[['DiagnosisCode_1', 'DiagnosisCode_2', 'DiagnosisCode_3',
       'DiagnosisCode_4', 'DiagnosisCode_5', 'DiagnosisCode_6',
       'DiagnosisCode_7', 'DiagnosisCode_8', 'DiagnosisCode_9',
       'DiagnosisCode_10',]] = np.where(tmp_outpatients_df[['DiagnosisCode_1', 'DiagnosisCode_2', 'DiagnosisCode_3',
       'DiagnosisCode_4', 'DiagnosisCode_5', 'DiagnosisCode_6',
       'DiagnosisCode_7', 'DiagnosisCode_8', 'DiagnosisCode_9',
       'DiagnosisCode_10',]].isnull(), 0, 1)                                                                                                 
tmp_outpatients_df['numDiagnosisCode'] = tmp_outpatients_df['DiagnosisCode_1'] + tmp_outpatients_df['DiagnosisCode_2']+tmp_outpatients_df['DiagnosisCode_3'] + tmp_outpatients_df['DiagnosisCode_4']+tmp_outpatients_df['DiagnosisCode_5'] + tmp_outpatients_df['DiagnosisCode_6']+tmp_outpatients_df['DiagnosisCode_6'] + tmp_outpatients_df['DiagnosisCode_8']+tmp_outpatients_df['DiagnosisCode_9'] + tmp_outpatients_df['DiagnosisCode_10']

outpatients_df['numDiagnosisCode'] = tmp_outpatients_df['numDiagnosisCode']



tmp_outpatients_df[['ProcedureCode_1',
       'ProcedureCode_2', 'ProcedureCode_3', 'ProcedureCode_4',
       'ProcedureCode_5', 'ProcedureCode_6',]] = np.where(tmp_outpatients_df[['ProcedureCode_1',
       'ProcedureCode_2', 'ProcedureCode_3', 'ProcedureCode_4',
       'ProcedureCode_5', 'ProcedureCode_6',]].isnull(), 0, 1)                                                                                                 
tmp_outpatients_df['numProcedureCode'] = tmp_outpatients_df['ProcedureCode_1'] + tmp_outpatients_df['ProcedureCode_2']+tmp_outpatients_df['ProcedureCode_3'] + tmp_outpatients_df['ProcedureCode_4']+tmp_outpatients_df['ProcedureCode_5'] + tmp_outpatients_df['ProcedureCode_6']
outpatients_df['numProcedureCode'] = tmp_outpatients_df['numProcedureCode']



# outpatients_df['DiagnosisCode_10']=outpatients_df['DiagnosisCode_10'].astype(str)
# outpatients_df['AdmitDiagnosisCode']=outpatients_df['AdmitDiagnosisCode'].astype(str)
# codelist=outpatients_df['DiagnosisCode_10']
# codelist = codelist.append(outpatients_df['AdmitDiagnosisCode'])
# for i in range(1,10):
#     outpatients_df['DiagnosisCode_'+str(i)]=outpatients_df['DiagnosisCode_'+str(i)].astype(str)
#     tmplist = outpatients_df['DiagnosisCode_'+str(i)]
#     codelist = codelist.append(tmplist)

# le = preprocessing.LabelEncoder()
# le.fit(codelist)
# le.classes_
# outpatients_df['AdmitDiagnosisCode'] = le.transform(outpatients_df['AdmitDiagnosisCode'])

# for i in range(1,11):
#     outpatients_df['DiagnosisCode_'+str(i)]=le.transform(outpatients_df['DiagnosisCode_'+str(i)])



# outpatients_df['ProcedureCode_6']=outpatients_df['ProcedureCode_6'].astype(str)
# codelist=outpatients_df['ProcedureCode_6']
# for i in range(1,6):
#     outpatients_df['ProcedureCode_'+str(i)]=outpatients_df['ProcedureCode_'+str(i)].astype(str)
#     tmplist = outpatients_df['ProcedureCode_'+str(i)]
#     codelist = codelist.append(tmplist)

# le = preprocessing.LabelEncoder()
# le.fit(codelist)
# le.classes_

# for i in range(1,7):
#     outpatients_df['ProcedureCode_'+str(i)]=le.transform(outpatients_df['ProcedureCode_'+str(i)])


#existFrequentFraudCode
frequentFraudCode = ['V401','87364','81402','81302','80842','79506','37420','36543',
                     '36119','33390','33371','33181','30410','20960','20712','20215',
                     '9930','9565','9515','5695','31'] #over 80%


for i in range(1,11):
    outpatients_df['DiagnosisCode_'+str(i)]=outpatients_df['DiagnosisCode_'+str(i)].astype(str)


# inpatients_df['existFrequentFraudCode']=inpatients_df['AdmitDiagnosisCode'].apply(lambda x: 1 if x=='73381' else 0)

outpatients_df['existFrequentFraudCode']=0
for i in range(1,11):
    outpatients_df['existFrequentFraudCode_'+str(i)]=outpatients_df['DiagnosisCode_'+str(i)].apply(lambda x: 1 
                                                                               if (x in frequentFraudCode) else 0)

outpatients_df['existFrequentFraudCode'] =  outpatients_df['existFrequentFraudCode_1'] + outpatients_df['existFrequentFraudCode_2']+ outpatients_df['existFrequentFraudCode_3'] +outpatients_df['existFrequentFraudCode_4'] + outpatients_df['existFrequentFraudCode_5'] + outpatients_df['existFrequentFraudCode_6'] + outpatients_df['existFrequentFraudCode_7'] + outpatients_df['existFrequentFraudCode_8'] + outpatients_df['existFrequentFraudCode_9'] + outpatients_df['existFrequentFraudCode_10'] 
outpatients_df['existFrequentFraudCode']= outpatients_df['existFrequentFraudCode'].apply(lambda x: 1 if x >= 1 else 0)
 
# # add code count
# df_patients =outpatients_df.copy()
# df_codeCounts = pd.DataFrame()
# for i in range(1,11):
#     df_codeCounts= pd.concat([df_codeCounts, df_patients['DiagnosisCode_'+str(i)].value_counts()],
#                               )
# df_codeCounts=df_codeCounts.reset_index()
# df_codeCounts = df_codeCounts.groupby(['index']).agg({0: 'sum'})
# df_codeCounts=df_codeCounts.reset_index()
# df_codeCounts.columns= ['code','totalCount']
# df_codeCounts.loc[0, 'totalCount'] = 0

# outpatients_df['codeCount']=0
# for i in range(1,11):
#     df_codeCounts.columns= ['DiagnosisCode_'+str(i),'totalCount_' +str(i)]
#     outpatients_df = pd.merge(outpatients_df, df_codeCounts[['DiagnosisCode_'+str(i),'totalCount_'+str(i)]],
#                         on=['DiagnosisCode_'+str(i)])
    
    
# outpatients_df['codeCount'] =  outpatients_df['totalCount_1'] + outpatients_df['totalCount_2']+ outpatients_df['totalCount_3'] +outpatients_df['totalCount_4'] + outpatients_df['totalCount_5'] + outpatients_df['totalCount_6'] + outpatients_df['totalCount_7'] + outpatients_df['totalCount_8'] + outpatients_df['totalCount_9'] + outpatients_df['totalCount_10'] 


outpatients_df.isna().any()
outpatients_df.columns
col = ['BID','PID','CID', 'AmtReimbursed',
       'AdmitDiagnosisCode', 'DeductibleAmt',
       'DiagnosisCode_1', 'DiagnosisCode_2',
       'DiagnosisCode_3', 'DiagnosisCode_4', 'DiagnosisCode_5',
       'DiagnosisCode_6', 'DiagnosisCode_7', 'DiagnosisCode_8',
       'DiagnosisCode_9', 'DiagnosisCode_10', 'ProcedureCode_1',
       'ProcedureCode_2', 'ProcedureCode_3', 'ProcedureCode_4',
       'ProcedureCode_5', 'ProcedureCode_6', 'ClaimDuration', 
       'numPhysicians', 'numDiagnosisCode','numProcedureCode',
       'existFrequentFraudCode']

outpatients_df[col].to_csv('NewOutpatients.csv')

# outpatients_df[col].head()
