import pandas as pd
import numpy as np
import scipy.stats as stats

heart = pd.read_csv('heart_disease.csv')
yes_hd = heart[heart.heart_disease == 'presence']
no_hd = heart[heart.heart_disease == 'absence']

chol_hd = yes_hd.chol
chol_no_hd = no_hd.chol
chol_mean = np.mean(chol_hd)
chol_no_mean = np.mean(chol_no_hd)

stat, pval = stats.ttest_1samp(chol_hd, 240)
stat, pval = stats.ttest_1samp(chol_no_hd, 240)


num_patience = len(heart.age)

num_highfbs_patients = np.sum(heart['fbs'] == 1.0)

pval = stats.binomtest(num_highfbs_patients, n=num_patience, p=0.08, alternative='greater')

print(pval)
