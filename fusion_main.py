from autoencoder import autoencoder_main
from cnnlstm import cnnlstm
from icecream import ic
import numpy as np

multivariable_anomaly = autoencoder_main()
abnormal_counts, len_of_data = cnnlstm()

univariate_count = np.array(list(abnormal_counts.values())).T

sj_wts = np.array([0.0, 1.0, 5.0, 3.0])
sj = np.dot(sj_wts, univariate_count)

smul = (multivariable_anomaly/len_of_data)*100

beta = [1.0, 1.2, .8, 0.5]
sladle = smul + np.dot(beta, sj)
print("Final anomaly score: ", sladle)
sthreshold = 5
if sladle<sthreshold:
    print("Normal process!")