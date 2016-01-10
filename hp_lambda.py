from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from numpy.linalg import inv
from statsmodels.tsa.filters.hp_filter import hpfilter
from time import clock



#*********************************************************************
# HP Filter with standard Lambda, (Lambda =14,400)
# Normal Rule of thumb is Lambda = 100*(number of periods in a year)^2
# For Quarterly Data = 100 * 4^2 = 1600
# For Monthly Data = 100*12^2 = 14400
#*********************************************************************

#**************************************************************************
# Modified HP Filter - Determining Optimal Lambda
# Ratio of variance of cyclical component TO Change in growth of trend  
# cannot be assumed to be same across all countries. 
# To address this, we use Leave One-out Cross Validation Method (LOOCV)
# proposed by Craven and Wahba (1979) and developed by Mcdermott (1997).
# To see more details please read the following paper below:
# http://mpra.ub.uni-muenchen.de/45630/1/MPRA_paper_45630.pdf (Pages 3-5)
#**************************************************************************


def optimum_lambda(df, lamda_min, lamda_max):
    """Determining optimum value of lamda"""
    T = df.size
    lamda1 = lamda_min # lower limit of lamda for iteration
    lamda2 = lamda_max # upper limit of lamda for iteration
    step = 100  # step fo riteration
    Gtk = np.zeros(T)
    GCV = np.zeros((lamda2-lamda1)/step+1)
    ffr_idx = df.columns[0]
    ffr = df[ffr_idx].values # FFR column from input

    A = np.zeros((T-2, T))

    for i in range(T-2):
        for j in range(T):
            if i == j or (i+2) == j:
                A[i,j] = 1
            elif (i+1) == j:
                A[i,j] = -2

    k = 0
    for lamda in range(lamda1, lamda2+1, step):
        for i in range(T):
            As = np.delete(A, i, axis=1)
            la = np.dot(inv(np.eye(T-1) - lamda*(np.dot(As.T, As))), np.delete(ffr, i))
            Gtk[i] = np.sum(la)
        GCV[k] = (1/T + 2/lamda)*np.sum((ffr-Gtk)*(ffr-Gtk))
        k += 1

    # Value of lamda corresponding to min GCV value
    lamdamin = lamda1 + np.argmin(GCV)*step
    return lamdamin



# Reading in the Input File
input_name = 'INPUT.csv'
df = pd.read_csv(input_name, index_col=0,parse_dates=True)

 #Determine optimum
t1=clock()
opt_lambda = optimum_lambda(df, 10000, 20000)
t2=clock()
print ('Time: {0} s'.format(t2-t1))
print(opt_lambda)

 #Trend component for optimum lamda
_, trend = hpfilter(df[1:],opt_lambda)
