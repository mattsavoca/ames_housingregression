# -*- coding: utf-8 -*-
"""
Plot different submissions against each other
"""

# Libraries
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd

# Datas
data1 = pd.read_csv('./GBR_MattCAT_Submission.csv')
data2 = pd.read_csv('./RFDUM_Submission.csv')

# Plot
plt.figure(figsize=(20,10)) 
plt.plot(data1['SalePrice']-data2['SalePrice'], '-')
plt.show()
