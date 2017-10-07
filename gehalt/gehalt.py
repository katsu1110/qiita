# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:42:50 2017

@author: katsuhisa
"""

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# EUR to JPY =======================================
def eur2jpy(x):
    y = np.round(x*130/10000)
    return y.astype(int)

# the size of a company =============================
# list of data
med = np.array([42170,45108,48534])
err = np.array([med-[36167,40257,42925],[47203,49337,55532]-med])

# visualization
plt.figure()
plt.errorbar(np.arange(3), eur2jpy(med), yerr=eur2jpy(err), \
             fmt='--o', linewidth=6, alpha=0.5, c='k')
for i in np.arange(3):
    plt.text(i-0.2,eur2jpy(med[i])+10,str(eur2jpy(med[i])), fontsize=15)
    
plt.xticks(np.arange(3), ['small (< 100)','mediam','large (> 1000)'], fontsize=15)
plt.xlabel('Company Size', fontsize=15)
plt.ylabel(r'Income / year  (x $10^{4}$ JPY)', fontsize=15)
plt.xlim([-0.5, 2.5])
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()

# education history =============================
# list of data
med = np.array([42379,46422,57370])
err = np.array([med-[36835,41968,50967],[47364,51423,62965]-med])

# visualization
plt.figure()
plt.errorbar(np.arange(3), eur2jpy(med), yerr=eur2jpy(err), \
             fmt='--o', linewidth=6, alpha=0.5, c='k')
for i in np.arange(3):
    plt.text(i-0.2,eur2jpy(med[i])+10,str(eur2jpy(med[i])), fontsize=15)
    
plt.xticks(np.arange(3), ['Bachelor','Master','Doctor'], fontsize=15)
plt.xlabel('Education History', fontsize=15)
plt.ylabel(r'Income / year  (x $10^{4}$ JPY)', fontsize=15)
plt.xlim([-0.5, 2.5])
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()

# Field =========================================
# list of data
med = np.array([48700,48040,47600,47114,46499,\
                46455,46011,44762,44555,43433,\
                43379,42379,41258,40584,37876])
err = np.array([med - [44391,42192,40159,42243,39943, \
                40395,42379,40903,36123,39191, \
                34205,36306,33014,35094,32737], \
               [59093,59440,52132,53534,50268, \
                54978,51822,50128,47835,54113, \
                60541,50103,44541,50090,45341] - med])

# visualization
plt.figure()
plt.errorbar(np.arange(15), eur2jpy(med), yerr=eur2jpy(err), \
             fmt='o', linewidth=6, alpha=0.5, c='k')
for i in np.arange(15):
    plt.text(i+0.1,eur2jpy(med[i])+15,str(eur2jpy(med[i])), fontsize=15)
    
plt.xticks(np.arange(15), \
           ['Automobile','Space & Aviation','Energy','Medical',\
           'Telecomunication','E-technique','Consulting','Logistic','Building',\
           'Insurance','Consumer goods','Hardware','Civil Service','Commerce','Health'], \
    fontsize=15, rotation=45, ha='right')

plt.xlabel('Field', fontsize=15)
plt.ylabel(r'Income / year  (x $10^{4}$ JPY)', fontsize=15)
plt.xlim([-0.5, 14.5])
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()

# new-graduates' income ====================================
v = np.array([[44869,48921],[45219,46398],[40667,45341],\
              [43600,44425],[41846,44132],[35429,35735]])
leg = ['Consulting, Engineering','Project- & Quality management',\
       'Software development','Database development','Admin & Helpdesk',\
       'Web development']
plt.subplot(121)
for i in np.arange(6):
    plt.plot(np.arange(2), eur2jpy(v[i]), label=leg[i], \
             linestyle='-', linewidth=6, alpha=0.5)

plt.xticks(np.arange(2), ['bachelor','master'], fontsize=15)
plt.xlabel('Education', fontsize=15)
plt.ylabel(r'Income / year  (x $10^{4}$ JPY)', fontsize=15)
plt.xlim([-0.5, 1.5])
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), fontsize=15, loc=2, borderaxespad=0.)
plt.show()

# IT jobs ==================================================
v = np.array([72400,70763,69850,65040,58000,54000,\
              51891,47848,43200,42260,35810])
leg = ['Project manager','Security expert','SAP consultant','IT consultant',\
       'SAP developer','Mobile developer','System- & net administrator','Administrator',\
       'Frontend developer','User support','Web development']

plt.figure()
plt.bar(np.arange(11), eur2jpy(v), alpha=0.5)
for i in np.arange(11):
    plt.text(i-0.2, eur2jpy(v[i]) + 10, str(eur2jpy(v[i])), fontsize=15)

plt.xticks(np.arange(11), leg, fontsize=15, rotation=45, ha='right')
plt.xlabel('IT job', fontsize=15)
plt.ylabel(r'Income / year  (x $10^{4}$ JPY)', fontsize=15)
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.show()


# income as a function of working years ====================
v = np.array([[47331,56678,65692],[46302,53296,63500],[46306,59289,63881],\
              [44409,52900,56405],[44409,47412,54532],[44733,46542,52700]])

leg = ['Consulting, Engineering','Project- & Quality management',\
       'Software development','Database development','Admin & Helpdesk',\
       'Web development']
plt.subplot(121)
for i in np.arange(6):
    plt.plot(np.arange(3), eur2jpy(v[i]), label=leg[i], \
             linestyle='-', linewidth=6, alpha=0.5)

plt.xticks(np.arange(3), ['till 2 yrs','3 - 5 yrs', '6 - 10 yrs'], fontsize=15)
plt.xlabel('Work Experience', fontsize=15)
plt.ylabel(r'Income / year  (x $10^{4}$ JPY)', fontsize=15)
plt.xlim([-0.5, 2.5])
plt.tick_params(labelsize=15)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), fontsize=15, loc=2, borderaxespad=0.)
plt.show()


