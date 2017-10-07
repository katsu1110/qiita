# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:19:52 2017

@author: katsuhisa
"""

# import libraries
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# path to save figs
figpath = r'C:\\python_dir\\qiita\\optimalStop\\'

# function to find a partner based on a given decision-time
def getmeplease(rest, dt, fig):
    ## INPUT: rest ... integer representing the number of the fixed future encounters
    #         dt ... integer representing the decision time to set the threshold
    #         fig ... 1 if you want to plot the data
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    # score ranging from 0 to 100
    scoremax = 100
    #random.seed(1220)
    
    # sequence of scores 
    distribution = random.sample(range(scoremax), k=rest)
    
    # visualize distribution
    if fig==1:
        # visualize distribution of score
        plt.close('all')
        plt.bar(range(1,rest+1), distribution, width=1)
        plt.xlabel('sequence')
        plt.ylabel('score')
        plt.xlim((0.5,rest+0.5))
        plt.ylim((0,scoremax))
        plt.xticks(range(1, rest+1))

    # remember the highest score among the 100 x dt %
    if dt > 0:
        threshold = max(distribution[:dt])
    else:
        threshold = 0
    
    # pick up the first one whose score exceeds the threshold
    partner_id = 1
    partner_score = distribution[0]
    t = dt
    for t in range(dt+1, rest):
        if partner_score < threshold:
            partner_score = distribution[t]
        else:
            partner_id = t
            break
    else:
        partner_score = distribution[-1]
        partner_id = rest
        
    # get the actual ranking of the partner
    array = np.array(distribution)
    temp = array.argsort()
    ranks = np.empty(len(array), int)
    ranks[temp] = np.arange(len(array))
    partner_rank = rest - ranks[partner_id-1]
            
    # visualize all
    if fig==1:
        plt.plot([decisiontime+0.5,decisiontime+0.5],[0,scoremax],'--k')
        plt.bar(partner_id,partner_score, color='g', width=1)   
    
    return [partner_id, partner_score, partner_rank]

# basic parameters =======================================================
rest = 10
decisiontime = int(round(rest/np.exp(1)))
partner = getmeplease(rest, decisiontime, 0)

# simulation =============================================================
# parameters
repeat = 10000
rest = [5,10,20,100]
opt_dt = [int(round(x/np.exp(1))) for x in rest]
tooearly_dt = [int(round(x*0.1)) for x in rest]
toolate_dt = [int(round(x*0.8)) for x in rest]
half_dt = [int(round(x*0.5)) for x in rest]

# initialization
opt_rank = np.zeros(repeat*len(rest))
early_rank = np.zeros(repeat*len(rest))
late_rank = np.zeros(repeat*len(rest))
half_rank = np.zeros(repeat*len(rest))
opt_score = np.zeros(repeat*len(rest))
early_score = np.zeros(repeat*len(rest))
late_score = np.zeros(repeat*len(rest))
half_score = np.zeros(repeat*len(rest))

# loop to find the actual rank and score of a partner
k = 0
for r in range(len(rest)):
    for i in range(repeat):    
        # optimal model
        partner_opt = getmeplease(rest[r], opt_dt[r], 0)
        opt_score[k] = partner_opt[1]
        opt_rank[k] = partner_opt[2]
        
        # too-early model
        partner_early = getmeplease(rest[r], tooearly_dt[r], 0)
        early_score[k] = partner_early[1]
        early_rank[k] = partner_early[2]
        
        # too-late model
        partner_late = getmeplease(rest[r], toolate_dt[r], 0)
        late_score[k] = partner_late[1]
        late_rank[k] = partner_late[2]
        
        # half-time model
        partner_half = getmeplease(rest[r], half_dt[r], 0)
        half_score[k] = partner_half[1]
        half_rank[k] = partner_half[2]
        
        k += 1
        
# visualize distributions of ranks of chosen partners
plt.close('all')
begin = 0
for i in range(len(rest)):
    plt.figure(i+1)
    
    plt.subplot(2,2,1)
    plt.hist(opt_rank[begin:begin+repeat],color='blue')
    med = np.median(opt_rank[begin:begin+repeat])
    plt.plot([med, med],[0, repeat*0.8],'-r')
    plt.title('optimal: %i' %int(med))
    
    plt.subplot(2,2,2)
    plt.hist(early_rank[begin:begin+repeat],color='blue')
    med = np.median(early_rank[begin:begin+repeat])
    plt.plot([med, med],[0, repeat*0.8],'-r')
    plt.title('early: %i' %int(med))
    
    plt.subplot(2,2,3)
    plt.hist(late_rank[begin:begin+repeat],color='blue')
    med = np.median(late_rank[begin:begin+repeat])
    plt.plot([med, med],[0, repeat*0.8],'-r')
    plt.title('late: %i' %int(med))
    
    plt.subplot(2,2,4)
    plt.hist(half_rank[begin:begin+repeat],color='blue')
    med = np.median(half_rank[begin:begin+repeat])
    plt.plot([med, med],[0, repeat*0.8],'-r')
    plt.title('half: %i' %int(med))
        
    fig = plt.gcf()
    fig.canvas.set_window_title('rest' + ' ' + str(rest[i]))
    
    begin += repeat
    
    plt.savefig(figpath + 'rank_' + str(rest[i]))

begin = 0
for i in range(len(rest)):
    plt.figure(i+10)
    
    plt.subplot(2,2,1)
    plt.hist(opt_score[begin:begin+repeat],color='green')
    med = np.median(opt_score[begin:begin+repeat])
    plt.plot([med, med],[0, repeat*0.8],'-r')
    plt.title('optimal: %i' %int(med))
    
    plt.subplot(2,2,2)
    plt.hist(early_score[begin:begin+repeat],color='green')
    med = np.median(early_score[begin:begin+repeat])
    plt.plot([med, med],[0, repeat*0.8],'-r')
    plt.title('early: %i' %int(med))
    
    plt.subplot(2,2,3)
    plt.hist(late_score[begin:begin+repeat],color='green')
    med = np.median(late_score[begin:begin+repeat])
    plt.plot([med, med],[0, repeat*0.8],'-r')
    plt.title('late: %i' %int(med))
    
    plt.subplot(2,2,4)
    plt.hist(half_score[begin:begin+repeat],color='green')
    med = np.median(half_score[begin:begin+repeat])
    plt.plot([med, med],[0, repeat*0.8],'-r')
    plt.title('half: %i' %int(med))
        
    fig = plt.gcf()
    fig.canvas.set_window_title('rest' + ' ' + str(rest[i]))

    begin += repeat
    
    plt.savefig(figpath + 'score_' + str(rest[i]))