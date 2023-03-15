import pickle
import numpy as np
from os import path
import pandas as pd
import re

rdmpth='/foundcog/foundcog_results/pictures/rdms_pairwise'
toremove = ['IRN1','ICN2','IRC3','IRC9','IRC13']
with open(path.join(rdmpth, f'rdms_within-subjects.pickle'),'rb') as f:
    within_dat = pickle.load(f)
    isblank = np.isnan(within_dat['all_rdm_within_across_reps']['lh_evc'][0,0,:])
    for sub in toremove:
        idx = [i for i,x in enumerate(within_dat['sub_list_used']) if sub + '_' in x]
        isblank[idx[0]] = True
    within_dat['sub_list_used'] = [x for i,x in enumerate(within_dat['sub_list_used']) if not isblank[i]]
    for region in  within_dat['all_rdm_within_across_reps']:
        within_dat['all_rdm_within_across_reps'][region] =  within_dat['all_rdm_within_across_reps'][region][:,:,~ isblank]
    to_analyse = [1,2,4,5,6,7,8,9,10,11]
    for region in within_dat['all_rdm_within_across_reps']:
        within_dat['all_rdm_within_across_reps'][region] = within_dat['all_rdm_within_across_reps'][region][to_analyse, :,:][:,to_analyse,:]
    within_dat['conditions_split'] = [ within_dat['conditions_split'][x] for x in to_analyse]
    ind = np.argsort(within_dat['sub_list_used'])
    within_dat['sub_list_used'] = [within_dat['sub_list_used'][i] for i in ind]
    for region in  within_dat['all_rdm_within_across_reps']:
        within_dat['all_rdm_within_across_reps'][region] =  within_dat['all_rdm_within_across_reps'][region][:,:,ind]
    sub_list_tidy = [x.split('/')[-2].split('_')[0] for x in within_dat['sub_list_used']]


# Without crabs and squirrels
condnames_by_task = {
    'pictures': ['crab_rep1', 'crab_rep2', 'seabird_rep1', 'seabird_rep2',  'rubberduck_rep1', 'rubberduck_rep2', 'shoppingcart_rep1', 'shoppingcart_rep2', 'towel_rep1', 'towel_rep2', 'food_rep1', 'food_rep2', 'dishware_rep1', 'dishware_rep2', 'fence_rep1', 'fence_rep2',   'shelves_rep1', 'shelves_rep2',   'tree_rep1', 'tree_rep2']
    }

# Animate or not
anim = {'seabird':1, 'squirrel':1, 'crab':1, 'rubberduck':0, 'shoppingcart':0, 'towel':0, 'food':0, 'dishware':0,'fence':0, 'shelves':0,'tree':0}


for task in ['pictures']:
    ncond = len(condnames_by_task[task])

ncond_split = ncond//2 # make ncond_split = 10 because N_cond is 20 because 20 condnames by task (10 categories 2 reps)

conditions = condnames_by_task[task]
conditions_split = [re.sub(r'_rep[0-9]', '', x) for x in conditions[::2]] # get rid of rep1 etc

mvpacons = {}

id_con = np.eye(ncond_split)
id_con[id_con==1] /= -ncond_split    # ncond_split leading diagonal elements
id_con[id_con==0] = 1/(ncond_split*(ncond_split-1))
mvpacons['id'] = id_con

# Animacy rdm
animvalue=np.array([anim[cond.split('_')[0]] for cond in conditions_split]) # get list of animacy values for each of our items by looking up in dict

# Make animacy RDM
rdm_anim = 2*(animvalue[:,np.newaxis] == animvalue[np.newaxis,:]).astype(np.float64)-1    # just same (1) or different (0)
rdm_anim[np.eye(ncond_split)==1] = 0 # blank out leading diagonal
rdm_anim[rdm_anim==1] = -1/np.sum(rdm_anim==1)     # divide 1's by number of them
rdm_anim[rdm_anim==-1] = 1/np.sum(rdm_anim==-1)      # same for 0's
mvpacons['anim'] = rdm_anim

animcon_per_subj = np.sum(np.sum(rdm_anim[:,:,np.newaxis]*within_dat['all_rdm_within_across_reps']['both_vvc'],axis=1),axis=0)
# Now have one animacy contrast value for each subject - positive and bigger than 0 = more within than between

#Export animacy contrast values to a neat df and csv
dfA = pd.DataFrame(columns=['value','subject'])
dfA['value']=animcon_per_subj
dfA['subject']=sub_list_tidy
dfA.to_csv(r'AnimacyContrast.csv')

print(dfA)

import matplotlib.pyplot as plt
plt.hist(dfA['value'], bins=20)
plt.xlabel('Animacy Contrast Values')
plt.ylabel('Frequency')
plt.savefig('AnimacyContrastHist.png')

import statistics
mean = statistics.mean(dfA['value'])
SD = statistics.stdev(dfA['value'])

#Test for normality
import scipy
from scipy import stats
print(stats.shapiro(dfA['value']))
#NOT NORMAL - so need to do bootstrapping

#One sample T-test to assess whether contrast values greater than 0 - ie whether leading diagonal greater than off diagonal
print(stats.ttest_1samp(dfA['value'], 0))

# calculating Cohen's d
# d = (sample mean - 0)/ sample sd with n-1 dof
d = mean/SD
print(d)



#convert pandas df to array
array = dfA['value'].to_numpy()

#convert array to sequence
data = (array,)
print(np.mean(data))

#calculate 95% bootstrapped confidence interval for mean

from scipy.stats import bootstrap

bootstrap_ci = bootstrap(data, np.mean, confidence_level=0.95,
                         random_state=1, method='percentile')

#view 95% boostrapped confidence interval
print(bootstrap_ci.confidence_interval)
print(100)

