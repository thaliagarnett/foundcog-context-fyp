import pickle
import numpy as np
from os import path
import pandas as pd

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

# Up to here just loading the fMRI data like in contrast.explained.py

# Load the context data as df3
df3 = pd.read_csv('ContextQs2.csv') 

# The list of objects as they are in the fMRI data
print(within_dat['conditions_split'])

# Create empty dictionary to store all the contrast values
df_contrast={}

# Create empty list to store all average context ratings for H4
AvgRatings = []

#Create empty list to stor all average contrast values for H4
AvgContrast = []

#Start a for loop so that everything below will be done for each object
for objind,objname in enumerate(within_dat['conditions_split']):
#question - why is there objind AND objname - why not just one

#Question - I did this line 

    print(objname)

    id_con=np.zeros((10,10)) #Makes a 10x10 matrix
    id_con[objind,:]=1/9 #set's all rows to -1/9
    id_con[objind,objind]=-1 #overrides leading diagonal with 1

    #Get one contrast value for each subject for the object (x10)
    objcon_per_subj = np.sum(np.sum(id_con[:,:,np.newaxis]*within_dat['all_rdm_within_across_reps']['both_vvc'],axis=1),axis=0)

    #Create a dataframes with contrast values for each participant for object (x10)
    df_contrast[objname] = pd.DataFrame(columns=['objvalue','subject'])
    df_contrast[objname]['objvalue']=objcon_per_subj #puts the contrast values for each subject in the "objvalue" column
    df_contrast[objname]['subject']=sub_list_tidy #puts the subject ID in the "subject" column
    df_contrast[objname].to_csv(f'contrast_{objname}.csv') #exports to csv ----- don't know if I actually need to do this because the dataframe is in this script anyway

    #Now start the analysis

    # Check for normality
    from scipy import stats
    print(stats.shapiro(df_contrast[objname]['objvalue']))
    print(stats.shapiro(df3[objname]))

    #Scatterplot showing correlation between contrast values and context ratings
    from matplotlib import pyplot as plt
    plt.figure(objind, figsize=(8,6))
    font = {'size' : 22}
    plt.rc('font', **font)
    plt.scatter(df3[objname], df_contrast[objname]['objvalue'])
    m, b = np.polyfit(df3[objname], df_contrast[objname]['objvalue'], 1)
    plt.xlabel('Visual Experience Rating')
    plt.xticks([1,2,3,4,5])
    plt.ylabel('Contrast Value')
    plt.plot(df3[objname], m*df3[objname]+b, color='grey')
    plt.savefig(f'ContextContrast_{objname}.png')

    #Spearman Rank Test
    res = stats.spearmanr(df3[objname], df_contrast[objname]['objvalue'])
    print(res)

    #H4
    AvgRatings.append(df3[objname].mean()) # Calculates the mean of ratings for each object and adds to list called AvgRatings
    AvgContrast.append(df_contrast[objname]['objvalue'].mean()) # Calculates the mean of contrast valyes for each object and adds to list called AvgContrast

#H3 Shopping Cluster
shopping_ratings = pd.concat((df3['shoppingcart'], df3['shelves']))
shopping_contrast = pd.concat((df_contrast['shoppingcart']['objvalue'], df_contrast['shelves']['objvalue']))

print('shopping')
print(stats.shapiro(shopping_ratings))
print(stats.shapiro(shopping_contrast))
plt.figure(11, figsize=(10,10))
font = {'size' : 22}
plt.rc('font', **font)
plt.scatter(shopping_ratings, shopping_contrast)
m, b = np.polyfit(shopping_ratings, shopping_contrast, 1)
plt.plot(shopping_ratings, m*shopping_ratings+b, color='grey')
plt.xlabel('Visual Experience Rating')
plt.ylabel('Contrast Value')
plt.savefig(f'ContextContrast_ShoppingCluster.png')

res2 = stats.spearmanr(shopping_ratings, shopping_contrast)
print(res2)

# H4 Print the average lists (NB they are in the same order as in conditions_split)
print('H4')
print(AvgRatings)
print(AvgContrast)

print(stats.shapiro(AvgRatings))
print(stats.shapiro(AvgContrast))

plt.figure(12,figsize=(8,6))
plt.scatter(AvgRatings, AvgContrast)
m, b = np.polyfit(AvgRatings, AvgContrast, 1)
plt.plot(AvgRatings, m*np.array(AvgRatings)+b, color='grey')
plt.xlabel('Mean Visual Experience Rating')
plt.ylabel('Mean Contrast Value')
plt.savefig('AvgRatingsAvgContrast.png')

res3 = stats.pearsonr(AvgRatings, AvgContrast)
print(res3)