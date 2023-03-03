import pickle
import numpy as np
from os import path
import pandas as pd

# Across subject - I think just labels the across subjects RDMs as rdmpth
rdmpth='/foundcog/foundcog_results/pictures/rdms_pairwise'

#Subjects with no rating data, remove from fMRI
toremove = ['IRN1','ICN2','IRC3','IRC9','IRC13']

# Within subject data - Not actually sure what this bit does
with open(path.join(rdmpth, f'rdms_within-subjects.pickle'),'rb') as f:
    within_dat = pickle.load(f)
 
    # Remove subjects with no fMRI data
    isblank = np.isnan(within_dat['all_rdm_within_across_reps']['lh_evc'][0,0,:])

    # Remove subjects without context ratings
    for sub in toremove:
        idx = [i for i,x in enumerate(within_dat['sub_list_used']) if sub + '_' in x]
        isblank[idx[0]] = True
    within_dat['sub_list_used'] = [x for i,x in enumerate(within_dat['sub_list_used']) if not isblank[i]]
    for region in  within_dat['all_rdm_within_across_reps']:
        within_dat['all_rdm_within_across_reps'][region] =  within_dat['all_rdm_within_across_reps'][region][:,:,~ isblank]

    print(within_dat['all_rdm_within_across_reps']['both_vvc'].shape)

    # Remove objects without context ratings from conditions_split and rdms
    print(within_dat['conditions_split'])
    to_analyse = [1,2,4,5,6,7,8,9,10,11] #omit 0 and 3 because they correspond to cat and squirrel
    for region in within_dat['all_rdm_within_across_reps']:
        within_dat['all_rdm_within_across_reps'][region] = within_dat['all_rdm_within_across_reps'][region][to_analyse, :,:][:,to_analyse,:]
    within_dat['conditions_split'] = [ within_dat['conditions_split'][x] for x in to_analyse]

    print(within_dat['all_rdm_within_across_reps']['both_vvc'].shape)

    # Sorting subjects into alphabetical order
    ind = np.argsort(within_dat['sub_list_used'])
    within_dat['sub_list_used'] = [within_dat['sub_list_used'][i] for i in ind]
    for region in  within_dat['all_rdm_within_across_reps']:
        within_dat['all_rdm_within_across_reps'][region] =  within_dat['all_rdm_within_across_reps'][region][:,:,ind]

    # Get short subject identifiers - ie instead of having long names for each subject LOCATION just have each subject ID
    # Now can refer to within_dat['sub_list_used'] as sub_list_tidy
    sub_list_tidy = [x.split('/')[-2].split('_')[0] for x in within_dat['sub_list_used']]
    print(sub_list_tidy)
    print(sub_list_tidy.index('sub-ICC133')) # this tell you where a subject is located in the matrix (ie 1 means 2nd in list)
    
#ABOVE - for all hypotheses, use to load data in correct format

    # within_dat['all_rdm_within_across_reps']['both_vvc'] = the massive 3D matrix
    # shape is 12 (average of 1st presentations of types), 12 (average of second presentations of types), 93 (subjects)
    # for bilateral = both hemispheres of vvc
    print(within_dat['all_rdm_within_across_reps']['both_vvc'].shape)

    # List of all the subject names we have within subject RDMs for:
    print(within_dat['sub_list_used'])
    # Printing the file name for a subject (eg) sub 3's RDM (as zero based):
    print(within_dat['sub_list_used'][2])
    # Print the RDM values for subject 3:
    print(within_dat['all_rdm_within_across_reps']['both_vvc'][:,:,2])
    # Note - closer to 0 is more similar
    
    # .shape[0] gets numbers of rows / shape[1] gets number of columns
    ncond = within_dat['all_rdm_within_across_reps']['both_vvc'].shape[0] # Label ncond as number of rows in the matrix - ie 12

    # Identity contrast - average of off diagonal values minus average of leading diagonal values
    id_con = np.eye(ncond)     #create a matrix called id_con - diagonals are 1 and off diagonals are 0, with 12 (ncond) rows
    id_con[id_con==1] /= -ncond    #multiple leading diagonal by -1/N / ncond leading diagonal elements ?
    id_con[id_con==0] = 1/(ncond*(ncond-1)) #multiply off diagonals by +1/(N*(N-1))

#Code below here exactly same for H3

        # BELOW - running the contrast in lots of steps
        # id_con[:,:,np.newaxis] is the funny thing we created above (special maths) but made into 3D matrix same size as big RDM
        # x is each value in this created matrix multiplied by each value in the actual 3D RDM
        # x=id_con[:,:,np.newaxis]*within_dat['all_rdm_within_across_reps']['both_vvc'] # Shape of x is (12,12,93)
        # below is adding up every row in x so left with one value for each type for each subject
        # np.sum(x,axis=1).shape # shape of this is (12,93)
        # below is adding up all the rows and then all the columns for each subject in one go - left with one value for each subject
        # np.sum(np.sum(x,axis=1),axis=0) # shape of this is (93)
        # label the 93 values as con_per_subj because it is the contrast per subject
        # con_per_subj = np.sum(np.sum(x,axis=1),axis=0)
  
    # BELOW - Running contrast in just one line
    con_per_subj = np.sum(np.sum(id_con[:,:,np.newaxis]*within_dat['all_rdm_within_across_reps']['both_vvc'],axis=1),axis=0)
    
    # One value per subject - 93 values
    print(con_per_subj.shape)

    # below creates a df which has 2 columns - "value" and "subject" - both currently undefined
    df = pd.DataFrame(columns=['value','subject'])
    # below - puts the contrast values for each subject in the "value" column
    df['value']=con_per_subj
    # below - puts the subject ID in the "subject" column
    df['subject']=sub_list_tidy
    # below - exports this new df to a csv
    df.to_csv(r'Contrast1.csv')

    # overall - this page has created one contrast value for each paricipant which shows whether the leading diagonal is greater than the off diagonal - ie whether responses to same type is more similar than responses to different types
    # these values have been put into one df and exported to a csv called "contrast.csv"

print(100)