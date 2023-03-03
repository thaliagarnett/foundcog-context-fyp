import pickle
import numpy as np
from os import path

# Across subject
rdmpth='/foundcog/foundcog_results/pictures/rdms_pairwise'


# Within subject data
with open(path.join(rdmpth, f'rdms_within-subjects.pickle'),'rb') as f:
    within_dat = pickle.load(f)

    # Check one rdm per person
    print(within_dat['all_rdm_within_across_reps']['both_vvc'].shape)

# Summarise across subject pairwise RDMs
avg_rdm_this_sub={}
rois = ['both_evc', 'both_vvc'] # just bilateral for now
for roi in rois:
    with open(path.join(rdmpth, f'rdms_across-subjects_roi-{roi}.pickle'),'rb') as f:
        across_dat = pickle.load(f)

        print(across_dat.keys())
        nsub = len(across_dat['sub_list_used'])
        
        print(f'Number of subjects {nsub}')

        # Get the RDM for every pairwise comparison involving this subject
        avg_rdm_this_sub[roi]=[]
        for sub in range(nsub):
            pairs=across_dat['all_rdm'].keys()
            pairs_with_this_sub = [item for item in pairs if item[0]==sub or item[1]==sub]
            print(f'sub {sub} pairs with this sub {len(pairs_with_this_sub)}')

            all_rdm_this_sub = np.stack([across_dat['all_rdm'][x] for x in pairs_with_this_sub], axis=2)
            
            avg_rdm_this_sub[roi].append(np.nanmean(all_rdm_this_sub, axis=2))

    avg_rdm_this_sub[roi] = np.stack(avg_rdm_this_sub[roi], axis=2)
    print(avg_rdm_this_sub[roi].shape)