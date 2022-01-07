#%%
import os
import mne
import numpy as np
# Pipeline object:
from sklearn.pipeline import Pipeline
# Component analysis:
from PCA import PCA
# CrossVal tools:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
from sklearn.model_selection import cross_val_score
# Save modeles tools:
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
import pickle

DATA_PATH = 'files/'

SUBJECT_LIST = ['S001'] # 0.96
# SUBJECT_LIST = ['S015'] # 0.84
# SUBJECT_LIST = ['S042'] # 0.87


# Dataset:

# left vs right fist Real [3, 7, 11] :
# left vs right fist Imagine [4, 8, 12]
    # t0 = rest
    # t1 = left
    # t2 = right
# left and right fist vs feet Real [5, 9, 13]
# left and right fist vs feet Imagine [6, 10, 14]
    # t0 = rest
    # t1 = main
    # t2 = pied

# RUN = [3,7,11]
# RUN = [4,8,12]
# RUN = [5, 9, 13]
# RUN = [6,10,14]
RUN = [6,10,14,5,9,13] 
# RUN = [4,8,12,3,7,11]

#%%

def load_dataset(data_path, subjects_list=['S001'], runs=RUN):
    dataset_list = []
    sfreq = None
    for subject in subjects_list:
        try:
            file_path_list = [(data_path + subject + '/' + file) for i, file in enumerate(os.listdir(data_path + subject))]
            data_list = []
            for file in file_path_list:
                if file.endswith(".edf"):
                    run = int(file.split('R')[1].split(".")[0])
                    if run in runs:
                        tmp = mne.io.read_raw_edf(file, preload=True)
                        list_annot = tmp.annotations.description
                        for i in range(len(list_annot)):
                            if list_annot[i] == 'T0':
                                list_annot[i] = 'Rest'
                            if list_annot[i] == 'T1':
                                list_annot[i] = 'L'
                            if list_annot[i] == 'T2':
                                list_annot[i] = 'R'
                        annot = mne.Annotations(onset=tmp.annotations.onset, duration=tmp.annotations.duration, description=list_annot)
                        tmp.set_annotations(annot)
                        if sfreq == None:
                            sfreq = tmp.info['sfreq']
                        if tmp.info['sfreq'] == sfreq:
                            data_list.append(tmp)
            data = mne.concatenate_raws(data_list)
            dataset_list.append(data)
            print(f'Data load for subject: {subject}')
        except ValueError:
            print(f'{ValueError}: Fail to load data for subject: {subject}')
            return
    data = mne.concatenate_raws(dataset_list)
    
    # # Selected channels:
    # See - https://arxiv.org/pdf/1312.2877.pdf - p209
    channels = data.info["ch_names"]
    good_channels = ['Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 
                     'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 
		            'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.']
    bad_channels = [x for x in channels if x not in good_channels] 
    data.drop_channels(bad_channels)

    # Add filter:
    # A band pass filter from 7 Hz to 30 Hz was applied to remove the DC 
    # (direct current) shifts and to minimize the presence of 
    # filtering artifacts at epoch boundaries. A Notch filter was also 
    # Applied to remove the 60 Hz line noise
    # See - https://arxiv.org/pdf/1312.2877.pdf - p209
    data.notch_filter(60, method='iir')
    data.filter(7, 30, picks=mne.pick_types(data.info, eeg=True), method='iir')

    # Allowed montage value: 
    # 'EGI_256', 'GSN-HydroCel-128', 'GSN-HydroCel-129', 'GSN-HydroCel-256', 
    # 'GSN-HydroCel-257', 'GSN-HydroCel-32', 'GSN-HydroCel-64_1.0', 
    # 'GSN-HydroCel-65_1.0', 'biosemi128', 'biosemi16', 'biosemi160', 
    # 'biosemi256', 'biosemi32', 'biosemi64', 'easycap-M1', 'easycap-M10', 
    # 'mgh60', 'mgh70', 'standard_1005', 'standard_1020', 'standard_alphabetic', 
    # 'standard_postfixed', 'standard_prefixed', 'standard_primed', 
    # 'artinis-octamon', 'artinis-brite23'
    # See https://archive.physionet.org/pn4/eegmmidb/
    montage = mne.channels.make_standard_montage("standard_1020")
    mne.datasets.eegbci.standardize(data)
    data.set_montage(montage)

    # Dataframe creation
    all_events, all_event_id = mne.events_from_annotations(data, event_id=dict(L=0, R=1))
    picks = mne.pick_types(data.info, eeg=True, exclude='bads')
    epochs = mne.Epochs(data, all_events, all_event_id, -1., 4., proj=True, picks=picks, baseline=None, preload=True)   

    y = epochs.events[:, -1]
    X = epochs.get_data()

    # plot
    # montage.plot()
    mne.viz.plot_raw(data, scalings={"eeg": 75e-6}, block=True)
    times = np.arange(-0.5, 0.5, 0.1)
    epochs.average().plot_topomap(times, ch_type='eeg')
    data.plot_psd()
            
    print(f'Data load for subject: {subjects_list}')
    return(X, y)


# Classifier:
from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def create_model(X, y):
    pca = PCA(n_components=4)

    svc = SVC(gamma='auto')
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, 
        # hidden_layer_sizes=(5, 2), random_state=1)
    # lda = LinearDiscriminantAnalysis()

    pipe = Pipeline([('PCA', pca), ('SVC', svc)])
    # pipe = Pipeline([('PCA', pca), ('CLF', clf)])
    # pipe = Pipeline([('PCA', pca), ('LDA', lda)])

    model = pipe.fit(X, y)
    return model

#%%

if __name__ == "__main__":
    
    X, y = load_dataset(DATA_PATH, subjects_list=SUBJECT_LIST)

    
    model = create_model(X, y)

    scores = cross_val_score(model, X, y, cv=3)
    print('Score: ', scores.mean())
    
    # Save fit model
    pickle.dump(model, open('model.save', 'wb'))

#%%