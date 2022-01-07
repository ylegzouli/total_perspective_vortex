#%%
import pickle
import numpy as np
from train import *


# [6,10,14,5,9,13] 
# [4,8,12,3,7,11]

# %%

def predict(subject=['S042'], runs=RUN):
    try:
        model = pickle.load(open("model.save", "rb"))
        X, y = load_dataset(DATA_PATH, subjects_list=subject, runs=runs )
    except:
        print('Error while loading model or data')
        return
    scores = []
    for n in range(X.shape[0]):
        pred = model.predict(X[n:n + 1, :, :])
        print("prediction = ", pred, "truth = ", y[n:n + 1])
        scores.append(1 - np.abs(pred[0] - y[n:n + 1][0]))
    print("Mean accuracy = ", np.mean(scores))


#%%
if __name__ == "__main__":
    predict(subject=['S001'], runs=[6,10,14,5,9,13])
    # predict(subject=['S001'], runs=[4,8,12,3,7,11])

#%%