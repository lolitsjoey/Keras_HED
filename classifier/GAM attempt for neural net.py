import pygam
from pygam.datasets import wage
from pygam import LinearGAM,LogisticGAM, s, f
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
import cv2
import shutil
import pickle
import uuid
from pygam.datasets import mcycle
from sklearn.linear_model import LogisticRegression


def score_notes_from_network(csv_dir, name = False, pca = False, load = False):
    dataset = pd.read_csv(csv_dir)
    y = dataset['truth']
    del dataset['truth']

    if not name:
        name = str(uuid.uuid4())
    if not pca:
        pca = PCA(0.95)
        pca.fit(dataset)

    new_values = pca.transform(dataset)
    num_classes = len(set(y))
    if num_classes != 2:
        if not load:
            print('WARNING: No multinomial GAM implementation in Python. Resorting to MultiNomial Regression as a compromise')
            rand_gam = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=10000).fit(new_values, y)
        else:
            with open(load, 'rb') as f:
                rand_gam = pickle.load(f)
        scores_pre = rand_gam.predict_proba(new_values)
        genuine_classes = np.arange(0,num_classes/2)
        class_guess = np.argmax(scores_pre,axis = 1)
        scores = []
        for idx, cls in enumerate(class_guess):
            if cls in genuine_classes:
                scores.append(scores_pre[idx,cls])
            else:
                scores.append(1 - scores_pre[idx,cls])
        filename = name + '.pkl'
        with open('./GAM_model/models/' + filename, 'wb') as f:
            pickle.dump(rand_gam, f)
        return scores

    if not load:
        lams = np.random.rand(200, new_values.shape[1]) # random points on [0, 1], with shape (100, 3)
        lams = lams * 6 - 3 # shift values to -3, 3
        lams = 10 ** lams # transforms values to 1e-3, 1e3

        rand_gam = LogisticGAM().gridsearch(new_values, y, lam=lams)
    else:
        with open(load, 'rb') as f:
            rand_gam = pickle.load(f)

    rand_gam.summary()
    scores = rand_gam.predict_proba(new_values)
    filename = name + '.pkl'
    with open('./GAM_model/models/' + filename, 'wb') as f:
        pickle.dump(rand_gam, f)
    return scores

def write_out_scores(scores, images_scored = './pl_train_station/output_hed_images/', rgb_dir = './pl_train_station/pretty_good_set_input/'):
    scr_order = np.argsort(scores)
    w = 480
    h = 480

    dirList = [images_scored, rgb_dir]
    ordered_list = [[os.listdir(images_scored)[i] for i in scr_order]][0]
    ordered_labels = [[y[i] for i in scr_order]][0]

    shutil.rmtree('./GAM_model/scored/rgb/')
    shutil.rmtree('./GAM_model/scored/edge/')
    os.makedirs('./GAM_model/scored/rgb/')
    os.makedirs('./GAM_model/scored/edge/')

    for i, img in enumerate(ordered_list):
        res_img = cv2.resize(cv2.imread(dirList[0] + img), (w, h))
        cv2.imwrite('./GAM_model/scored/edge/' + str(i) + '_' + str(ordered_labels[i]) + '_' + str(scores[scr_order[i]]) + '.bmp',
                    res_img)

    for i, img in enumerate(ordered_list):
        img = img[0:-10] + '.bmp'
        res_img = cv2.imread(dirList[1] + img)
        cv2.imwrite('./GAM_model/scored/rgb/' + str(i) + '_' + str(ordered_labels[i]) + '_' + str(scores[scr_order[i]]) + '.bmp', res_img)

#scores = score_notes_from_network('./intermediate_output.csv')
#print(scores)
scores = score_notes_from_network('./multiclasshyper.csv')
dataset = pd.read_csv('./multiclasshyper.csv')
y = dataset['truth']
print(list(zip(scores,y)))