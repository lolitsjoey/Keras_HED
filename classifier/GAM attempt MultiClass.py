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
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
X, y = datasets.load_iris(return_X_y=True)
OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
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
    new_values, y = datasets.load_iris(return_X_y=True)
    if not load:
        from sklearn.linear_model import LogisticRegression
        X, y = datasets.load_iris(return_X_y=True)
        clf = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=1000).fit(X, y)
        aaa = clf.predict_proba(X)



        base_estimator = LogisticGAM(s(0) + s(1) + s(2) + s(3))
        ensemble = OneVsRestClassifier(base_estimator, n_jobs=1).fit(new_values, y).predict(new_values)
        ensemble.fit(new_values, y)
        ensemble.predict_proba(new_values)

    else:
        with open(load, 'rb') as f:
            rand_gam = pickle.load(f)

    rand_gam.summary()
    scores = rand_gam.predict_proba(new_values)
    rand_gam.accuracy(new_values, y)
    scr_order = np.argsort(scores)
    filename = name + '.pkl'

    with open('./GAM_model/models/' + name, 'wb') as f:
        pickle.dump(rand_gam, f)

    w = 480
    h = 480

    columns = 21
    rows = 1

    # ax enables access to manipulate each of subplots
    dirList =['./pl_train_station/output_hed_images/', './pl_train_station/pretty_good_set_input/']
    ordered_list = [[os.listdir('./pl_train_station/output_hed_images/')[i] for i in scr_order]][0]
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

score_notes_from_network('./intermediate_output.csv')