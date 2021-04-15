import pygam
from pygam.datasets import wage, toy_interaction
from pygam import LinearGAM, LogisticGAM, s, f
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


def score_notes_from_network(csv_dir, number_of_searches = 5000, pca_quality = 0.99, pca_splines = 20, pca_lam = 0.4, pred_lam = 0.02, pred_splines = 50,
                             pred_factor=True, save_name=False, pca=False, load=False):
    if isinstance(csv_dir, str):
        dataset = pd.read_csv(csv_dir)
    else:
        dataset = csv_dir

    y = list(dataset['truth'])
    del dataset['truth']
    pred_y = list(dataset['prediction'])
    del dataset['prediction']

    if not save_name:
        name = str(uuid.uuid4()) + '.pkl'
        save_path = './GAM_model/models/' + name

    if not pca:
        pca = PCA(pca_quality)
        pca.fit(dataset)

    new_values = pca.transform(dataset)
    print('Reduced to {}-dimensions'.format(str(new_values.shape[1])))
    num_classes = len(set(y))
    if num_classes != 2:
        if not load:
            print('WARNING: No multinomial GAM implementation in Python. Resorting to MultiNomial Regression as a compromise')
            rand_gam = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=10000).fit(new_values, y)
        else:
            with open(load, 'rb') as f:
                rand_gam = pickle.load(f)
        scores_pre = rand_gam.predict_proba(new_values)
        if genuine_classes is None:
            genuine_classes = np.arange(0, num_classes / 2)

        class_guess = np.argmax(scores_pre, axis=1)
        scores = []
        for idx, clss in enumerate(class_guess):
            if clss in genuine_classes:
                scores.append(scores_pre[idx, clss])
            else:
                scores.append(1 - scores_pre[idx, clss])

        with open('./GAM_model/models/' + save_name, 'wb') as f:
            pickle.dump(rand_gam, f)

        transformed_scores = []
        upper_b = max(scores) + (1 - max(scores)) * 0.1
        lower_b = float(min(scores) / 1.2)
        slope = (100 - 0) / (-np.log(1 / upper_b - 1) + np.log(1 / lower_b - 1))
        for scr in scores:
            transformed_scores.append(slope * (-np.log(1 / scr - 1) + np.log(1 / lower_b - 1)))

        return transformed_scores

    if not load:
        lams = np.random.rand(number_of_searches, new_values.shape[1]+1)  # random points on [0, 1], with shape (1000, 3)
        lams = lams * 8 - 4  # shift values to -4, 4
        lams = 10 ** lams  # transforms values to 1e-4, 1e4
        new_values = np.append(new_values, np.array(pred_y).reshape(-1,1), axis = 1)

        titles = []
        for i in range(new_values.shape[1] - 1):
            titles.append(str(i))
            if i ==0:
                x = s(i, n_splines=pca_splines, lam = pca_lam)
            else:
                x = x + s(i, n_splines=pca_splines, lam = pca_lam)
        if pred_factor:
            x = x + pygam.terms.f(i+1, lam = pred_lam)
        else:
            x = x + s(i + 1, n_splines=pred_splines, lam=pred_lam)

        rand_gam = LogisticGAM(x).gridsearch(new_values, y, lam=lams)
    else:
        with open(load, 'rb') as f:
            rand_gam = pickle.load(f)

    rand_gam.summary()
    fig, axs = plt.subplots(1, new_values.shape[1])
    titles.append('class_guess')
    for i, ax in enumerate(axs):
        XX = rand_gam.generate_X_grid(term=i)
        pdep, confi = rand_gam.partial_dependence(term=i, width=.95)
        ax.plot(XX[:, i], pdep)
        ax.plot(XX[:, i], confi, c='r', ls='--')
        ax.set_title(titles[i])
    plt.show()

    scores = rand_gam.predict_proba(new_values)
    predictions = np.array(rand_gam.predict(new_values), dtype = np.int)
    print('Model Accuracy: {}'.format(rand_gam.accuracy(new_values,y)))

    transformed_scores = []
    upper_b = max(scores) + (1-max(scores))*0.1
    lower_b = float(min(scores)/1.2)
    slope = (100 - 0) / (-np.log(1 / upper_b - 1) + np.log(1 / lower_b - 1))
    for scr in scores:
        transformed_scores.append(slope * (-np.log(1 / scr - 1) + np.log(1 / lower_b - 1)))

    arguments = np.argsort(transformed_scores)
    ordered_scores = np.array(transformed_scores)[arguments]
    ordered_labels = predictions[arguments]

    genuine_scores = [ordered_scores[i] for i in range(len(ordered_scores)) if ordered_labels[i] == 1]
    cft_scores = [ordered_scores[i] for i in range(len(ordered_scores)) if ordered_labels[i] == 0]

    crit_genuine = min(genuine_scores)
    crit_cft = max(cft_scores)

    buffer = (crit_genuine - crit_cft)*1.1
    if buffer < 0.2:
        pass

    print('lowest_scoring_genuine: {}: {}'.format(np.where(ordered_scores == crit_genuine)[0][0],
                                                  min(genuine_scores)))

    print('highest_scoring_counterfeit: {}: {}'.format(np.where(ordered_scores == crit_cft)[0][0],
                                                       max(cft_scores)))
    with open(save_name, 'wb') as f:
        pickle.dump(rand_gam, f)
    return ordered_scores, ordered_labels, arguments, np.array(y)[arguments]


def write_out_scores(scores, y, scr_order, truth, images_scored='./pl_train_station/output_hed_images/',
                     rgb_dir='./pl_train_station/pretty_good_set_input/'):
    w = 480
    h = 480

    dirList = [images_scored, rgb_dir]
    ordered_list = [[os.listdir(images_scored)[i] for i in scr_order]][0]

    shutil.rmtree('./GAM_model/scored/rgb/')
    shutil.rmtree('./GAM_model/scored/edge/')
    shutil.rmtree('./GAM_model/scored/transgressional_cfts/')
    shutil.rmtree('./GAM_model/scored/transgressional_gens/')
    os.makedirs('./GAM_model/scored/rgb/')
    os.makedirs('./GAM_model/scored/edge/')
    os.makedirs('./GAM_model/scored/transgressional_cfts/')
    os.makedirs('./GAM_model/scored/transgressional_gens/')

    for i, img in enumerate(ordered_list):
        res_img = cv2.resize(cv2.imread(dirList[0] + img), (w, h))
        cv2.imwrite('./GAM_model/scored/edge/' + str(i) + '_' + str(truth[i]) + '_' + str(
            scores[i]) + '.jpg',
                    res_img)


    for i, img in enumerate(ordered_list):
        img = img[0:-10] + '.jpg'
        res_img = cv2.imread(dirList[1] + img)
        cv2.imwrite('./GAM_model/scored/rgb/' + str(i) + '_' + str(truth[i]) + '_' + str(
            scores[i]) + '.jpg', res_img)

    genuine_scores = [scores[i] for i in range(len(scores)) if truth[i] == 1]
    print('lowest_scoring_genuine: {}: {}'.format(np.where(scores == min(genuine_scores))[0][0], min(genuine_scores)))

    cft_scores = [scores[i] for i in range(len(scores)) if truth[i] == 0]
    print('highest_scoring_counterfeit: {}: {}'.format(np.where(scores == max(cft_scores))[0][0], max(cft_scores)))

    cftBoolMask = [True if ii == 0 else False for ii in truth]
    scoreBoolMask = (scores > min(genuine_scores))
    transgressional = [x and y for x, y in zip(cftBoolMask, scoreBoolMask)]
    transgress_indexes = [i for i in range(len(transgressional)) if transgressional[i] == True]
    print('number of transgressional counterfeits: {}'.format(len(transgress_indexes)))

    trans_dir = './GAM_model/scored/transgressional_cfts/'
    for i in transgress_indexes:
        for img_name in os.listdir('./GAM_model/scored/rgb/'):
            if img_name.startswith(str(i)):
                cv2.imwrite(trans_dir + img_name,
                            cv2.imread('./GAM_model/scored/rgb/' + img_name))
                cv2.imwrite(trans_dir + img_name[0:-4] + 'edges.bmp',
                            cv2.imread('./GAM_model/scored/edge/' + img_name))

    genBoolMask = [not i for i in cftBoolMask]
    scoreBoolMask = (scores < max(cft_scores))
    transgressional = [x and y for x, y in zip(genBoolMask, scoreBoolMask)]
    transgress_indexes = [i for i in range(len(transgressional)) if transgressional[i] == True]
    print('number of transgressional genuines: {}'.format(len(transgress_indexes)))

    trans_dir = './GAM_model/scored/transgressional_gens/'
    for i in transgress_indexes:
        for img_name in os.listdir('./GAM_model/scored/rgb/'):
            if img_name.startswith(str(i)):
                cv2.imwrite(trans_dir + img_name,
                            cv2.imread('./GAM_model/scored/rgb/' + img_name))
                cv2.imwrite(trans_dir + img_name[0:-4] + 'edges.bmp',
                            cv2.imread('./GAM_model/scored/edge/' + img_name))


if __name__ == '__main__':
    ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network('./intermediate_output.csv', number_of_searches=1000)
    write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth, 'D:/FRNLib/federal seals/edges/', 'D:/FRNLib/federal seals/all/')

    x1 = range(len(ordered_scores[ordered_labels == 1]))
    x2 = range(len(ordered_scores[ordered_labels == 0]))
    plt.scatter(x1, ordered_scores[ordered_labels == 1], color='blue', alpha=0.2)
    plt.scatter(x2, ordered_scores[ordered_labels == 0], color='yellow', alpha=0.2)
    plt.show()
