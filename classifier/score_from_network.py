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

def save_model(save_name, rand_gam, pca):
    if os.path.exists(save_name):
        print('Save Path already existes, appending (new)')
        os.makedirs(save_name + '_new')
    else:
        os.makedirs(save_name)

    with open(save_name + '/' + save_name.split('/')[-1] + '_model', 'wb') as f:
        pickle.dump(rand_gam, f)

    with open(save_name + '/' + save_name.split('/')[-1] + '_pca', 'wb') as f:
        pickle.dump(pca, f)

def load_model(save_name):
    with open(save_name + '/' + save_name.split('/')[-1] + '_model', 'r') as f:
        rand_gam = pickle.load(rand_gam, f)

    with open(save_name + '/' + save_name.split('/')[-1] + '_pca', 'r') as f:
        pca = pickle.load(pca, f)

    return rand_gam, pca

def prep_dataset(csv_dir):
    if isinstance(csv_dir, str):
        dataset = pd.read_csv(csv_dir)
    else:
        dataset = csv_dir
    y = list(dataset['truth'])
    del dataset['truth']
    pred_y = list(dataset['prediction'])
    del dataset['prediction']
    return dataset, y, pred_y

def transform_scores(scores, predictions):
    transformed_scores = []
    upper_b = max(scores) + (1 - max(scores)) * 0.1
    lower_b = float(min(scores) / 1.2)
    slope = (100 - 0) / (-np.log(1 / upper_b - 1) + np.log(1 / lower_b - 1))
    for scr in scores:
        transformed_scores.append(slope * (-np.log(1 / scr - 1) + np.log(1 / lower_b - 1)))

    arguments = np.argsort(transformed_scores)
    ordered_scores = np.array(transformed_scores)[arguments]
    ordered_labels = predictions[arguments]

    return ordered_scores, ordered_labels, arguments

def create_rand_gam(number_of_searches, new_values, pred_y, y, pca_splines, pca_lam, pred_splines, pred_lam, pred_factor):
    lams = np.random.rand(number_of_searches, new_values.shape[1] + 1)  # random points on [0, 1], with shape (1000, 3)
    lams = lams * 8 - 4  # shift values to -4, 4
    lams = 10 ** lams  # transforms values to 1e-4, 1e4
    new_values = np.append(new_values, np.array(pred_y).reshape(-1, 1), axis=1)

    titles = []
    for i in range(new_values.shape[1] - 1):
        titles.append(str(i))
        if i == 0:
            x = s(i, n_splines=pca_splines, lam=pca_lam)
        else:
            x = x + s(i, n_splines=pca_splines, lam=pca_lam)
    if pred_factor:
        x = x + pygam.terms.f(i + 1, lam=pred_lam)
    else:
        x = x + s(i + 1, n_splines=pred_splines, lam=pred_lam)

    rand_gam = LogisticGAM(x).gridsearch(new_values, y, lam=lams)
    return rand_gam, new_values, titles

def plot_variables(rand_gam, new_values, titles):
    fig, axs = plt.subplots(1, new_values.shape[1])
    titles.append('class_guess')
    for i, ax in enumerate(axs):
        XX = rand_gam.generate_X_grid(term=i)
        pdep, confi = rand_gam.partial_dependence(term=i, width=.95)
        ax.plot(XX[:, i], pdep)
        ax.plot(XX[:, i], confi, c='r', ls='--')
        ax.set_title(titles[i])
    plt.show()

def print_stats(rand_gam, ordered_scores, ordered_labels, new_values, y):
    genuine_scores = [ordered_scores[i] for i in range(len(ordered_scores)) if ordered_labels[i] == 1]
    cft_scores = [ordered_scores[i] for i in range(len(ordered_scores)) if ordered_labels[i] == 0]

    crit_genuine = min(genuine_scores)
    crit_cft = max(cft_scores)

    print('lowest_scoring_genuine: {}: {}'.format(np.where(ordered_scores == crit_genuine)[0][0],
                                                  min(genuine_scores)))

    print('highest_scoring_counterfeit: {}: {}'.format(np.where(ordered_scores == crit_cft)[0][0],
                                                       max(cft_scores)))
    print('Model Accuracy: {}'.format(rand_gam.accuracy(new_values, y)))

def clean_directories():
    shutil.rmtree('./GAM_model/scored/rgb/')
    shutil.rmtree('./GAM_model/scored/edge/')
    shutil.rmtree('./GAM_model/scored/transgressional_cfts/')
    shutil.rmtree('./GAM_model/scored/transgressional_gens/')
    os.makedirs('./GAM_model/scored/rgb/')
    os.makedirs('./GAM_model/scored/edge/')
    os.makedirs('./GAM_model/scored/transgressional_cfts/')
    os.makedirs('./GAM_model/scored/transgressional_gens/')

def write_edge(dirList, img, imShape, truth, scores, i):
    res_img = cv2.resize(cv2.imread(dirList[0] + img), imShape)
    cv2.imwrite('./GAM_model/scored/edge/' + str(i) + '_' + str(truth[i]) + '_' + str(
        scores[i]) + '.jpg',
                res_img)

def write_rgb(dirList, img, truth, scores, i):
    img = img[0:-10] + '.jpg'
    res_img = cv2.imread(dirList[1] + img)
    cv2.imwrite('./GAM_model/scored/rgb/' + str(i) + '_' + str(truth[i]) + '_' + str(
        scores[i]) + '.jpg', res_img)

def write_transgressionals(class_mask, score_mask, class_string):
    transgressional = [x and y for x, y in zip(class_mask, score_mask)]
    transgress_indexes = [i for i in range(len(transgressional)) if transgressional[i] == True]
    print('number of transgressional {}: {}'.format(class_string, len(transgress_indexes)))

    trans_dir = './GAM_model/scored/transgressional_{}/'.format(class_mask)
    for i in transgress_indexes:
        for img_name in os.listdir('./GAM_model/scored/rgb/'):
            if img_name.startswith(str(i)):
                cv2.imwrite(trans_dir + img_name,
                            cv2.imread('./GAM_model/scored/rgb/' + img_name))
                cv2.imwrite(trans_dir + img_name[0:-4] + 'edges.bmp',
                            cv2.imread('./GAM_model/scored/edge/' + img_name))



def score_notes_from_network(csv_dir, number_of_searches = 5000, pca_quality = 0.99, pca_splines = 20, pca_lam = 0.4, pred_lam = 0.02, pred_splines = 50,
                             pred_factor=True, save_name=False, pca=False, load=False, genuine_classes = None):

    dataset, y, pred_y = prep_dataset(csv_dir)
    num_classes = len(set(y))

    if not save_name:
        name = str(uuid.uuid4()) + '.pkl'
        save_name = './GAM_model/models/' + name

    if not load:
        pca = PCA(pca_quality)
        pca.fit(dataset)
    else:
        rand_gam, pca = load_model(save_name)

    new_values = pca.transform(dataset)
    print('Reduced to {}-dimensions'.format(str(new_values.shape[1])))

    if num_classes != 2:
        print('WARNING: No multinomial GAM implementation in Python. Resorting to MultiNomial Regression')
        if not load:
            rand_gam = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=10000).fit(new_values, y)
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

        save_model(save_name, rand_gam, pca)

        ordered_scores, ordered_labels, arguments = transform_scores(scores, class_guess)

        return ordered_scores, ordered_labels, arguments, np.array(y)[arguments]

    if not load:
        rand_gam, new_values, titles = create_rand_gam(number_of_searches, new_values, pred_y, y,
                        pca_splines, pca_lam, pred_splines, pred_lam, pred_factor)

    rand_gam.summary()
    save_model(save_name, rand_gam, pca)

    plot_variables(rand_gam, new_values, titles)

    scores = rand_gam.predict_proba(new_values)
    predictions = np.array(rand_gam.predict(new_values), dtype=np.int)
    ordered_scores, ordered_labels, arguments = transform_scores(scores, predictions)

    print_stats(rand_gam, ordered_scores, ordered_labels, new_values, y)

    return ordered_scores, ordered_labels, arguments, np.array(y)[arguments]

def write_out_scores(scores, y, scr_order, truth, imShape = (480, 480),  images_scored='./pl_train_station/output_hed_images/',
                     rgb_dir='./pl_train_station/pretty_good_set_input/'):
    dirList = [images_scored, rgb_dir]
    ordered_list = [[os.listdir(images_scored)[i] for i in scr_order]][0]

    clean_directories()
    #TODO change so actually uses genuine classes not just 0,1
    for i, img in enumerate(ordered_list):
        write_edge(dirList, img, imShape, truth, scores, i)

    for i, img in enumerate(ordered_list):
        write_rgb(dirList, img, truth, scores, i)

    genuine_scores = [scores[i] for i in range(len(scores)) if truth[i] == 1]
    print('lowest_scoring_genuine: {}: {}'.format(np.where(scores == min(genuine_scores))[0][0], min(genuine_scores)))

    cft_scores = [scores[i] for i in range(len(scores)) if truth[i] == 0]
    print('highest_scoring_counterfeit: {}: {}'.format(np.where(scores == max(cft_scores))[0][0], max(cft_scores)))

    cftBoolMask = [True if ii == 0 else False for ii in truth]
    scoreBoolMask = (scores > min(genuine_scores))
    write_transgressionals(cftBoolMask, scoreBoolMask, 'cfts')

    genBoolMask = [not i for i in cftBoolMask]
    scoreBoolMask = (scores < max(cft_scores))
    write_transgressionals(genBoolMask, scoreBoolMask, 'gens')



if __name__ == '__main__':
    ordered_scores, ordered_labels, arguments, ordered_truth = score_notes_from_network('./intermediate_output.csv', number_of_searches=1000)
    write_out_scores(ordered_scores, ordered_labels, arguments, ordered_truth, 'D:/FRNLib/federal seals/edges/', 'D:/FRNLib/federal seals/all/')

    x1 = range(len(ordered_scores[ordered_labels == 1]))
    x2 = range(len(ordered_scores[ordered_labels == 0]))
    plt.scatter(x1, ordered_scores[ordered_labels == 1], color='blue', alpha=0.2)
    plt.scatter(x2, ordered_scores[ordered_labels == 0], color='yellow', alpha=0.2)
    plt.show()
