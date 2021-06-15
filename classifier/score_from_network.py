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
import decimal
import math
import warnings
from sklearn.metrics import confusion_matrix

def load_score_model(load_name):
    with open(load_name + '/' + load_name.split('/')[-1] + '_model.pkl', 'rb') as f:
        rand_gam = pickle.load(f)

    with open(load_name + '/' + load_name.split('/')[-1] + '_pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    return rand_gam, pca


def save_model(save_name, rand_gam, pca):
    if os.path.exists(save_name):
        suffix = str(uuid.uuid4())[0:5]
        print('Save Path already exists, appending ({})'.format(suffix))
        save_name = save_name + '_' + str(suffix)
        os.makedirs(save_name)
    else:
        os.makedirs(save_name)

    with open(save_name + '/' + save_name.split('/')[-1] + '_model.pkl', 'wb') as f:
        pickle.dump(rand_gam, f)

    with open(save_name + '/' + save_name.split('/')[-1] + '_pca.pkl', 'wb') as f:
        pickle.dump(pca, f)
    return save_name

def load_model(load_name):
    with open(load_name + '/' + load_name.split('/')[-1] + '_model.pkl', 'rb') as f:
        rand_gam = pickle.load(f)

    with open(load_name + '/' + load_name.split('/')[-1] + '_pca.pkl', 'rb') as f:
        pca = pickle.load(f)

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

def decimal_from_value(value):
    return decimal.Decimal(value)

def learn_transform_scores(scores, predictions, load_name, save_name, load):
    transformed_scores = []
    scores = np.array(scores, dtype=decimal.Decimal)
    if load:
        aaa = pd.read_csv(load_name + '/' + load_name.split('/')[-1] + '_extreme_scores.csv', converters={'bounds': decimal_from_value})
        slope = (100 - 0)/(aaa.values[1][0] - aaa.values[0][0])
        for idx, scr in enumerate(scores):
            if scr == 1.0:
                scr = decimal.Decimal(1) / (1 + np.exp(-decimal.Decimal(aaa.values[1][0])))
            elif scr == 0.:
                scr = decimal.Decimal(1) / (1 + np.exp(-decimal.Decimal(aaa.values[0][0])))
            trns_score = round(slope * (- decimal.Decimal(1 / scr - 1).ln() - aaa.values[0][0]),3)
            if predictions[idx] == 1:
                trns_score = round(trns_score + (100-trns_score) * decimal.Decimal(0.7), 3)
            transformed_scores.append(trns_score)
    else:
        lower_b = float(min(scores) / 1.2)
        upper_b = float(max(scores) + (1 - max(scores)) * 0.1)

        if lower_b == 0.:
            print('Log GAM or approximation almost definitly overfit')
            decimal.getcontext().prec = abs(int(math.floor(math.log10(upper_b)))) + 5
            lower_b = 1 - decimal.Decimal(upper_b)
            scores[scores <= lower_b] = lower_b
        else:
            decimal.getcontext().prec = abs(int(math.floor(math.log10(lower_b)))) + 5

        if upper_b == 1.0:
            print('Log GAM or approximation almost definitly overfit')
            upper_b = 1 - decimal.Decimal(lower_b)
            scores[scores >= upper_b] = upper_b
        try:
            slope = (100 - 0) / (- decimal.Decimal(1/upper_b - 1).ln() + decimal.Decimal(1/lower_b - 1).ln())
        except ZeroDivisionError:
            print('Not this error again....')
            print(upper_b)
            print(lower_b)
            pd.DataFrame(scores).to_csv('./GAM_model/scores.csv')
        for idx, scr in enumerate(scores):
            trns_score = round(slope * (- decimal.Decimal(1/scr - 1).ln() + decimal.Decimal(1/lower_b - 1).ln()),3)
            if predictions[idx] == 1:
                trns_score = round(trns_score + (100 - trns_score) * decimal.Decimal(0.7), 3)
            if predictions[idx] == 0:
                trns_score = round(trns_score * decimal.Decimal(0.6), 3)
            transformed_scores.append(trns_score)

        scoring_df = pd.DataFrame([-decimal.Decimal(1 / lower_b - 1).ln(), -decimal.Decimal(1 / upper_b - 1).ln()])
        scoring_df.columns = ['bounds']
        scoring_df.to_csv(save_name + '/' + save_name.split('/')[-1] + '_extreme_scores.csv', index=False)

    arguments = np.argsort(transformed_scores)
    ordered_scores = np.array(transformed_scores)[arguments]
    ordered_labels = predictions[arguments]

    return ordered_scores, ordered_labels, arguments

def create_rand_gam(number_of_searches, new_values, pred_y, y, pca_splines, pca_lam, pred_splines, pred_lam, pred_factor):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    lams = np.random.rand(number_of_searches, new_values.shape[1] + 1)  # random points on [0, 1], with shape (1000, 3)
    lams = lams * 8 - 4  # shift values to -4, 4
    lams = 10 ** lams  # transforms values to 1e-4, 1e4
    lams[:,-1] = [10 ** i for i in np.random.rand(number_of_searches) * 4]
    new_values = np.append(new_values, np.array(pred_y).reshape(-1, 1), axis=1)

    titles = []
    dtype_string = []
    for i in range(new_values.shape[1] - 1):
        titles.append(str(i))
        if i == 0:
            x = s(i, n_splines=pca_splines, lam=pca_lam)
        else:
            x = x + s(i, n_splines=pca_splines, lam=pca_lam)
        dtype_string.append('numerical')
    if pred_factor:
        x = x + pygam.terms.f(i + 1, lam=pred_lam)
        dtype_string.append('categorical')
    else:
        x = x + s(i + 1, n_splines=pred_splines, lam=pred_lam)
        dtype_string.append('numerical')

    rand_gam = LogisticGAM(x).gridsearch(new_values, y, lam=lams)
    return rand_gam, new_values, titles

def plot_variables(rand_gam, new_values, titles, save_name, load):
    try:
        fig, axs = plt.subplots(1, new_values.shape[1])
        titles.append('class_guess')
        for i, ax in enumerate(axs):
            XX = rand_gam.generate_X_grid(term=i)
            pdep, confi = rand_gam.partial_dependence(term=i, width=.95)
            ax.plot(XX[:, i], pdep)
            ax.plot(XX[:, i], confi, c='r', ls='--')
            ax.set_title(titles[i])
        if not load:
            plt.savefig(save_name + '/' + save_name.split('/')[-1] + '_plot.png')
            plt.close()
        else:
            plt.close()
    except ValueError:
        print('Cant Plot')

def print_stats(rand_gam, ordered_scores, ordered_labels, ordered_truth, new_values, y):

    genuine_scores = [ordered_scores[i] for i in range(len(ordered_scores)) if ordered_truth[i] == 1]
    cft_scores = [ordered_scores[i] for i in range(len(ordered_scores)) if ordered_truth[i] == 0]

    if genuine_scores:
        crit_genuine = min(genuine_scores)
        print('lowest_scoring_genuine: {}: {}'.format(np.where(ordered_scores == crit_genuine)[0][0],
                                                      min(genuine_scores)))
    if cft_scores:
        crit_cft = max(cft_scores)
        print('highest_scoring_counterfeit: {}: {}'.format(np.where(ordered_scores == crit_cft)[0][0],
                                                       max(cft_scores)))
    try:
        print('Model Accuracy: {}'.format(rand_gam.accuracy(new_values, y)))
    except AttributeError:
        num_correct = 0
        gen_cft_correct = 0
        for idx,lab in enumerate(ordered_labels):
            if lab == ordered_truth[idx]:
                num_correct += 1
            if ((ordered_truth[idx] < 4) & (lab < 4)) or ((ordered_truth[idx] >= 4) & (lab >= 4)):
                gen_cft_correct += 1
        print('Model Accuracy Segments: {}'.format(num_correct/len(ordered_labels)))
        print('Model Accuracy CftGen: {}'.format(gen_cft_correct / len(ordered_labels)))


def clean_directories():
    shutil.rmtree('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/rgb/')
    shutil.rmtree('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/edge/')
    shutil.rmtree('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/transgressional_cfts/')
    shutil.rmtree('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/transgressional_gens/')
    os.makedirs('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/rgb/')
    os.makedirs('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/edge/')
    os.makedirs('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/transgressional_cfts/')
    os.makedirs('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/transgressional_gens/')

def write_edge(dirList, img, imShape, truth, scores, i):
    res_img = cv2.resize(cv2.imread(dirList[0] + img), imShape)
    cv2.imwrite('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/edge/' + str(i) + '_' + str(truth[i]) + '_' + str(
        scores[i]) + '.jpg',
                res_img)

def write_rgb(dirList, img, truth, scores, i):
    img = img[0:-10] + '.jpg'
    res_img = cv2.imread(dirList[1] + img)
    try:
        cv2.imwrite('C:/Users/joeba/Documents/github/Keras_HED/GAM_model/scored/rgb/' + str(i) + '_' + str(truth[i]) + '_' + str(
            scores[i]) + '.jpg', res_img)
    except Exception:
        try:
            img = img[0:-4] + '.bmp'
            res_img = cv2.imread(dirList[1] + img)
            cv2.imwrite('./GAM_model/scored/rgb/' + str(i) + '_' + str(truth[i]) + '_' + str(
                scores[i]) + '.jpg', res_img)
        except Exception:
            img = img[0:-4] + '.png'
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


def multinomial_glm(pca, new_values, y, pred_y, save_name, load_name, num_classes, genuine_classes, load, rand_gam=False):
    print('WARNING: No multinomial GAM implementation in Python. Resorting to MultiNomial Regression')
    if not load:
        new_values = np.append(new_values, np.array(pred_y).reshape(-1, 1), axis=1)
        rand_gam = LogisticRegression(multi_class='multinomial', max_iter=10000).fit(new_values, y)

    scores_pre = rand_gam.predict_proba(new_values)

    if genuine_classes is None:
        genuine_classes = np.arange(0, num_classes / 2)

    # TODO MAYBE SOME CLEVER ENSEMBLE CHOICE HERE

    class_guess_gam = np.argmax(scores_pre, axis=1)
    #wrong_rows_log = [i for i in range(len(y)) if class_guess_gam[i] != y[i]]

    class_guess = np.array(pred_y)
    #wrong_rows_classifier = [i for i in range(len(y)) if class_guess[i] != y[i]]

    # ii = np.array(wrong_rows_classifier)
    # print(
    #     'Incorrect Rows Classifier: {}, the log has {}'.format(wrong_rows_classifier, scores_pre[ii, np.array(y)[ii]]))
    # print('Incorrect Rows Logistic: {}'.format(wrong_rows_log))

    scores = []
    for idx, clss in enumerate(class_guess):
        if clss in genuine_classes:
            scores.append(scores_pre[idx, clss])
        else:
            scores.append(1 - scores_pre[idx, clss])
    if not load:
        save_name = save_model(save_name, rand_gam, pca)

    ordered_scores, ordered_labels, arguments = learn_transform_scores(scores, class_guess, load_name, save_name, load)
    ordered_truth = np.array(y)[arguments]
    if not load:
        print_stats(rand_gam, ordered_scores, ordered_labels, ordered_truth, new_values, y)

    return ordered_scores, ordered_labels, arguments, np.array(y)[arguments]


def logistic_gam(pca, new_values, y, pred_y, number_of_searches, pca_splines, pca_lam,
                             pred_lam, pred_splines, pred_factor, save_name, load, load_name, rand_gam=False, confidence=True):
    if not load:
        rand_gam, new_values, titles = create_rand_gam(number_of_searches, new_values, pred_y, y,
                                                       pca_splines, pca_lam, pred_splines, pred_lam, pred_factor)

        rand_gam.summary()
        save_name = save_model(save_name, rand_gam, pca)

        # plot_variables(rand_gam, new_values, titles, load)
    else:
        new_values = np.append(new_values, np.array(pred_y).reshape(-1, 1), axis=1)

    scores = rand_gam.predict_proba(new_values)
    if confidence:
        predictions = np.array(rand_gam.predict(new_values), dtype=np.int)
    else:
        predictions = np.array(pred_y)
    ordered_scores, ordered_labels, arguments = learn_transform_scores(scores, predictions, load_name, save_name, load)

    titles = []
    for i in range(new_values.shape[1] - 1):
        titles.append(str(i))
    titles.append('class_guess')
    plot_variables(rand_gam, new_values, titles, save_name, load)
    ordered_truth = np.array(y)[arguments]
    if not load:
        print_stats(rand_gam, ordered_scores, ordered_labels, ordered_truth, new_values, y)

    return ordered_scores, ordered_labels, arguments, ordered_truth


def score_notes_from_network(csv_dir, num_class=False, number_of_searches=5000, pca_quality=0.99, pca_splines=20, pca_lam=0.4,
                             pred_lam=0.02, pred_splines=50, pred_factor=True,
                             save_name=False, pca=False, load=False, load_name=False, genuine_classes=None, confidence=True):

    dataset, y, pred_y = prep_dataset(csv_dir)
    if not num_class:
        num_classes = len(set(y))
    else:
        num_classes = num_class

    if load:
        num_classes = num_class
        if not load_name:
            load_name = save_name
        rand_gam, pca = load_model(load_name)
    else:
        pca = PCA(pca_quality)
        pca.fit(dataset)
        rand_gam=False
    if not save_name:
        name = str(uuid.uuid4()) + '.pkl'
        save_name = './GAM_model/models/' + name

    new_values = pca.transform(dataset)
    print('Reduced to {}-dimensions'.format(str(new_values.shape[1])))

    if num_classes != 2:
        ordered_scores, ordered_labels, arguments, ordered_truth = multinomial_glm(pca, new_values, y, pred_y,
                                                                                   save_name, load_name, num_classes,
                                                                                   genuine_classes, load, rand_gam)
    else:
        ordered_scores, ordered_labels, arguments, ordered_truth = logistic_gam(pca, new_values, y, pred_y,
                                                                                number_of_searches, pca_splines, pca_lam,
                                                                                pred_lam, pred_splines, pred_factor,
                                                                                save_name, load, load_name, rand_gam, confidence=confidence)

    return ordered_scores, ordered_labels, arguments, ordered_truth


def write_out_scores(scores, y, scr_order, truth, imShape = (480, 480), genuine_classes = None,  images_scored='./pl_train_station/output_hed_images/',
                     rgb_dir='./pl_train_station/pretty_good_set_input/'):
    dirList = [images_scored, rgb_dir]
    ordered_list = [[os.listdir(images_scored)[i] for i in scr_order]][0]

    clean_directories()
    #TODO change so actually uses genuine classes not just 0,1
    for i, img in enumerate(ordered_list):
        write_edge(dirList, img, imShape, truth, scores, i)

    if rgb_dir is not False:
        for i, img in enumerate(ordered_list):
            write_rgb(dirList, img, truth, scores, i)

    if genuine_classes is None:
        genuine_classes = [1]
        cft_classes = [0]
    else:
        classes = list(range(len(np.unique(truth))))
        cft_classes = [i for i in classes if i not in genuine_classes]


    genuine_scores = [scores[i] for i in range(len(scores)) if truth[i] in genuine_classes]
    cft_scores = [scores[i] for i in range(len(scores)) if truth[i] in cft_classes]
    if (genuine_scores and cft_scores) and (rgb_dir is not False):
        print('lowest_scoring_genuine: {}: {}'.format(np.where(scores == min(genuine_scores))[0][0], min(genuine_scores)))
        cftBoolMask = [True if ii == 0 else False for ii in truth]
        scoreBoolMask = (scores > min(genuine_scores))
        write_transgressionals(cftBoolMask, scoreBoolMask, 'cfts')

        print('highest_scoring_counterfeit: {}: {}'.format(np.where(scores == max(cft_scores))[0][0], max(cft_scores)))
        genBoolMask = [not i for i in cftBoolMask]
        scoreBoolMask = (scores < max(cft_scores))
        write_transgressionals(genBoolMask, scoreBoolMask, 'gens')

def write_out_scores_noimages(scores, y, scr_order, truth, array_scored, genuine_classes = None):
    clean_directories()

    # TODO change so actually uses genuine classes not just 0,1
    for i, img in enumerate(array_scored[scr_order,:,:,:]):
        old_value = img
        old_min = np.min(img)
        old_max = np.max(img)
        new_min = 0
        new_max = 256.
        new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        im_color = np.array(new_value, dtype = 'uint8')
        im_color = cv2.applyColorMap(im_color, cv2.COLORMAP_HOT)
        cv2.imwrite('./GAM_model/scored/edge/' + str(i) + '_' + str(truth[i]) + '_' + str(scores[i]) + '.jpg', im_color)



    if genuine_classes is None:
        genuine_classes = [1]
        cft_classes = [0]
    else:
        classes = list(range(len(np.unique(truth))))
        cft_classes = [i for i in classes if i not in genuine_classes]

    genuine_scores = [scores[i] for i in range(len(scores)) if truth[i] in genuine_classes]
    print('lowest_scoring_genuine: {}: {}'.format(np.where(scores == min(genuine_scores))[0][0], min(genuine_scores)))

    cft_scores = [scores[i] for i in range(len(scores)) if truth[i] in cft_classes]
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
