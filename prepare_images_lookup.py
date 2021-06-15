import numpy as np
import cv2
from src.networks.hed_reduced import hed
import os
from skimage.feature import local_binary_pattern
import tensorflow as tf
from sklearn.cluster import KMeans


def lookup_model_input(FEATURE, feat_img, model_dict, feat_img_2=None):
    if FEATURE == 'Paper':
        wtrmrk = cv2.resize(feat_img, (480, 480))
        wtrmrk = ((wtrmrk - np.mean(wtrmrk)) / np.std(wtrmrk))

        note_loc = cv2.resize(feat_img_2, (480, 480))
        note_loc = ((note_loc - np.mean(note_loc)) / np.std(note_loc))
        model_input = np.concatenate((wtrmrk, note_loc), axis=2)[None, :, :, :]

    elif FEATURE == 'Treasury Seal':  # trs seal model multi input cnn
        tool = model_dict['edge_tool']

        teeth_indexs = np.hstack((np.arange(0, 300), np.arange(830, feat_img.shape[0])))
        teeth = feat_img[teeth_indexs, :, :]
        teeth = teeth[:, int(teeth.shape[1] * 0.25):int(teeth.shape[1] * (1 - 0.25))]
        teeth = cv2.resize(teeth, (480, 480))
        teeth_edges = tool.predict(teeth[None, :, :, :])[0][0, :, :, 0] * 255
        teeth_edges = ((teeth_edges - np.mean(teeth_edges)) / np.std(teeth_edges))
        teeth_edges = cv2.cvtColor(teeth_edges, cv2.COLOR_GRAY2BGR)

        crest = feat_img[300:830, 300:800, :]
        crest = cv2.resize(crest, (480, 480))
        crest = ((crest - np.mean(crest)) / np.std(crest))
        model_input = np.concatenate((crest, teeth_edges), axis=2)[None, :, :, :]

    elif FEATURE == 'Stripe':  # stripe model multi input lbp with aug
        tool = model_dict['edge_tool']
        lbp_tool = model_dict['lbp_tool']  # radius = 1.631, points = 6

        bridge = feat_img[1234: 1234 + 465, :]
        bridge = cv2.resize(bridge, (480, 480))
        bridge = tool.predict(bridge[None, :, :, :])[0][0, :, :, 0] * 255
        bridge = lbp_tool.describe(bridge)

        zoom = feat_img[2200:2600, :, :]
        zoom = zoom[:, int(zoom.shape[1] * 0.25):int(zoom.shape[1] * (1 - 0.25))]
        zoom = cv2.resize(zoom, (480, 480))
        zoom = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
        zoom = lbp_tool.describe(zoom)

        hist_all = np.vstack((zoom, bridge))
        model_input = hist_all[None, :, :].transpose(0, 2, 1)

    elif FEATURE == '100 OVI':
        tool = model_dict['edge_tool']
        ovi = cv2.resize(feat_img, (480, 480))
        ovi = tool.predict(ovi[None, :, :, :])[0][0, :, :, 0] * 255
        ovi = cv2.cvtColor(ovi, cv2.COLOR_GRAY2BGR)
        ovi = np.round(ovi)
        ovi = ((ovi - np.mean(ovi)) / np.std(ovi))
        model_input = ovi[None, :, :, :]

    elif FEATURE == 'Federal Seal':
        tool = model_dict['edge_tool']
        fed_seal = cv2.resize(feat_img, (480, 480))
        fed_seal_edge = tool.predict(fed_seal[None, :, :, :])[0][0, :, :, 0] * 255
        fed_seal_edge = cv2.resize(fed_seal_edge, (128, 128))
        fed_seal_edge = np.expand_dims(np.array(fed_seal_edge), axis=-1)
        model_input = fed_seal_edge
    else:
        model_input = None

    return model_input


def fedSealPrediction(image):
    model = loadModel()

    overlay, labels = getAutoSegment(image, 3)
    minLabel = np.argmin([np.percentile(np.where(labels == i, image, 255), 25) for i in range(3)])
    reqSegment = cv2.resize(np.where(labels == minLabel, image, 255), (128, 128))

    testImage = np.expand_dims(np.array(reqSegment), axis=-1)

    return np.argmax(model.predict(testImage))

def runKMeans(array, k):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(array.ravel().reshape(-1, 1))
    return kmeans.labels_


def restoreArrayToImage(labelArr, originalArr):
    shp = originalArr.shape
    return labelArr.reshape((shp[0], shp[1]))


def displayOverlay(im, mask):
    colorList = [(0,0,255), (0,255,0), (255,0,0), (255,0,255),
                 (255,255,0), (0,255,255), (255,100,255),
                 (0,0,128), (0,128,0), (128,0,0), (128,0,128),
                 (128,128,0), (0,128,128), (128,100,128)]
    #colorList = [(0,0,255), (255,255,255)]
    shp = im.shape
    maskRGB = np.zeros((shp[0], shp[1], 3))
    uniqueSegs = list(np.unique(mask))
    for s in uniqueSegs:
        if s == 0: continue
        colour = colorList[s]
        segColour = np.zeros((shp[0], shp[1], 3))
        segColour[:, :, 2] = np.where(mask == s, colour[0], 0)
        segColour[:, :, 1] = np.where(mask == s, colour[1], 0)
        segColour[:, :, 0] = np.where(mask == s, colour[2], 0)
        maskRGB += segColour
    im = np.array(cv2.normalize(im, None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    imRGB = np.zeros((shp[0], shp[1], 3))
    for i in range(3):
        imRGB[:, :, i] = im
    return cv2.addWeighted(imRGB, 0.1, maskRGB, 0.9, 0)


def getAutoSegment(sample, numSegs):
    #sample = cv2.GaussianBlur(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    #sample = sample[:int(0.5*sample.shape[0]), int(0.5*sample.shape[1]):]
    labels = runKMeans(sample, numSegs)
    labelIm = restoreArrayToImage(labels, sample)
    return displayOverlay(sample, labelIm), labelIm

# ####  #   TEST ## ## # #


#
# radius = 1.631
# points = 6
# lbp_class = LocalBinaryPatterns(points, radius)
#
# k=1
# tool = hed('./model_dir/weights_robust_lighting_texture.h5')
# feature = '100ovi'
# for trs_seal in os.listdir('D:/scoring_and_profiling/{}'.format(feature)):
#     model_input = lookup_model_input(feature, cv2.imread('D:/scoring_and_profiling/{}/'.format(feature) + trs_seal, 1), {'edge_tool': tool, 'lbp_tool': lbp_class})
#     pred = model.predict(model_input)
#     if 'genuine' in trs_seal:
#         truth = 1
#     else:
#         truth = 0
#     print(f'truth: {truth}, pred: {np.argmax(pred)}')
#     if truth != np.argmax(pred):
#         print('fucked up {}'.format(k))
#         k += 1