import os
import cv2
import time
from random import randint
import numpy as np
import pywt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from hyperspec_classifier.notemethods import straightenHS
import pandas as pd


class HyperSpecImage:
    def __init__(self, pathHDR, pathRAW, bString=False, rotation=-1, hFlip=False, vFlip=False):
        self.status      = False
        self.inputType   = 'Bytes' if bString else 'File'
        self.info        = self.readHeaderDict(pathHDR)
        self.array       = self.readData(pathRAW)
        self.path        = {'HDR': pathHDR,
                            'RAW': pathRAW}
        self.setRotation(rotation)
        self.setFlip(hFlip, vFlip)

    def readHeaderDict(self, pathHDR):
        with open(pathHDR, 'r') as hf:
            hdrFile = hf.read()
        return self.parseHDRToDict(hdrFile)

    def readData(self, pathRAW):
        with open(pathRAW, 'rb') as df:
            rawFile = df.read()
        return self.convertByteStringToArray(rawFile, self.info)

    def setRotation(self, rot):
        if rot not in [-1, 0, 1, 2, 90, 180, 270]:
            print(f'Invalid rotation value, choose 90, 180, or 270. (The input was {rot})')
            raise ValueError
        if rot == 0 or rot == 90:
            self.array = np.rot90(self.array, 1, axes=(0,1))
        if rot == 1 or rot == 180:
            self.array = np.rot90(self.array, 2, axes=(0,1))
        if rot == 2 or rot == 270:
            self.array = np.rot90(self.array, 3, axes=(0,1))

    def setFlip(self, horizontal=False, vertical=False):
        if horizontal:
            self.array = np.fliplr(self.array)
        if vertical:
            self.array = np.flipud(self.array)

    def parseHDRToDict(self, hdrString):
        splitStr = hdrString.replace(',\n', ',').split('\n')
        headerDict = {}
        for idx, row in enumerate(splitStr):
            if 'samples' == row[:7]:
                headerDict['samples'] = int(row.split('=')[1].replace(' ', ''))
            if 'bands' == row[:5]:
                headerDict['bands'] = int(row.split('=')[1].replace(' ', ''))
            if 'lines' == row[:5]:
                headerDict['lines'] = int(row.split('=')[1].replace(' ', ''))
            if 'wavelength' in row.lower():
                headerDict['wavelengths'] = [float(w) for w in splitStr[idx+1].split(',') if w!= '}']
        return headerDict

    def convertByteStringToArray(self, bString, hDict):
        arr = np.frombuffer(bString, dtype='<u2')
        try:
            arr = np.reshape(arr, (hDict['lines'], hDict['bands'], hDict['samples']))
            arr = np.transpose(arr, axes=(0,2,1))
        except ValueError:
            print('ERROR: The collected byte string could not be reshaped. Diagnosing...')
            arr = self.diagnoseConversionError(bString, hDict)
        if arr is not None:
            self.status = True
        return arr

    def diagnoseConversionError(self, bString, hDict):
        expectedSize  = hDict['lines'] * hDict['bands'] * hDict['samples'] * 2
        actualSize    = len(bString)
        diff          = expectedSize - actualSize
        if not diff:
            print('The file is corrupted and could not be recovered')
            return None
        print('The byte string is too short, data might be missing. Filling with zeroes and retrying...')
        bString += b'0' * diff
        arr = np.frombuffer(bString, dtype='<u2')
        arr = np.reshape(arr, (hDict['lines'], hDict['bands'], hDict['samples']))
        arr = np.transpose(arr, axes=(0, 2, 1))
        return arr

    def viewSample(self, band):
        im = np.array(cv2.normalize(self.array[:,:,band], None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        wavelength = self.info['wavelengths'][band]
        cv2.imshow(f'sampleband {wavelength}', im)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def cropAndStraighten(self):
        st = time.time()
        self.array = self.array[250:-250, 20:-20, :]
        self.array = straightenHS(self.array)
        print(f'Time to straighten: {round(time.time() - st, 2)}s')

    def repair(self, pathOut):
        if not self.status:
            print(f'The file {self.path["HDR"]} could not be recovered')
            return False

        # Write out HDR
        with open(self.path['HDR'], 'r') as rf:
            temp = rf.read()
        outHDR = self.path['HDR'].replace('/noteLibrary/', pathOut)
        if not os.path.exists(outHDR): os.makedirs(outHDR)
        with open(outHDR, 'w+') as wf:
            wf.write(temp)

        # Write out RAW
        with open(self.path['RAW'], 'rb') as rf:
            temp = rf.read()
        outHDR = self.path['RAW'].replace('/noteLibrary/', pathOut)
        if not os.path.exists(outHDR):
            os.makedirs(outHDR)
        with open(outHDR, 'wb+') as wf:
            wf.write(temp)

def displayOverlay(im, mask):
    colorList = [(0,0,255), (0,255,0), (255,0,0), (255,0,255),
                 (255,255,0), (0,255,255), (255,100,255),
                 (0,0,128), (0,128,0), (128,0,0), (128,0,128),
                 (128,128,0), (0,128,128), (128,100,128)]

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
        imRGB[:,:,i] = im

    return cv2.addWeighted(imRGB, 0.9, maskRGB, 0.1, 0)

def runKMeans(array, k):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(array)
    return kmeans.labels_

def flattenHSToDataFrame(arr):
    shp = arr.shape
    return arr.reshape((shp[0]*shp[1], shp[2]))

def restoreArrayToImage(labelArr, originalArr):
    shp = originalArr.shape
    return labelArr.reshape((shp[0], shp[1]))

def reorder_labels(labels, order):
    labels_new = labels.copy()
    for idx in order:
        labels_new[np.where(labels==order[idx])] = idx

    return labels_new

def extractMasks(labels, seg):
    mask = labels.copy()
    on = np.where(labels == seg)
    off = np.where(labels != seg)
    mask[on] = 1
    mask[off] = 0
    return mask


def write_hyper_spec_from_array(folder_of_folders, hyperspec_destination, num_segs, fileString, scales, wavelet, feature, do_pca=False):
    kk = 0
    for img_num, img_folder in enumerate(os.listdir(folder_of_folders)):
        noteDir = folder_of_folders + img_folder
        if '_' + feature + '.' not in noteDir:
            continue
        hsSample = np.load(noteDir)
        try:
            arrMod = flattenHSToDataFrame(hsSample)
        except Exception:
            print(hsSample.shape)
            print(noteDir)
            print(img_folder)
            print(folder_of_folders)
        labels = runKMeans(arrMod, num_segs)

        globalMean = np.mean(arrMod, axis=0)
        smooth = savgol_filter(globalMean, 201, 2)

        hyper_arr = np.zeros((len(np.unique(labels)), 224))
        peak = []
        for idx, label in enumerate(np.unique(labels)):

            for band in range(224):
                hyper_arr[idx, band] = np.mean(arrMod[labels == label, band]) - smooth[band]
            peak.append(np.max(hyper_arr[idx, :]))

        order = np.argsort(peak)
        # labels = reorder_labels(labels, order)

        for true_seg, seg in enumerate(order):
            with open(hyperspec_destination + './{}_{}_{}_{}_{}.txt'.format(folder_of_folders.split('/')[-2][0:-4], kk, true_seg, fileString, feature),
                      'w+') as output:
                output.write(str(hyper_arr[seg, :]))

            mask = extractMasks(labels, seg)
            labelIm = restoreArrayToImage(mask, hsSample)
            overlay = displayOverlay(hsSample[:, :, 70], labelIm)
            cv2.imwrite(
                hyperspec_destination + 'rgb/{}_{}_{}_{}_{}.bmp'.format(folder_of_folders.split('/')[-2][0:-4], kk, true_seg, fileString, feature),
                overlay * 255)

            coeffs, freqs = pywt.cwt(hyper_arr[seg, :], scales, wavelet)
            old_value = coeffs
            old_min = np.min(coeffs)
            old_max = np.max(coeffs)
            new_min = 0
            new_max = 256.
            new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            new_value = np.array(new_value, dtype='uint8')
            im_color = cv2.applyColorMap(new_value, cv2.COLORMAP_HOT)
            cv2.imwrite(
                hyperspec_destination + 'cwt/{}_{}_{}_{}_{}_waves.bmp'.format(folder_of_folders.split('/')[-2][0:-4], kk, true_seg, fileString, feature),
                im_color)
        kk += 1


def write_hyper_spec(folder_of_folders, hyperspec_destination, num_segs, fileString, scales, wavelet, do_pca=False):
    if do_pca:
        pca_frame = pd.DataFrame()
        pca_list = []
        index_list = []
    for img_num, img_folder in enumerate(os.listdir(folder_of_folders)):
        noteDir = folder_of_folders + img_folder
        hsList = [f for f in os.listdir(noteDir) if 'HSI' in f]
        try:
            frontHDR = noteDir + '/' + hsList[2]
            frontRAW = noteDir + '/' + hsList[3]
        except IndexError:
            print('Cant find hyperspec for {}, skipping (1) note'.format(img_folder))
            continue

        if fileString == 'genuine':
            #hsFront = HyperSpecImage(frontHDR, frontRAW, rotation=270, hFlip=False, vFlip=True)
            hsFront = HyperSpecImage(frontHDR, frontRAW, rotation=90, hFlip=True, vFlip=True)
        else:
            hsFront = HyperSpecImage(frontHDR, frontRAW, rotation=0, hFlip=True, vFlip=True)
        hsFront.cropAndStraighten()

        hsSample = np.array([hsFront.array[260:380,630:770,i] for i in range(0,224)]).transpose((1,2,0))

        arrMod = flattenHSToDataFrame(hsSample)
        labels = runKMeans(arrMod, num_segs)

        globalMean = np.mean(arrMod, axis=0)
        smooth = savgol_filter(globalMean, 201, 2)

        hyper_arr = np.zeros((len(np.unique(labels)), 224))
        peak = []
        for idx,label in enumerate(np.unique(labels)):

            for band in range(224):
                hyper_arr[idx,band] = np.mean(arrMod[labels == label,band]) - smooth[band]
            peak.append(np.max(hyper_arr[idx,:]))

        order = np.argsort(peak)
        #labels = reorder_labels(labels, order)

        for true_seg, seg in enumerate(order):
            with open(hyperspec_destination + "./{}_file{}_{}.{}.txt".format(fileString, img_num, seg, num_segs), 'w+') as output:
                output.write(str(hyper_arr[seg,:]))

            mask = extractMasks(labels, seg)
            labelIm = restoreArrayToImage(mask, hsSample)
            overlay = displayOverlay(hsSample[:, :, 70], labelIm)
            cv2.imwrite(
                hyperspec_destination + 'rgb/{}_file{}_{}.{}.jpg'.format(fileString, img_num, true_seg, num_segs),
                overlay * 255)

            coeffs, freqs = pywt.cwt(hyper_arr[seg,:], scales, wavelet)
            old_value = coeffs
            old_min = np.min(coeffs)
            old_max = np.max(coeffs)
            new_min = 0
            new_max = 256.
            new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            new_value = np.array(new_value, dtype='uint8')
            im_color = cv2.applyColorMap(new_value, cv2.COLORMAP_HOT)
            cv2.imwrite(hyperspec_destination + 'cwt/{}_file{}_{}.{}_waves.bmp'.format(fileString, img_num, true_seg, num_segs),
                                                                                                                        im_color)
            if do_pca:
                coeffs = (coeffs - coeffs.mean())/coeffs.std()
                pca = PCA(n_components=1)
                feat_vec = pca.fit_transform(coeffs).flatten()
                pca_list.append(feat_vec)
                index_list.append('{}_file{}_{}.{}'.format(fileString, img_num, true_seg, num_segs))


    if do_pca:
        pca_frame = pd.DataFrame(pca_list)
        pca_frame.index = index_list
        pca_frame.to_csv(hyperspec_destination + '/pca_frame_' + fileString +  '.csv')

def scale_values(values):
    old_min = np.min(values)
    old_max = np.max(values)
    new_min = 0
    new_max = 1.
    new_values = ((values - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    new_values = np.round(new_values, 5)
    return new_values


def write_hyper_spec_just_feature(folder_of_folders, hyperspec_destination, num_segs, fileString, indicator, scales, wavelet, do_pca=False):
    if do_pca:
        pca_frame = pd.DataFrame()
        pca_list = []
        index_list = []
    for img_num, hyperspec in enumerate(os.listdir(folder_of_folders)):
        if '.hdr' not in hyperspec:
            continue
        frontHDR = folder_of_folders + hyperspec
        frontRAW = folder_of_folders + hyperspec[0:-4] + '.raw'

        note_number = frontHDR.replace('.','/').replace('_','/').split('/')[-3]
        save_name = fileString + str(int(note_number) - 1) + indicator + 'rig_' + note_number + '_HSI_featName'

        hsFront = HyperSpecImage(frontHDR, frontRAW, rotation=90, hFlip=True, vFlip=True)
        hsFront.cropAndStraighten()
        hsSample = np.array([hsFront.array[:,:,i] for i in range(0,224)]).transpose((1,2,0))

        arrMod = flattenHSToDataFrame(hsSample)
        labels = runKMeans(arrMod, num_segs)

        globalMean = np.mean(arrMod, axis=0)
        smooth = savgol_filter(globalMean, 201, 2)

        hyper_arr = np.zeros((len(np.unique(labels)), 224))
        peak = []
        for idx,label in enumerate(np.unique(labels)):

            for band in range(224):
                hyper_arr[idx,band] = np.mean(arrMod[labels == label,band]) - smooth[band]
            peak.append(np.max(hyper_arr[idx,:]))

        order = np.argsort(peak)
        #labels = reorder_labels(labels, order)

        for true_seg, seg in enumerate(order):
            coeffs, freqs = pywt.cwt(hyper_arr[seg,:], scales, wavelet)
            if do_pca:
                pca = PCA(n_components=1)
                feat_vec = pca.fit_transform(coeffs).flatten()
                pca_list.append(feat_vec)
                index_list.append(save_name + '_' + str(true_seg))


    if do_pca:
        pca_frame = pd.DataFrame(pca_list)
        old_values = pca_frame.values
        new_values = np.zeros(pca_frame.shape)
        for i in range(int(len(pca_list)/num_segs)):
            new_values[(i)*num_segs:(i+1)*num_segs, :] = scale_values(old_values[(i)*num_segs:(i+1)*num_segs, :])
        pca_frame = pd.DataFrame(new_values)
        pca_frame.index = index_list
        pca_frame.to_csv(hyperspec_destination + '/pca_frame_' + fileString + '.csv')
        print('oi oi')