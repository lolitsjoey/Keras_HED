# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:37:49 2020

@author: chris
"""

import cv2
import random
import numpy as np
from operator import add
from scipy import ndimage

def getLaserImage(path):
    cap = cv2.VideoCapture(path)

    ret = True
    augList = []
    while ret == True:
        capList = []
        debList = []
        ret, x = cap.read()
        if not ret:
            break
        vals = x[:,:,1][:,3]
        togs = x[:,:,1][:,2]
        togs = np.where(togs > 9, togs-89, togs)
        vals = vals*(1/255.0)

        capList.append(vals)
        debList.append(togs)
        k = np.array(list(map(add, capList, debList)))
        augList.append(k)

    capArr = np.array(augList)[:,0,:]

    # Rescale to 0-255 and int
    capArr = np.uint8((capArr - capArr.min()) * 255 / np.percentile(capArr, 99.9))   #capArr.max())

    lowpass = ndimage.gaussian_filter(capArr, 1)
    gauss_highpass = capArr - lowpass

    kernel = np.ones((3,3), np.float32)/8
    imRet = cv2.filter2D(gauss_highpass, -1, kernel)

    im = cv2.flip(imRet, 1)
    im = cv2.resize(im, (2000, 800))
    return im


def getWarp(im):
    contours, hierarchy = cv2.findContours(im.copy().astype('uint8') ,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    contour = c #contours[0]

    rotrect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)

    # get width and height of the detected rectangle
    width = int(rotrect[1][0])
    height = int(rotrect[1][1])
    if width < height:
        temp = width
        width = height
        height = temp
    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M, width, height

def straightenNote(image, imageIR=None):
    # Rotates a given input image to adjust for camera misalignment
    # KNOWN ISSUES:
        # Cannot currently determine if image is upside down (AI)
        # Background paper remains because it's too similar to the note
        #     (This should be dealt with in object detection)
    # Pad border

    image = cv2.copyMakeBorder(image, 0, 0, 0, 0, cv2.BORDER_CONSTANT, None, [0,0,0])
    if imageIR is not None:
        imageIR = cv2.copyMakeBorder(imageIR, 0, 0, 0, 0, cv2.BORDER_CONSTANT, None, [0,0,0])
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((200,200), np.float32)/40000
    im = cv2.filter2D(im, -1, kernel)
    im[im > 60] = 255
    im[im < 65] = 0

    M, w, h = getWarp(im)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(image, M, (w, h))

    if imageIR is not None:
        warpedIR = cv2.warpPerspective(imageIR, M, (w, h))
    else:
        warpedIR = None

    return warped, warpedIR

def straightenHS(imageHS):

    im = imageHS[:, :, 100]
    im = 255*np.array(cv2.normalize(im, None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    kernel = np.ones((4,4), np.float32)/16
    im = cv2.filter2D(im, -1, kernel)
    im[im > 60] = 255
    im[im < 65] = 0

    M, w, h = getWarp(im)
    # directly warp the rotated rectangle to get the straightened rectangle
    numSlices = imageHS.shape[2]
    return np.array([cv2.warpPerspective(imageHS[:,:,x], M, (w, h)) for
                     x in range(numSlices)]).transpose(1, 2, 0)

def padLP(imageLP):
    padTop = 100
    padBottom = 0
    return cv2.copyMakeBorder(imageLP, padTop, padBottom, 0, 0,
                       cv2.BORDER_CONSTANT, None, [0,0,0])

def straightenLP(imageLP):
    pad = 200
    imLP = cv2.copyMakeBorder(imageLP, pad, pad, pad, pad,
                           cv2.BORDER_CONSTANT, None, [0,0,0])
    kernel = np.ones((50,50), np.float32)/2500
    im = cv2.filter2D(imLP, -1, kernel)
    im[im > 40] = 255
    im[im < 45] = 0
    M, w, h = getWarp(im)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(imLP, M, (w, h))
    if w < h:
        warped = cv2.rotate(warped, 2)
    return padLP(warped)

def templateMatch(image, template):
    kernel = np.ones((5,5),np.float32)/25
    image = cv2.filter2D(image,-1,kernel)
    template = cv2.filter2D(template,-1,kernel)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res == res.max())
    return loc

def determineDenom(im, df):
    matchScore = []
    matchList = []
    df = df.reset_index(drop=True)
    im = cv2.cvtColor(cv2.resize(im, (1200, 500)), cv2.COLOR_BGR2GRAY)
    for idx, row in df.iterrows():
        template = row['image']
        for scale in [0.9, 0.95, 1]:
            template = cv2.resize(template, (int(1200*scale), int(500*scale)))
            res = cv2.matchTemplate(im, template, cv2.TM_CCOEFF_NORMED)
            matchScore.append(res.max())
            matchList.append([row['Denomination'], scale])
    bestFit = matchList[matchScore.index(max(matchScore))]
    denom = bestFit[0]
    scale = bestFit[1]
    print("Denom: {}, Scale: {}".format(denom, scale))
    return denom, scale

def extractFeatureLocation(note, featDict):
    if featDict['Side'] == 'Front':
        im = note.thumbs.rgbFront[:,:,2]
    else:
        im = note.thumbs.rgbBack[:,:,2]

    featureIm = featDict['image']
    scale = 1200, 500

    heightOriginal, widthOriginal = note.images.rgbFront.shape[:2]
    heightRat, widthRat = heightOriginal / scale[1], widthOriginal / scale[0]   # Numbers swapped to account for rotation

    # Rough Area Approximation Values
    minX = int(scale[0] * featDict['MinX'])
    minY = int(scale[1] * featDict['MinY'])
    maxX = int(scale[0] * featDict['MaxX'])
    maxY = int(scale[1] * featDict['MaxY'])
    im = np.array(im)
    #print("{} - {}".format(featDict['Side'], featDict['Feature Name']))
    lz = templateMatch(im[minY:maxY, minX:maxX], featureIm[5:-5,5:-5])
    w, h = featureIm.shape[::-1]
    for pt in zip(*lz[::-1]):
        # Scaled points
        minOri = int((minX + pt[0])*widthRat), int((minY + pt[1])*heightRat)
        maxOri = int((minX + pt[0] + w)*widthRat), int((minY + pt[1] + h)*heightRat)

        minOri = (minX + pt[0]) / scale[0], (minY + pt[1]) / scale[1]
        maxOri = (minX + pt[0] + w) / scale[0], (minY + pt[1] + h) / scale[1]

        location = {'MinX' : minOri[0], 'MinY' : minOri[1],
                    'MaxX' : maxOri[0], 'MaxY' : maxOri[1]}
    return location

def inspectFeature(inDict):
    im = inDict['Image']
    shp = im.shape
    specDict = {'Red' : 2,
                'Green' : 1,
                'Blue' : 0
                }
    resultsDict = {'Feature Name' : inDict['Feature Name'],
                   'Side' : inDict['Side']}
    if shp[0] < 250 or shp[1] < 250:
        resultsDict['Results'] = None
        return resultsDict
    model = inDict['Model']
    imList = []
    for _ in range(5):
        hMin = random.randint(0, shp[0]-210)
        wMin = random.randint(0, shp[1]-210)
        hMax = hMin + 200
        wMax = wMin + 200
        imList.append(np.expand_dims(im[hMin:hMax,wMin:wMax,specDict[inDict['Spectrum']]], axis=-1))
    imArr = np.array(imList) / 255.0
    results = predictImageSet(imArr, model)
    if results is not None:
        resultMean = np.mean(results[:,1])
    else:
        resultMean = None
    resultsDict = {'Index' : inDict['Index'],
                   'Score' : resultMean
                   }
    return resultsDict

def binariseImages(img, thresh1, thresh2, denoise=None, open=None, close=None, blur=None):
    """
    Function to binarise an image of a serial number, by blocking out background noise and extracting
    just the alphanumeric characters.
    Args:
        img: image to be binarised
        thresh1, thresh2, denoise, open, close, blur: opencv parameters to be tweaked to get optimum
        binarisation
    Returns:
        final: the binarised version of the input image
    Function parameters being used:
        new function parameters (for rgb): (path,60,80,10,10,15,3)
    """

    if denoise:
        dst = cv2.fastNlMeansDenoisingColored(img, None, denoise, denoise, 7, 21)
        dst[:,:,0] = np.zeros(dst[:,:,0].shape)
        dst[:,:,1] = np.zeros(dst[:,:,1].shape)
        ret, thresh_1 = cv2.threshold(dst, thresh1, 255, cv2.THRESH_BINARY)
    else:
        img[:,:,0] = np.zeros(img[:,:,0].shape)
        img[:,:,1] = np.zeros(img[:,:,1].shape)
        ret, thresh_1 = cv2.threshold(img, thresh1, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(thresh_1, cv2.COLOR_BGR2GRAY)

    ret, thresh_2 = cv2.threshold(gray, thresh2, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(thresh_2)

    if blur:
        blurred = cv2.GaussianBlur(thresh_2,(blur,blur),0)
        inv = cv2.bitwise_not(blurred)
    else:
        inv = cv2.bitwise_not(thresh_2)

    if open and close:
        open_krn = np.ones((open,open),np.uint8)
        cls_krn = np.ones((close,close),np.uint8)
        opening = cv2.morphologyEx(inv, cv2.MORPH_OPEN, open_krn)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cls_krn)
        final = closing
    elif open:
        open_krn = np.ones((open,open),np.uint8)
        opening = cv2.morphologyEx(inv, cv2.MORPH_OPEN, open_krn)
        final = opening
    else:
        final = inv

    return final

def serialNumberCharacterDistances(img):
    """
    Function to find the distances between characters of a binarized image
    of the serial number
    Args:
        image: binarised image of a serial number
    Returns:
        corrected_distances: list of floats correpsonding to the distances between
                    the respective characters
        error_list: list of zeros returned when one or more types of errors are encountered
    """
    note_width_pixels = 15800 # put in config (CFG)
    ratio = 156/note_width_pixels # the size of 1 pixel
    # 156 from above goes into config (CFG)
    profile = img.sum(axis=0) # sum along each vertical columng to give an array of the same size as the width of the image

    threshold = 2000 # CFG
    positions = [index for index, n in enumerate(profile) if n > threshold] # all x values with a correspoinding y value greater than the threshold

    characters_start_and_end = []
    for i in range(len(positions) - 1):
        p1 = positions[i]
        p2 = positions[i+1]
        if abs(p2 -p1) > 5:
            characters_start_and_end.append([p1, p2])

    distances = [round((points[1]*ratio - points[0]*ratio),2) for points in  characters_start_and_end]

    while len(distances) > 10:
        distances.remove(min(distances))

    # offset correction
    offset = [0.35,0.6,0.35,0.35,0.35,0.35,0.35,0.35,0.35,0.2]

    corrected_distances = [x+y for x,y in zip(distances, offset)]
    corrected_distances = [round(x,2) for x in corrected_distances]

    # error handling
    error_list = [0 for number in range(10)]

    if len(distances) == 10:
        if distances[0] >= 1:
            print('Incorrect extraction: Other partial features in image') # when other features partially appear
            return error_list
        else:
            print('Serial Number analysed successfully')
            return corrected_distances
    else:
        print('Incorrect input image') # happens when feature extraction messes up and misses the sn completely
        return error_list

def predictImageSet(imSet, model):
    try:
        return model.predict(imSet)
    except AttributeError:
        return None

def generateJSON(note):
    jsonContents = {
        "attributes" : getAttributes(note),
        "features" : parseFeaturesToJSON(note.tags, note.subtags),
        "id" : note.id,
        "images" : getImageLocations(note),
        "metadata" : getMetaData(note),
        "tags" : 2
        }
    return jsonContents

def getAttributes(note):
    ret = {
            "denomination": note.denom,
            "sheetPosition": "",
            "serialNumberLeft": "LZ53569532W",
            "serialNumberRight": "",
            "plateNumberFront": "",
            "plateNumberBack": "",
            "series": "SERIES 2009A",
            "facilityId": "",
            "distances": {
              "serialNumberLeft": {
                "L to Z": 0.95,
                "Z to 5": 1.6,
                "5 to 3": 0.9,
                "3 to 5": 0.91,
                "5 to 6": 0.95,
                "6 to 9": 0.88,
                "9 to 5": 0.9,
                "3 to 2": 0.93,
                "2 to W": 1.71
              },
              "serialNumberRight": {
                "1st to 2nd": 1.05,
                "2nd to 3rd": 1.49,
                "3rd to 4th": 0.9,
                "4th to 5th": 0.85,
                "5th to 6th": 0.5,
                "6th to 7th": 0.68,
                "7th to 8th": 0.93,
                "8th to 9th": 0.95,
                "9th to 10th": 1.77,
                "10th to 11th": 1.21
                }
              }
            }
    return ret

def parseFeaturesToJSON(dfTags, dfSubTags):
    featureList = []
    for tagIdx, tagRow in dfTags.iterrows():
        subTags = dfSubTags[dfSubTags['Feature Name'] == tagRow['Feature Name']]
        tagDict = []
        for subTagIdx, subTagRow in subTags.iterrows():
            tagDict.append([{
                "name" : subTagRow['Inspection Type'],
                "score" : subTagRow['Score'],
                "value" : "PLACEHOLDER"
                }])
        featureParse = {
          "featureType": tagRow['Feature Name'],
          "score": tagRow['Score'] * 100,
          "rect": {
            "minX": tagRow['MinX'],
            "minY": tagRow['MinY'],
            "maxX": tagRow['MaxX'],
            "maxY": tagRow['MaxY']
          },
          "tags": tagDict
        }
        featureList.append(featureParse)
    return featureList

def getImageLocations(note):
    imageLocs = {"images": {
        "bgr": {
          "front": {
            "fullSize": "C:/Output/{note.id}/RGB/Front/{note.id}.bmp",
            "thumbnail": "C:/Output/{note.id}/RGB/Front/{note.id}_Thumb.bmp"
          },
          "back": {
            "fullSize": "C:/Output/{note.id}/RGB/Back/{note.id}.bmp",
            "thumbnail": "C:/Output/{note.id}/RGB/Back/{note.id}_Thumb.bmp"
          }
        },
        "ir": {
          "front": {
            "fullSize": "C:/Output/{note.id}/IR/Front/{note.id}.bmp",
            "thumbnail": "C:/Output/{note.id}/IR/Front/{note.id}_Thumb.bmp"
          },
          "back": {
            "fullSize": "C:/Output/{note.id}/IR/Back/{note.id}.bmp",
            "thumbnail": "C:/Output/{note.id}/IR/Back/{note.id}_Thumb.bmp"
          }
        },
        "hs": {
          "front": {},
          "back": {}
        },
        "laser": {
          "front": {
            "0": "C:/Output/3be9e8a4-efd0-4de1-a2d4-1a7e0041ee1f_Front_LP_0.png"
          },
          "back": {
            "0": "C:/Output/3be9e8a4-efd0-4de1-a2d4-1a7e0041ee1f_Back_LP_0.png"
          }
        }
      }
    }
    return imageLocs

def getMetaData(note):
    metaData = {
    "machineId": "128d80a0-19cd-4d85-ae91-c6fe781949ed",
    "designFamily": "ABC",
    "reportFormId": "ZXC",
    "timeInspected": "2020-09-24T11:57:20.9793510Z",
    "timeReported": "2020-09-24T11:57:20.9793510Z"
    }
    return metaData

def writeJSON(jsonFile):
    pass
