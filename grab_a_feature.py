import albumentations as A
import os
import cv2


def augmentImage(image, n):
    aug = A.Compose([
        # A.Flip(p=0.25),
        A.RandomGamma(gamma_limit=(20, 300), p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.Rotate(limit=2, p=0.6),
        # A.RandomRotate90(p=0.25),
        # A.RGBShift(p=0.75),
        A.GaussNoise(p=0.25)
    ])

    outData = []
    for i in range(n):
        augmented = aug(image=image)
        outData.append(augmented['image'])
    return outData


def getAllFeatures(path, positive=False, label = False):
    for noteFolder in os.listdir(path):
        for feature in os.listdir(path + noteFolder):
            name = feature.split('_')
            name = name[0][0:5] + '_' + name[1] + '_' + name[2] + '.bmp'
            if positive:
                if 'treasuryseal' in name:
                    image = cv2.imread(path + noteFolder + '/' + feature, 0)
                    augList = augmentImage(image, 15)
                    if label is False:
                        cv2.imwrite(savePath + name[0:4] + '.bmp', image)
                    else:
                        cv2.imwrite(savePath + name[0:4] + '_{}.bmp'.format(label), image)

                    for idx, augImage in enumerate(augList):
                        if label is False:
                            cv2.imwrite(savePath + name[0:-4] + '_{}.bmp'.format(idx + 1), augImage)
                        else:
                            cv2.imwrite(savePath + name[0:-4] + '_{}_{}.bmp'.format(idx + 1, label), augImage)

                continue
            if not 'back_ovithread' in name:
                image = cv2.imread(path + noteFolder + '/' + feature, 0)
                cv2.imwrite(savePath + name, image)
                augList = augmentImage(image, 5)
                cv2.imwrite(savePath + name, image)
                for idx, augImage in enumerate(augList):
                    cv2.imwrite(savePath + name[0:-4] + '_{}.bmp'.format(idx + 1), augImage)


if __name__ == '__main__':
    savePath = './pl_train_station/input_hed_images/'
    featPath = 'D:/FRNLib/featureLibrary/Genuine/100/'
    getAllFeatures(featPath, positive = True,label = 'genuine')
    savePath = './pl_train_station/input_hed_images/'
    featPath = 'D:/FRNLib/featureLibrary/Counterfeit/100/'
    getAllFeatures(featPath, positive=True, label = 'counterfeit')