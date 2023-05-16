import numpy as np
import pickle
import pandas as pd
import cv2
from tqdm import tqdm
import math
from transformers import ViTFeatureExtractor
from PIL import Image

def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def processRawData(padding, useVITFeature, gazePath, stimuliPath, saveFilePath, N=4):
    gazesExcel = pd.read_excel(gazePath)
    oneEntry = {'sub': None, 'imagePath': None, 'scanpath': [], 'imageSize': None,
                'patchIndex': None, 'scanpathInPatch': []}
    processed_dataset = []
    totalPoints = 0
    negativePoints = 0
    dataEntry = 0
    numOfRows = len(gazesExcel)
    allImages = {}
    tenPointSeq = 0
    if useVITFeature:
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        resizeFactor = 1
    else:
        # NOTE: hardcode to 2 in order to compare with customize VIT
        resizeFactor = 2

    for i in tqdm(range(numOfRows)):
        row = gazesExcel.loc[i]
        subject = row['Sub']
        task = row['Task']
        index = row['T']
        x_coor = row['X']
        y_coor = row['Y']

        # save the current entry and start a new one
        if index == 1 and i != 0: #and not negativeValue:
            assert oneEntry['sub'] is not None
            assert oneEntry['imagePath'] is not None
            assert oneEntry['imageSize'] is not None
            #assert oneEntry['imageFeature'] is not None

            if len(oneEntry['scanpath']) <= 9:
                tenPointSeq += 1
            else:
                oneEntry['scanpathInPatch'] = np.stack(oneEntry['scanpathInPatch'][:10])
                oneEntry['scanpath'] = oneEntry['scanpath'][:10]
                processed_dataset.append(oneEntry)
            dataEntry += 1
            oneEntry = {'sub': None, 'imagePath': None, 'scanpath': [], 'imageSize': None,
                        'patchIndex': None, 'scanpathInPatch': []}

        imagePath = stimuliPath + task #+ '.jpeg'
        if index == 1:
            oneEntry['sub'] = subject
            oneEntry['imagePath'] = task #imagePath

            # process image feature
            image1 = Image.open(imagePath)

            if useVITFeature:
                if padding:
                    image1 = make_square(image1)
                image_to_save = feature_extractor(image1)['pixel_values'][0]

            image = cv2.imread(imagePath)
            if not useVITFeature:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (int(image.shape[1] / resizeFactor), int(image.shape[0] / resizeFactor)))
                image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_32F)
                image_to_save = image
            imageH_original = image.shape[0]
            imageW_original = image.shape[1]
            oneEntry['imageSize'] = [imageH_original, imageW_original]
            # padding, make it dividable by N
            margin1 = N * (math.ceil(image.shape[0] / N)) - image.shape[0]
            margin2 = N * (math.ceil(image.shape[1] / N)) - image.shape[1]
            top = int(margin1 / 2)
            bottom = margin1 - top
            left = int(margin2 / 2)
            right = margin2 - left
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
            assert image.shape[0] % N == 0
            assert image.shape[1] % N == 0
            patchH = int(image.shape[0] / N)
            patchW = int(image.shape[1] / N)
            if task not in allImages:
                allImages[task] = image_to_save
        else:
            assert oneEntry['imagePath'] == task
            assert oneEntry['sub'] == subject
            assert oneEntry['imageSize'] == [imageH_original, imageW_original]
        if x_coor > 0 and y_coor > 0 and x_coor < imageW_original and y_coor < imageH_original:
            oneEntry['scanpath'].append([y_coor, x_coor])
            before = np.array([math.floor(y_coor / imageH_original * N), math.floor(x_coor / imageW_original * N)])
            pos = np.ravel_multi_index(before, (N, N))
            oneEntry['scanpathInPatch'].append(pos)
        else:
            negativePoints += 1
        totalPoints += 1

    #if not negativeValue:
    assert oneEntry['sub'] is not None
    assert oneEntry['imagePath'] is not None
    assert oneEntry['imageSize'] is not None
    oneEntry['scanpathInPatch'] = np.stack(oneEntry['scanpathInPatch'])
    if len(oneEntry['scanpath']) == 1:
        onePointSeq += 1
    else:
        processed_dataset.append(oneEntry)
    dataEntry += 1

    processed_dataset.append(allImages)
    if saveFilePath is not None:
        with open(saveFilePath, "wb") as fp:  # Pickling
            pickle.dump(processed_dataset, fp)
    print('# Negative/outOfImage points percentage, ', negativePoints/totalPoints)
    validPoints = totalPoints - negativePoints
    print('Valid points, ', validPoints)
    print('Invalid points, ', negativePoints)
    print('# Total data: ', dataEntry)
    avgLen = validPoints / dataEntry
    print('Average length: ', avgLen)
    print('# ten-point/zero-point gaze seq:', tenPointSeq)


def processRawData_joint(gazePath, stimuliPath, saveFilePath, N=4):
    gazesExcel = pd.read_excel(gazePath)
    oneEntry = {'sub': None, 'imagePath': None, 'scanpath': [],
                'scanpathInPatch': [], 'heatmap': None}
    processed_dataset = []
    totalPoints = 0
    negativePoints = 0
    dataEntry = 0
    numOfRows = len(gazesExcel)
    allImages = {}
    onePointSeq = 0
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    for i in tqdm(range(numOfRows)):
        row = gazesExcel.loc[i]
        subject = row['Sub']
        task = row['Task']
        index = row['T']
        x_coor = row['X']
        y_coor = row['Y']

        # save the current entry and start a new one
        if index == 1 and i != 0:
            assert oneEntry['sub'] is not None
            assert oneEntry['imagePath'] is not None

            if len(oneEntry['scanpath']) <= 1:
                onePointSeq += 1
            else:
                oneEntry['scanpathInPatch'] = np.stack(oneEntry['scanpathInPatch'])
                indices = oneEntry['scanpath']
                for ind in indices:
                    heatmap[ind[0],ind[1]] = 1
                oneEntry['heatmap'] = heatmap
                #assert oneEntry['heatmap'].sum() == len(oneEntry['scanpath'])
                processed_dataset.append(oneEntry)
            dataEntry += 1
            oneEntry = {'sub': None, 'imagePath': None, 'scanpath': [],
                        'scanpathInPatch': [], 'heatmap': None}

        imagePath = stimuliPath + task #+ '.jpeg'
        if index == 1:
            oneEntry['sub'] = subject
            oneEntry['imagePath'] = task #imagePath

            # process image feature
            image1 = Image.open(imagePath)
            image_to_save = feature_extractor(image1)['pixel_values'][0]

            image = cv2.imread(imagePath)
            heatmap = np.zeros((image.shape[0], image.shape[1]))
            patchH = image.shape[0] / N
            patchW = image.shape[1] / N
            if task not in allImages:
                allImages[task] = image_to_save
        else:
            assert oneEntry['imagePath'] == task
            assert oneEntry['sub'] == subject
        if x_coor > 0 and y_coor > 0 and x_coor < image.shape[1] and y_coor < image.shape[0]:
            oneEntry['scanpath'].append([y_coor, x_coor])
            assert math.floor(y_coor / patchH) < N
            assert math.floor(x_coor / patchW) < N

            before = np.array([math.floor(y_coor / patchH), math.floor(x_coor / patchW)])
            pos = np.ravel_multi_index(before, (N, N))
            oneEntry['scanpathInPatch'].append(pos)
        else:
            negativePoints += 1
        totalPoints += 1

    assert oneEntry['sub'] is not None
    assert oneEntry['imagePath'] is not None
    oneEntry['scanpathInPatch'] = np.stack(oneEntry['scanpathInPatch'])
    if len(oneEntry['scanpath']) == 1:
        onePointSeq += 1
    else:
        oneEntry['scanpathInPatch'] = np.stack(oneEntry['scanpathInPatch'])
        indices = oneEntry['scanpath']
        for ind in indices:
            heatmap[ind[0], ind[1]] = 1
        oneEntry['heatmap'] = heatmap
        # assert oneEntry['heatmap'].sum() == len(oneEntry['scanpath'])
        processed_dataset.append(oneEntry)

    dataEntry += 1
    processed_dataset.append(allImages)
    if saveFilePath is not None:
        with open(saveFilePath, "wb") as fp:  # Pickling
            pickle.dump(processed_dataset, fp)
    print('# Negative/outOfImage points percentage, ', negativePoints/totalPoints)
    validPoints = totalPoints - negativePoints
    print('Valid points, ', validPoints)
    print('Invalid points, ', negativePoints)
    print('# Total data: ', dataEntry)
    avgLen = validPoints / dataEntry
    print('Average length: ', avgLen)
    print('# one-point/zero-point gaze seq:', onePointSeq)

if __name__ == '__main__':
    #processRawData(gazePath='../dataset/MIT1003/MIT1003.xlsx', saveFilePath='../dataset/MIT1003/processedData')
    #processRawData()
    '''processRawData_joint(gazePath='../dataset/MIT1003/MIT1003.xlsx',
                   saveFilePath='../dataset/MIT1003/processedData_joint',
                   stimuliPath='../dataset/MIT1003/ALLSTIMULI/')'''
    processRawData(padding=True, useVITFeature=True, gazePath='../dataset/MIT1003/MIT1003.xlsx',
                         saveFilePath='../dataset/MIT1003/processedData_ori',
                         stimuliPath='../dataset/MIT1003/ALLSTIMULI/')
    # COMMENT: padding can pad all the images to square
    # useVITFeature: whether to use VIT pretrained feature extractor, if use no pretrained version pls set to False
    # padding is NOT used if useVITFeature=False
