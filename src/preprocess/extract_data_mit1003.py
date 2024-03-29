import numpy as np
import pickle
import pandas as pd
import cv2
from tqdm import tqdm
import math

def processRawData(resizeFactor, gazePath, stimuliPath, saveFilePath, N=4):
    gazesExcel = pd.read_excel(gazePath)
    oneEntry = {'sub': None, 'imagePath': None, 'scanpath': [], 'imageSize': None,
                'patchIndex': None, 'scanpathInPatch': []}
    processed_dataset = []
    #negativeValue = False
    totalPoints = 0
    negativePoints = 0
    dataEntry = 0
    numOfRows = len(gazesExcel)
    allImages = {}
    onePointSeq = 0
    #for i, row in tqdm(gazesExcel.iterrows()):
    for i in tqdm(range(numOfRows)):
        row = gazesExcel.loc[i]
        subject = row['Sub']
        task = row['Task']
        index = row['T']
        x_coor = row['X'] / resizeFactor  # shape[1]
        y_coor = row['Y'] / resizeFactor  # shape[0]

        # save the current entry and start a new one
        if index == 1 and i != 0: #and not negativeValue:
            assert oneEntry['sub'] is not None
            assert oneEntry['imagePath'] is not None
            assert oneEntry['imageSize'] is not None
            #assert oneEntry['imageFeature'] is not None

            if len(oneEntry['scanpath']) <= 1:
                onePointSeq += 1
            else:
                oneEntry['scanpathInPatch'] = np.stack(oneEntry['scanpathInPatch'])
                processed_dataset.append(oneEntry)
            dataEntry += 1
            oneEntry = {'sub': None, 'imagePath': None, 'scanpath': [], 'imageSize': None,
                        'patchIndex': None, 'scanpathInPatch': []}

        # check for negative values
        '''if index == 1:
            negativeValue = False
        if negativeValue:
            continue
        if x_coor < 0 or y_coor < 0:
            negativeValue = True
            oneEntry = {'sub': None, 'imagePath': None, 'scanpath': [], 'imageFeature': None}
            continue'''

        imagePath = stimuliPath + task #+ '.jpeg'
        if index == 1:
            oneEntry['sub'] = subject
            oneEntry['imagePath'] = task #imagePath

            # process image feature
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (int(image.shape[1]/resizeFactor), int(image.shape[0]/resizeFactor)))
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)
            imageH = image.shape[0]
            imageW = image.shape[1]
            oneEntry['imageSize'] = [imageH, imageW]
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
            patches = np.stack(np.split(image, patchH, axis=0))  # 96, 4, 512, 3
            patches = np.stack(np.split(patches, patchW, axis=2))  # 128, 96, 4, 4, 3
            patches = patches.reshape(patchW, patchH, -1, 3).transpose(2,1,0,3)  # 16, 96, 128, 3
            # indexs are always the same
            '''indexs = np.unravel_index(np.arange(N*N), (N, N))  # size: 16, 2
            indexs = np.concatenate((indexs[0].reshape(1, -1), indexs[1].reshape(1, -1)), axis=0)
            oneEntry['patchIndex'] = indexs
            oneEntry['imageFeature'] = patches'''
            if task not in allImages:
                allImages[task] = patches
        else:
            assert oneEntry['imagePath'] == task
            assert oneEntry['sub'] == subject
            assert oneEntry['imageSize'] == [imageH, imageW]
        if x_coor > 0 and y_coor > 0 and x_coor < imageW and y_coor < imageH:
            oneEntry['scanpath'].append([y_coor, x_coor])
            assert math.floor(y_coor / patchH) < N
            assert math.floor(x_coor / patchW) < N

            before = np.array([math.floor(y_coor / patchH), math.floor(x_coor / patchW)])
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
    print('# one-point/zero-point gaze seq:', onePointSeq)


def processRawDataCenterMode(resizeFactor, gazePath, stimuliPath, saveFilePath,
                             centerModeFilePath):
    file = open(centerModeFilePath, mode='r')
    centerModeIndex = file.read()
    file.close()
    list1 = centerModeIndex.split('\n')[1:]
    centerModeIndexDict = {}
    for a in list1:
        x = a.split(',')
        index = int(x[0])
        x_range = [int(x[1][1]), int(x[1][3])]
        y_range = [int(x[2][1]), int(x[2][3])]
        coor = [int(x[3][1]), int(x[3][3])]
        centerModeIndexDict[index] = {'x_range': x_range, 'y_range': y_range, 'coor': coor}
    N = 4
    gazesExcel = pd.read_excel(gazePath)
    oneEntry = {'sub': None, 'imagePath': None, 'scanpath': [], 'imageSize': None,
                'patchIndex': None, 'scanpathInPatch': []}
    processed_dataset = []
    #negativeValue = False
    totalPoints = 0
    negativePoints = 0
    dataEntry = 0
    numOfRows = len(gazesExcel)
    allImages = {}
    onePointSeq = 0
    #for i, row in tqdm(gazesExcel.iterrows()):
    for i in tqdm(range(numOfRows)):
        row = gazesExcel.loc[i]
        subject = row['Sub']
        task = row['Task']
        index = row['T']
        x_coor = row['X'] #/ resizeFactor  # shape[1]
        y_coor = row['Y'] #/ resizeFactor  # shape[0]

        # save the current entry and start a new one
        if index == 1 and i != 0: #and not negativeValue:
            assert oneEntry['sub'] is not None
            assert oneEntry['imagePath'] is not None
            assert oneEntry['imageSize'] is not None
            #assert oneEntry['imageFeature'] is not None

            if len(oneEntry['scanpath']) <= 1:
                onePointSeq += 1
            else:
                oneEntry['scanpathInPatch'] = np.stack(oneEntry['scanpathInPatch'])
                processed_dataset.append(oneEntry)
            dataEntry += 1
            oneEntry = {'sub': None, 'imagePath': None, 'scanpath': [], 'imageSize': None,
                        'patchIndex': None, 'scanpathInPatch': []}

        imagePath = stimuliPath + task #+ '.jpeg'
        if index == 1:
            oneEntry['sub'] = subject
            oneEntry['imagePath'] = task #imagePath

            if task not in allImages:
                # process image feature
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #image = cv2.resize(image, (int(image.shape[1]/resizeFactor), int(image.shape[0]/resizeFactor)))
                image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                   dtype=cv2.CV_32F)
                imageH = image.shape[0]
                imageW = image.shape[1]
                oneEntry['imageSize'] = [imageH, imageW]
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
                patches = np.stack(np.split(image, patchH, axis=0))  # 96, 4, 512, 3
                patches = np.stack(np.split(patches, patchW, axis=2))  # 128, 96, 4, 4, 3
                patches = patches.reshape(patchW, patchH, -1, 3).transpose(2,1,0,3)  # 16, 96, 128, 3

                # further partition:
                furtherPartitionIndex = [5, 6, 9, 10]
                resized_patches = []
                for i in range(16):
                    patch_entry = patches[i]
                    if i not in furtherPartitionIndex:
                        patch_entry = cv2.resize(patch_entry, (int(patch_entry.shape[1] / resizeFactor), int(patch_entry.shape[0] / resizeFactor)))
                        resized_patches.append(patch_entry)
                    else:
                        patchH_ = int(patch_entry.shape[0] / 2)
                        patchW_ = int(patch_entry.shape[1] / 2)
                        patch_entry = patch_entry[:(patchH_*2), :(patchW_*2)]
                        patch_entry = np.stack(np.split(patch_entry, patchH_, axis=0))  # 96, 4, 512, 3
                        patch_entry = np.stack(np.split(patch_entry, patchW_, axis=2))
                        patch_entry = patch_entry.reshape(patchW_, patchH_, -1, 3).transpose(2, 1, 0, 3)
                        resized_patches.extend(patch_entry)
                resized_patches = np.stack(resized_patches)
                allImages[task] = resized_patches
        else:
            assert oneEntry['imagePath'] == task
            assert oneEntry['sub'] == subject
            assert oneEntry['imageSize'] == [imageH, imageW]
        if x_coor > 0 and y_coor > 0 and x_coor < imageW and y_coor < imageH:
            oneEntry['scanpath'].append([y_coor, x_coor])
            # y/x_coor for indexing array
            y_coor1 = x_coor / image.shape[1] * 8
            x_coor1 = y_coor / image.shape[0] * 8

            for key in centerModeIndexDict:
                entry = centerModeIndexDict[key]
                x_range = entry['x_range']
                y_range = entry['y_range']
                if y_coor1 < y_range[1] and y_coor1 > y_range[0] and x_coor1 < x_range[1] and x_coor1 > x_range[0]:
                    pos = key
                    break
            oneEntry['scanpathInPatch'].append(pos) # pos must exist after for loop
        else:
            negativePoints += 1
        totalPoints += 1

    #if not negativeValue:
    assert oneEntry['sub'] is not None
    assert oneEntry['imagePath'] is not None
    assert oneEntry['imageSize'] is not None
    #assert oneEntry['imageFeature'] is not None
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
    print('# one-point/zero-point gaze seq:', onePointSeq)

if __name__ == '__main__':
    # Resize Factor: 2 for MIT1003, 1 for Toronto
    #processRawData(gazePath='../dataset/MIT1003/MIT1003.xlsx', saveFilePath='../dataset/MIT1003/processedData')
    #processRawData()
    processRawDataCenterMode(gazePath='../dataset/MIT1003/MIT1003.xlsx',
                   stimuliPath='../dataset/MIT1003/ALLSTIMULI/',
                   saveFilePath='../dataset/MIT1003/processedData_N4_centerMode',
                   centerModeFilePath='../dataset/MIT1003/centerModeIndex.txt',
                             resizeFactor=2)
    '''processRawDataCenterMode(gazePath='../dataset/Toronto/Toronto.xlsx',
                             stimuliPath='../dataset/Toronto/Images/',
                             saveFilePath='../dataset/Toronto/processedData_N4_centerMode',
                             centerModeFilePath='../dataset/MIT1003/centerModeIndex.txt',
                             resizeFactor=2)'''
    '''processRawData(gazePath='../dataset/Toronto/Toronto.xlsx',
                   saveFilePath='../dataset/Toronto/processedData_N6',
                   stimuliPath='../dataset/Toronto/Images/',
                   resizeFactor=1, N=6)'''