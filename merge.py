import os
import random
import time
import csv

from datetime import datetime

COMMA_DELIMITER = ",";
SEPERATER = "\\";

RAW_FOLDER = "";
NEW_FOLDER = "";

WRIST = "wrist34";
THIGH = "thigh34";

activityType = ["jogging", "sitting", "standing", "walkfast", "walkmod", "walkslow", "upstairs", "downstairs", "lying"]
idList = range(len(activityType))
activityIdDict = dict(zip(activityType, idList))

headers = ['time', 'x', 'y', 'z', 'class']


def read(folder):
    data = {}
    folderName = RAW_FOLDER + SEPERATER + folder;
    for f in os.listdir(folderName):
        newf = os.path.join(folderName, f)
        if os.path.isdir(newf):
            users = {}
            for ff in os.listdir(newf):
                if ff == '.DS_Store':
                    continue
                name = ff.replace(".csv", "")
                newff = os.path.join(newf, ff)
                users[name] = newff
            data[f] = users
    return data


def merge(data_wrist, data_thigh):
    data = {}
    for wristActivity in data_wrist:
        activity = wristActivity
        thigh_users = data_thigh[wristActivity]
        wrist_users = data_wrist[wristActivity]
        for wristUser in wrist_users:
            user = wristUser
            thighData = thigh_users[user]
            wristData = wrist_users[user]

            thighCSV = readFile(thighData)
            wristCSV = readFile(wristData)

            merged = mergeFiles(wristCSV, thighCSV)
            key = user + "_" + activity;

            data[key] = merged;
    return data


def writeData(data):
    keys = list(data.keys())
    random.shuffle(keys)

    for key in keys:
        activity = key.split('_')[1]
        activityClass = activityIdDict[activity]
        file = NEW_FOLDER + SEPERATER + key + '.txt'
        count = 0;
        value = data[key]
        writeFile(file, value, activityClass)


def writeFile(file, data, activity):
    file = open(file, 'w+')
    for new in data:
        file.write(",".join(new[:6]))
        file.write("\n")


def readFile(fileFullPath):
    list = []
    with open(fileFullPath, 'r') as csvfile:
        i = True
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if i:
                i = False
                continue
            list.append(row[:4])
    return list


def mergeFiles(wristContent, thighContent):
    returnData = []
    length = min(len(wristContent), len(thighContent))
    startj = 0;
    matchDistance = 0;
    for i in range(length):
        wristTimestamp = wristContent[i];
        utc_time1 = datetime.strptime(wristTimestamp[0],
                                      "%Y-%m-%d %H:%M:%S.%f")
        wristTime = time.mktime(utc_time1.timetuple()) * 1000 + int(utc_time1.microsecond / 1000)
        endj = min(i + matchDistance + 3, length);
        for j in range(startj, endj):
            # System.out.println("i-"+i+" j-"+j+" endj-"+endj+" matchDistance-"+matchDistance);
            thighTimestamp = thighContent[j];
            utc_time2 = datetime.strptime(thighTimestamp[0],
                                          "%Y-%m-%d %H:%M:%S.%f")
            thighTime = time.mktime(utc_time2.timetuple()) * 1000 + int(utc_time2.microsecond / 1000)
            if wristTime > thighTime:
                if wristTime - thighTime <= 5:
                    timestamp = []
                    timestamp.append(str(wristTimestamp[1]))
                    timestamp.append(str(wristTimestamp[2]))
                    timestamp.append(str(wristTimestamp[3]))
                    timestamp.append(str(thighTimestamp[1]))
                    timestamp.append(str(thighTimestamp[2]))
                    timestamp.append(str(thighTimestamp[3]))
                    timestamp.append(str(wristTime))
                    returnData.append(timestamp)
                    startj = j + 1
                    matchDistance = abs(i - j)
                    # print("yes")
                    break
                else:
                    # print("continue")
                    continue

            if thighTime > wristTime:
                if thighTime - wristTime <= 5:
                    timestamp = []
                    timestamp.append(str(wristTimestamp[1]))
                    timestamp.append(str(wristTimestamp[2]))
                    timestamp.append(str(wristTimestamp[3]))
                    timestamp.append(str(thighTimestamp[1]))
                    timestamp.append(str(thighTimestamp[2]))
                    timestamp.append(str(thighTimestamp[3]))
                    timestamp.append(str(wristTime))
                    returnData.append(timestamp)
                    startj = j + 1
                    matchDistance = abs(i - j)
                    # print("yes")
                    break
                else:
                    # print("continue")
                    continue
            else:
                timestamp = []
                timestamp.append(str(wristTimestamp[1]))
                timestamp.append(str(wristTimestamp[2]))
                timestamp.append(str(wristTimestamp[3]))
                timestamp.append(str(thighTimestamp[1]))
                timestamp.append(str(thighTimestamp[2]))
                timestamp.append(str(thighTimestamp[3]))
                timestamp.append(str(wristTime))
                returnData.append(timestamp)
                startj = j + 1
                matchDistance = abs(i - j)
                # print("yes")
                break
    return returnData


dataWrist = read(WRIST)
dataThigh = read(THIGH)

data = merge(dataWrist, dataThigh)
writeData(data)
