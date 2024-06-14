from ultralytics import YOLO

from colorthief import ColorThief
import cv2
import os
import tensorflow as tf
import json
import string

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
from PIL import Image


def learn_and_valudate_model(model: YOLO, epochs: int) -> None:
    results = model.train(data="coco8.yaml", epochs=epochs)
    results = model.val()

    print(results)


def add_color_info(pathlike: str, currObject: dict) -> dict:
    color_thief = ColorThief(pathlike)
    palette = color_thief.get_palette(color_count=6, quality=1)

    image = cv2.imread(pathlike)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    value_channel = hsv_image[:,:,2]
    average_brightness = value_channel.mean()

    currObject["commonColors"] = palette
    currObject["averageBrightess"] = round(average_brightness, 2)

    return currObject


def tensorConversion(*tensorArgs: tf.Tensor) -> float:
    if (len(tensorArgs) == 1): return round(float(tensorArgs[0]), 4)
    else: return list(map(lambda tensor: round(float(tensor), 4), tensorArgs))


def saveData(data: dict, jsonName: str) -> None:
    with open(jsonName, 'w') as file:
        json.dump(data, file, indent=4)


def createCaptionsDict(captionsPath: str) -> dict:
    result = {}
    specialCharacters = string.punctuation
    translation = str.maketrans('', '', specialCharacters)

    with open(captionsPath, 'r') as file:
        captionsFile = file.readlines()
        
    for line in captionsFile[1:]:
        textSplit = line.split(',')
        
        fileName = textSplit[0]
        description = textSplit[1].translate(translation)[:-1].strip()

        if (fileName in result):
            result[fileName].append(description)
        else:
            result[fileName] = [description]

    return result


def getImageLabels(model: VGG16, imagePath: str) -> list:
    img = Image.open(imagePath)

    img = image.load_img(imagePath, target_size=img.size)
    img = img.resize((224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)

    results = decode_predictions(predictions, top=10)[0]

    results = [label for (_, label, propability) in results if propability > 0.15]

    return results



DATASET_PATH = "datasets/flickr/Images/"
CAPTIONS_PATH = "datasets/flickr/captions.txt"


photosPaths = list(map(lambda path: "".join([DATASET_PATH, path]), os.listdir(DATASET_PATH)))
allDescriptions = createCaptionsDict(CAPTIONS_PATH)

model = YOLO('yolov8l.pt')
modelVGG = VGG16(weights='imagenet')

# learn_and_valudate_model(model, 5)


output = []
skip = 75

# for currSkip in range(len(photosPaths) // skip):
for currSkip in range(1):
    
    startIndex = currSkip * skip
    endIndex = (currSkip + 1) * skip
    results = model(photosPaths[startIndex:endIndex])
    allClasses = results[0].names
    numberOfIter = 0


    for result in results:
        currRecord = {}

        fileName = result.path.split('/')[-1]

        currRecord["name"] = fileName
        currRecord["descriptions"] = allDescriptions[fileName]
        currRecord["labels"] = getImageLabels(modelVGG, f"{DATASET_PATH}{fileName}")
        currRecord["objects"] = []

        for box in result.boxes:
            currObject = {}
            currObject["class"] = allClasses[tensorConversion(box.cls[0])]
            currObject["conf"] = tensorConversion(box.conf[0])
            currObject["startingPoint"] = tensorConversion((box.xywhn[0][0]), (box.xywhn[0][1]))
            currObject["size"] = tensorConversion(box.xywhn[0][2], box.xywhn[0][3])
            currRecord["objects"].append(currObject)

        currRecord = add_color_info(result.path, currRecord)

        output.append(currRecord)

        numberOfIter += 1
        print(numberOfIter)
        
saveData(output, "photos.json")