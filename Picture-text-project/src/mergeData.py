import json
import numpy as np
import webcolors
import random



def readJson(filePath: str) -> list:
    with open(filePath) as f:
        data = json.load(f)
    return data


def saySize(size: list) -> str:
    if (sum(size) > 1):
        return "big "
    elif (sum(size) < 0.4):
        return "small "
    else:
        return ""

def addObjects(input: list, output: list[str]) -> list[str]:
    for object in input["objects"]:
        size = saySize(object["size"])
        objectName = object["class"]
        output.append(size + objectName)

    return output
        


def getColorName(color: list):
    r, g, b = color

    try:
        return webcolors.rgb_to_name((r, g, b))
    except ValueError:
        closest_name = None
        min_diff = float('inf')
        for hex_value, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
            diff = (r - r_c)**2 + (g - g_c)**2 + (b - b_c)**2
            if diff < min_diff:
                min_diff = diff
                closest_name = name
        return closest_name
    


def addColors(input: list, output: list[str]) -> list[str]:
    brightness = ""
    mainColor = list(input["commonColors"][0])
    
    for color in input["commonColors"][1:]:
        mainColor = [mainColor[0] + color[0], mainColor[1] + color[1], mainColor[2] + color[2]]

    mainColor = list(map(lambda number: number // len(input["commonColors"]), mainColor))

    mainColorName = getColorName(mainColor)

    if (input["averageBrightess"] > 180):
        brightness = "bright"
    elif (input["averageBrightess"] < 75):
        brightness = "dark"

    output.append(mainColorName)
    if (brightness != ""): output.append(brightness)

    return output


def saveCsv(trainData: list[tuple[list, str]], savePath) -> None:
    np.savetxt(savePath, trainData, delimiter=";", fmt='%s')

    


trainData: list[tuple[list, str]] = []

data = readJson("photos.json")



for record in data:

    keyWords = [*record["labels"]]
    keyWords = addObjects(record, keyWords)
    keyWords = addColors(record, keyWords)

    for description in record["descriptions"]:
        trainData.append((" ".join(keyWords), description))


random.shuffle(trainData)

saveCsv(trainData, "train.csv")
