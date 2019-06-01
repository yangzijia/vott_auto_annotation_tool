'''
create by zjyang on 2019-06-01

Load model, auto-annotate, generate VOTT json file

    First, you need to modify the model load path and classes_name path
    in your yolo.py, Then you need to modify the JSON file path and the 
    picture directory path in the main method. finally, choose the model 
    you need to run the main method.

'''

import os
import json
import random
from yolo import YOLO
from PIL import Image
from tqdm import tqdm


def color(value):
    """ Abandoned method """
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7',
                '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#"+color


def get_color_list(num):
    color_list = []
    for i in range(num):
        color_list.append(randomcolor())
    return color_list


def lable_picture(yolo, param_dict):
    """
        Use traind model to label pictures
    """
    json_path = param_dict["json_path"]
    pictures_path = param_dict["pictures_path"]

    params = {}
    color_list, class_names = yolo.get_color_classes()
    params["classes_list"] = class_names
    for root, dirs, files in os.walk(pictures_path):
        pbar = tqdm(total=len(files))
        frames_dict = {}
        for index, img_name in enumerate(files):
            img_path = root + img_name
            image = Image.open(img_path)
            location = yolo.get_detect_location(image, params)
            frames_dict[str(index)] = location
            pbar.update(1)

        pbar.close()
        class_names = str(class_names).replace(
            "[", "").replace("]", "").replace("\'", "")
        final_dict = {"frames": frames_dict, "framerate": "1", 
                    "inputTags": class_names, "suggestiontype": "track",
                    "scd": "false", "visitedFrames": [i for i in range(int(index)+1)], 
                    "tag_colors": get_color_list(len(color_list))}

        final_str = str(final_dict).replace("\'", "\"")
        with open(json_path, 'a+') as f:
            f.write(final_str)


def add_new_classes(yolo, param_dict):
    """
        Adding new classes to labeled samples
    """
    json_path = param_dict["json_path"]
    pictures_path = param_dict["pictures_path"]
    need_add_class = param_dict["need_add_class_name"]
    params = {}
    # get json content
    with open(json_path, encoding="utf-8") as fr:
        content = json.load(fr)
    # fix json content
    content["inputTags"] = content["inputTags"] + \
        "," + ",".join(need_add_class)
    classes_list = content["inputTags"].split(",")
    content["tag_colors"] = get_color_list(len(classes_list))
    content["scd"] = "false"

    print("changing frames......")
    params["classes_list"] = need_add_class
    frames = content["frames"]
    for root, dirs, files in os.walk(pictures_path):
        pbar = tqdm(total=len(files))
        frames_dict = {}
        for index, img_name in enumerate(files):
            img_path = root + img_name
            image = Image.open(img_path)
            location = yolo.get_detect_location(image, params)

            frame = frames[str(index)]   # frame is a list
            location = location + frame
            name_index = 1
            for a in location:
                a["name"] = name_index
                name_index += 1

            frames_dict[str(index)] = location
            pbar.update(1)

        content["frames"] = frames_dict
        pbar.close()

        new_json_path = json_path.split(".json")[0] + "_new.json"
        final_str = str(content).replace("\'", "\"")
        with open(new_json_path, 'a+') as f:
            f.write(final_str)


if __name__ == "__main__":
    yolo = YOLO()
    param_dict = {}

    param_dict["json_path"] = "c:/Users/zjyan/Desktop/11.json"
    param_dict["pictures_path"] = "C:/Users/zjyan/Desktop/11/"

    # MODEL 1 : Use trained model to label pictures
    lable_picture(yolo, param_dict)

    # MODEL 2 : Adding new classes to labeled samples
    # param_dict["need_add_class_name"] = ["person"]
    # add_new_classes(yolo, param_dict)

    yolo.close_session()
