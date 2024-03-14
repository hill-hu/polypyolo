import json
import os

from ultralytics import utils
import traceback
import numpy as np


def read_img_file(path):
    try:
        with open(path, 'rb') as img_file:
            bytes = img_file.read()
            nparr = np.frombuffer(bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
    except:
        traceback.print_exc()
    return None


def save_img_file(path, img):
    try:
        with open(path, 'wb') as img_file:
            data = cv2.imencode('.jpg', img)[1]
            img_file.write(data)
    except:
        traceback.print_exc()
    return None


def stats_matrix(matrix, names, label_names, total):
    print("model names:", names)
    print("dataset names:", label_names)
    stats = []

    for i in range(len(names)):
        for j in range(len(names)):
            num = int(matrix[i][j])
            for k in range(num):
                stat = {"label": int(i), "predict": int(j)}
                stats.append(stat)

    # print(stats)
    stats_result(stats, names, label_names, total)


def stats_result(stats, names, label_names, total, save_dir=None):
    print("model names:", names)
    print("dataset names:", label_names)

    miss_count = 0
    for i in range(len(stats)):
        stat = stats[i]

        label_cls = str(label_names[stat['label']])
        if stat['predict'] < 0:
            miss_count += 1
            predict_cls = None
        else:
            predict_cls = str(names[stat['predict']])
        diff, precision = cal_diff(label_cls, predict_cls)

        stat.update({"diff": diff, "precision": precision,
                     "label_cls": label_cls, "predict_cls": predict_cls})

    print("miss count: ", miss_count, ",miss rate: ", miss_count / total)
    step, _ = cal_diff(str(label_names[1]), str(label_names[0]))
    print("step:", step)
    for i in range(0, 3):
        diff_stats = [stat for stat in stats if stat["diff"] <= i * step]
        diff_count = len(diff_stats)
        print(f"diff <={i}:", diff_count, f"/{total} ,rate:{diff_count / total}")

    if save_dir:
        diff_stats = [stat for stat in stats if stat["diff"] > 2 * step]
        export_files(save_dir, diff_stats, "fail")
        diff_stats = [stat for stat in stats if stat["diff"] <= 2 * step]
        export_files(save_dir, diff_stats, "ok")
    for p in [0.7, 0.8, 0.9, 1.0]:
        top_p = len([stat for stat in stats if p <= stat["precision"] <= 1])
        print(f"precision >={p}:", top_p, f"/{total} ,rate:{top_p / total}")

    return stats


def cal_diff(label, predict):
    # cal  abs diff and precision
    if predict is None:
        return 999, 0
    label = str(label).replace("mm", "").replace(">", "")
    predict = predict.replace("mm", "").replace(">", "")
    diff = int(label) - int(predict)
    return abs(diff), 1 - diff / int(label)


def read_dataset(data):
    #
    img_files = []
    dataset = utils.yaml_load(data)
    label_names = dataset['names']
    val_path = os.path.join(dataset['path'], dataset['val'])
    with open(val_path, encoding='utf-8') as file:
        img_files += file.readlines()
    # load test data
    img_files = [line.replace('\n', "") for line in img_files]
    labels = []
    for line in img_files:
        label_file = line.replace("images", "labels").replace(".jpg", ".txt")
        with open(label_file, encoding='utf-8') as file:
            labels += file.readlines()
    labels = [int(label.split(" ")[0]) for label in labels]
    print(f"load {len(img_files)} data from {data},labels={len(labels)},names={label_names}")
    return label_names, labels, img_files


def stats_json(json_file, labels, img_files, label_names, model_names):
    # stats from  "predictions.json" file
    stats_cache = {}
    for i in range(len(labels)):
        img_id = os.path.split(img_files[i])[1].replace(".jpg", "")
        stat = {"label": labels[i], "predict": -1, "img_file": img_files[i], "image_id": img_id,
                "bbox": [100, 100, 100, 100]}
        stats_cache[img_id] = stat

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
        # {"image_id": "3056_1572", "category_id": 8, "bbox": [536.373, 34.919, 499.416, 665.0], "score": 1.0}
        for i in range(len(data)):
            image_id = data[i]["image_id"]
            stat = stats_cache[image_id]
            if stat['predict'] >= 0:
                continue
            stat['predict'] = data[i]["category_id"]
            stat['bbox'] = data[i]["bbox"]
    stats = list(stats_cache.values())
    save_dir = os.path.split(json_file)[0]
    stats_result(stats, model_names, label_names, len(labels), save_dir)
    return stats


import cv2


def export_files(save_dir, top_diff, tag):
    output_folder = os.path.join(save_dir, "diff_" + tag)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for stat in top_diff:

        bbox = stat['bbox']
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[0]) + int(bbox[2]), int(bbox[1]) + int(bbox[3])
        color = (255, 0, 0)
        bbox_img = read_img_file(stat["img_file"])
        bbox_img = cv2.rectangle(bbox_img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(bbox_img, "P:" + str(stat['predict_cls']), (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.putText(bbox_img, "G:" + str(stat['label_cls']), (xmin, ymax - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        save_img_file(os.path.join(output_folder, stat["image_id"] + ".jpg"), bbox_img)


if __name__ == '__main__':
    data = 'custom\\datasets\\whu_1mm.yaml'
    save_dir = r'E:\medical\depth\polypyolo\runs\detect\train108'
    label_names, labels, img_files = read_dataset(data)
    results = stats_json(os.path.join(save_dir, "predictions.json"),
                         labels, img_files, label_names, label_names)
