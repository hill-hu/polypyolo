import os

from ultralytics import utils


def stats_matrix(matrix, names, label_names, total):
    print("model names:", names)
    print("dataset names:", label_names)
    stats = []

    for i in range(len(names)):
        for j in range(len(names)):
            num = int(matrix[i][j])
            for k in range(num):
                label_cls = str(label_names[i])
                predict = str(names[j])
                diff, precision = cal_diff(label_cls, predict)

                stat = {"diff": diff, "precision": precision,
                        "label": label_cls, "predict": predict}
                stats.append(stat)
    miss_count = total - len(stats)
    print("miss count: ", miss_count, ",miss rate: ", miss_count / total)
    step, _ = cal_diff(str(label_names[1]), str(label_names[0]))
    print("step:", step)
    for i in range(0, 3):
        top_diff = len([stat for stat in stats if stat["diff"] <= i * step])
        print(f"diff <={i}:", top_diff, f"/{total} ,rate:{top_diff / total}")

    for p in [0.7, 0.8, 0.9, 1.0]:
        top_p = len([stat for stat in stats if p <= stat["precision"] <= 1])
        print(f"precision >={p}:", top_p, f"/{total} ,rate:{top_p / total}")
    # print(stats)


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
    with open(val_path) as file:
        img_files += file.readlines()
    # load test data
    img_files = [line.replace('\n', "") for line in img_files]
    labels = []
    for line in img_files:
        label_file = line.replace("images", "labels").replace(".jpg", ".txt")
        with open(label_file) as file:
            labels += file.readlines()
    labels = [int(label.split(" ")[0]) for label in labels]
    print(f"load {len(img_files)} data from {data},labels={len(labels)},names={label_names}")
    return label_names, labels, img_files
