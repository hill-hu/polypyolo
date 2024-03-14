import argparse
import os
import time
from multiprocessing import freeze_support

import stats_holder as stats
from ultralytics import utils
from ultralytics.models.yolo.detect import DetectionValidator


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='model weights path')
    parser.add_argument("--data", type=str, help="data.yaml path ")

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    freeze_support()

    opts = parse_opt()
    print(opts)

    args = dict(model=opts.weights, data=opts.data, workers=2, save_json=True, max_det=1, conf=0.01)
    validator = DetectionValidator(args=args)
    validator()
    matrix = validator.confusion_matrix
    label_names, labels, lines = stats.read_dataset(opts.data)
    # print(matrix.matrix)
    stats.stats_matrix(matrix.matrix, validator.names, label_names, len(lines))

    stats.stats_json(os.path.join(validator.save_dir, "predictions.json"),
                     labels, lines, label_names, validator.names)
