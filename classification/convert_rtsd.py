from glob import glob
import argparse
from collections import defaultdict
import tqdm
import cv2
import os
import numpy as np
import sys
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='images')
    parser.add_argument('-a', '--annotations', dest='annotations')
    parser.add_argument('-o', '--output', dest='out_folder')

    args = parser.parse_args()

    output = []

    for annotation_file_name in tqdm.tqdm(os.listdir(args.annotations)):
        with open(os.path.join(args.annotations, annotation_file_name)) as f:
            annotations = f.readlines()

        annotations = annotations[1:]  # skip header
        for elem in annotations:
            obj_class, x_top_left, y_top_left, x_bottom_right, y_bottom_right, ext = elem.split()
            x_top_left = int(x_top_left)
            y_top_left = int(y_top_left)
            x_bottom_right = int(x_bottom_right)
            y_bottom_right = int(y_bottom_right)
            path_to_image = os.path.join(args.images,
                    os.path.splitext(annotation_file_name)[0] + os.path.extsep + ext)
            output.append({'image': os.path.abspath(path_to_image), 'class': obj_class,
                           "x0": x_top_left, 'x1': x_bottom_right, 'y0': y_top_left,
                           'y1': y_bottom_right, 'is_temporary': -1})

    with open(args.out_folder, 'w') as f:
        json.dump(output, f, indent=2)
