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

    for annotation_folder in tqdm.tqdm(os.listdir(args.annotations)):
        annotation_files = os.listdir(os.path.join(args.annotations, annotation_folder))
        annotation_files.sort()
        images_files = os.listdir(os.path.join(args.images, annotation_folder))
        images_files.sort()

        print len(images_files), 1
        images_files = filter(lambda x : x != 'index.tsv', images_files)
        print len(images_files), 2

        assert len(images_files) == len(annotation_files), "{} {} {}".format(len(images_files), len(annotation_files), annotation_folder)

        for i in range(len(images_files)):
            with open(os.path.join(args.annotations, annotation_folder, annotation_files[i])) as f:
                annotations = f.readlines()
            annotations = annotations[1:]  # skip header
            for elem in annotations:
                elem = elem.split()
                obj_class, x_top_left, y_top_left, x_bottom_right, y_bottom_right = elem[:5]
                is_temporary = elem[5] == 'true'
                is_occluded = elem[6] == "true"

                data = "" if len(elem) == 7 else elem[7]

                x_top_left = int(x_top_left)
                y_top_left = int(y_top_left)
                x_bottom_right = int(x_bottom_right)
                y_bottom_right = int(y_bottom_right)
                path_to_image = os.path.join(args.images, annotation_folder, images_files[i])
                output.append({'image': os.path.abspath(path_to_image), 'class': obj_class,
                               "x0": x_top_left, 'x1': x_bottom_right, 'y0': y_top_left,
                               'y1': y_bottom_right, 'is_temporary': int(is_temporary),
                               'is_occluded': int(is_occluded),
                               'data': data})

    with open(args.out_folder, 'w') as f:
        json.dump(output, f, indent=2)
