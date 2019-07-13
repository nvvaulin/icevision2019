import argparse
from pathlib import Path

import cv2 as cv
import imageio
import numpy as np
import pyflif
from tqdm.auto import tqdm


def rgba2rgb(rgba):
    basic = np.empty((2048, 2448), dtype=np.uint8)

    basic[1::2, 1::2] = rgba[:, :, 0]
    basic[0::2, 1::2] = rgba[:, :, 1]
    basic[0::2, 0::2] = rgba[:, :, 2]
    basic[1::2, 0::2] = rgba[:, :, 3] + rgba[:, :, 1] - 0x80

    rgb = cv.cvtColor(basic, cv.COLOR_BAYER_BG2BGR)

    return rgb


def read_frame(path):
    path = str(path)
    if path.endswith('.flif'):
        rgba = pyflif.read_flif(str(path))
        rgb = rgba2rgb(rgba)
        return rgb
    elif path.endswith('.jpg'):
        return imageio.imread(path)
    else:
        assert False


def convert_flifs(src: Path, dst: Path, use_tqdm=False):
    dst.mkdir(exist_ok=True, parents=True)

    flifs = sorted(src.iterdir())
    if use_tqdm:
        flifs = tqdm(flifs)

    for flif_path in flifs:
        png_path = dst / '{}.png'.format(flif_path.stem)

        if png_path.exists():
            continue

        frame = read_frame(flif_path)
        imageio.imsave(png_path, frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='action')

    convert_parser = subparsers.add_parser('convert')

    convert_parser.add_argument('src', type=Path, help='folder with flifs')
    convert_parser.add_argument('dst', type=Path, help='folder with pngs')
    convert_parser.add_argument('--progress', action='store_true', help='whether to display progress bar')

    args = parser.parse_args()

    convert_flifs(args.src, args.dst, args.progress)
