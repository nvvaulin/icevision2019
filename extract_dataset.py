import argparse
import tarfile
from pathlib import Path

from tqdm.auto import tqdm

from flif_utils import convert_flifs


def main(run_dir: Path):
    tars = ['left.tar', 'right.tar']
    tars = [run_dir / tar for tar in tars if (run_dir / tar).exists()]
    for tar_path in tqdm(tars, postfix='tars'):
        src = run_dir / tar_path.stem
        src.mkdir(exist_ok=True)
        with tarfile.open(str(tar_path)) as tar:
            tar.extractall(str(src))
        dst = run_dir / (tar_path.stem + '_pngs')
        convert_flifs(src, dst, use_tqdm=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path,
                        help='Path to a directory within the downloaded IceVision dataset (with {left,right}.tar inside)')
    args = parser.parse_args()

    main(args.run_dir)
