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
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, str(src))
        dst = run_dir / (tar_path.stem + '_pngs')
        convert_flifs(src, dst, use_tqdm=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path,
                        help='Path to a directory within the downloaded IceVision dataset (with {left,right}.tar inside)')
    args = parser.parse_args()

    main(args.run_dir)
