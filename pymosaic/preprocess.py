from argparse import ArgumentParser
from glob import glob
import os
from pathlib import Path
from typing import Union, List

import numpy as np
import skimage.io
from tqdm import tqdm

def get_image_paths(dir_path: Union[Path, str]) -> List[Path]:
    """ディレクトリ内の全画像ファイルPathを取得

    Args:
        dir_path: ディレクトリのパス
    """
    # TODO: 画像以外のPATHも取得しているので直す
    return glob(os.path.join(dir_path, "*"))

def resize_image(image: np.ndarray , ratio: float) -> np.ndarray:
    """画像データのリサイズ

    Args:
        image: (hight, width, RGB) 形式の画像データ 
        ratio: 拡大/縮小率 

    Returns:
        処理後の画像データ 
    """
    resize_image = skimage.transform.resize(
        image, 
        [image.shape[0] * ratio, image.shape[1] * ratio, 3]
    )
    return (resize_image * 255).astype(np.uint8) 
    
def save_image(image: np.ndarray, output_dir: Union[Path, str], file_name: str) -> None:
    skimage.io.imsave(os.path.join(output_dir, file_name), image)

def get_args():
    parser = ArgumentParser(description="入出力パス")
    parser.add_argument('-i', '--input_dir', type=str, help='input directory path')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory path')
    return parser.parse_args()

def preprocess():
    args = get_args() 
    image_paths = get_image_paths(args.input_dir)
    for i, path in enumerate(tqdm(image_paths)):
        resized_image = resize_image(skimage.io.imread(path), ratio=0.1)
        file_name = f'image_{i}.jpeg'
        save_image(resized_image, args.output_dir, file_name)
    
if __name__ == "__main__":
    preprocess()
