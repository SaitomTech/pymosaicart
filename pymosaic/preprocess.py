from argparse import ArgumentParser

from tqdm import tqdm

from image import Image
from utils import get_image_paths

def get_args():
    parser = ArgumentParser(description="入出力パス")
    parser.add_argument('-i', '--input_dir', type=str, help='input directory path')
    parser.add_argument('-o', '--output_dir', type=str, help='output directory path')
    return parser.parse_args()

def preprocess():
    args = get_args() 
    image_paths = get_image_paths(args.input_dir)
    for i, path in enumerate(tqdm(image_paths)):
        image = Image(path)
        image.resize() 
        image.save(args.output_dir, f'image_{i}.jpeg')
    
if __name__ == "__main__":
    preprocess()
