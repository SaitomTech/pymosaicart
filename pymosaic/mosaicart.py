from argparse import ArgumentParser
import itertools
from operator import itemgetter
import random

import numpy as np
import skimage.io

from image import Image
from utils import get_image_paths


def get_block_images(block_hight, block_width, folder_path):
    sub_images = [Image(skimage.io.imread(path)) for path in get_image_paths(folder_path)]
    for sub_image in sub_images:
        sub_image.resize_with_trim(block_hight, block_width)
    return sub_images 

def create_unified_image(separated_images, block_images, random_rate):
    def _get_similar_block_image_idx(image_target_lab, sub_image_labs):
        min_diff = float('inf')
        min_idx = 0
        for i, sub_image_lab in enumerate(sub_image_labs):
            diff = np.sum(skimage.color.deltaE_cie76(image_target_lab, sub_image_lab))
            if diff < min_diff:
                min_diff = diff
                min_idx = i
        return min_idx
    
    def _uniform_color_images(convert_image, target_image):
        uniformed_color_image = Image(convert_image.data * (target_image.rgb_mean / convert_image.rgb_mean))
        uniformed_color_image.min_max_scaling() 
        return uniformed_color_image

    def _replace_image(col, row, width, hight, replace_target, replace_source):
        x_i = col * width
        x_f = (col + 1) * width
        y_i = row * hight
        y_f = (row + 1) * hight
        replace_target[y_i:y_f, x_i:x_f] = replace_source.data
        return replace_target

    block_images_labs = [block_image.small_lab for block_image in block_images]
    n_vertical_block, n_horizon_block = np.shape(separated_images)
    hight_block, width_block, _ = separated_images[0][0].data.shape
    
    image_output = np.zeros((n_vertical_block * hight_block, n_horizon_block * width_block, 3))
    for row, col in itertools.product(range(n_vertical_block), range(n_horizon_block)):
        separated_image = separated_images[row][col]
        
        # Select block image without duplicating images
        sample_indexes = random.sample(range(len(block_images_labs)), 
                                       int(len(block_images_labs) * random_rate))
        block_images_sampled = itemgetter(*sample_indexes)(block_images)
        block_images_labs_sampled = itemgetter(*sample_indexes)(block_images_labs)
        min_diff_idx = _get_similar_block_image_idx(separated_image.small_lab,
                                                    block_images_labs_sampled)
        block_image = block_images_sampled[min_diff_idx] 

        # uniform color average
        uniformed_color_image = _uniform_color_images(block_image, separated_image)

        # replace
        image_output = _replace_image(col, row, width_block, hight_block, 
                                      image_output, uniformed_color_image)
    return Image(image_output)


def get_args():
    parser = ArgumentParser(description="入出力パス")
    parser.add_argument('-m', '--main_image_path', type=str, help='main image path')
    parser.add_argument('-i', '--input_directory', type=str, help='input sub-images directory path')
    parser.add_argument('-o', '--output_image_path', type=str, help='output image path')
    parser.add_argument('-b', '--blocks', type=int, help='number of blocks')
    parser.add_argument('-r', '--magnification_rate', type=int, help='magnification rate')
    return parser.parse_args()

def create_mosaic_art():
    args = get_args()
    main_image = Image(skimage.io.imread(args.main_image_path))
    separated_images = main_image.separate(n_horizon_block=args.blocks,
                                           magnification_rate=args.magnification_rate)
    block_hight, block_width, _ = separated_images[0][0].data.shape
    block_images = get_block_images(block_hight, block_width, args.input_directory)

    output_image = create_unified_image(separated_images, block_images, random_rate=0.05)
    output_image.save(args.output_image_path, 'mosaic_art.jpeg')

if __name__ == "__main__":
    create_mosaic_art()
