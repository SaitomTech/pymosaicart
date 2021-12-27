import os
from pathlib import Path
from typing import Union

import numpy as np
import skimage.io

SHRINK_RATIO = 0.1

class Image:
    def __init__(self, data: np.ndarray):
        self.data = data 
    
    def save(self, output_dir: Union[Path, str], file_name: str) -> None:
        skimage.io.imsave(os.path.join(output_dir, file_name), self.data)
    
    def shrink(self):
        """画像データの縮小"""
        resize_image = skimage.transform.resize(
            self.data, 
            [self.data.shape[0] * SHRINK_RATIO, self.data.shape[1] * SHRINK_RATIO, 3]
        )
        self.data = (resize_image * 255).astype(np.uint8) 

    def _crop_center(self, hight_target, width_target):
        hight_init, width_init, _ = self.data.shape
        x_i = (width_init - width_target) // 2
        x_f = (width_init + width_target) // 2 
        y_i = (hight_init - hight_target) // 2
        y_f = (hight_init + hight_target) // 2
        self.data = self.data[y_i:y_f, x_i:x_f]

    def resize_with_trim(self, hight_target, width_target):
        """resize & trim center part"""
        hight_init, width_init, _ = self.data.shape
        ratio_init = width_init / hight_init
        ratio_target = width_target / hight_target
        
        if ratio_init == ratio_target:
            self.data = skimage.transform.resize(self.data, (hight_target, width_target, 3),
                                                 anti_aliasing=True)
        elif ratio_init > ratio_target:
            image_resized = Image(
                skimage.transform.resize(self.data, 
                                         (hight_target, np.ceil(hight_target * ratio_init), 3), 
                                         anti_aliasing=True)
            )
            image_resized._crop_center(hight_target, width_target)
            self.data = image_resized.data
        else:
            image_resized =  Image(
                skimage.transform.resize(self.data, 
                                         (np.ceil(width_target / ratio_init), width_target, 3),
                                         anti_aliasing=True)
            )
            image_resized._crop_center(hight_target, width_target)    
            self.data = image_resized.data

    def separate(self, n_horizon_block, magnification_rate, block_aspect=3/4):
        image_expanded = Image(skimage.transform.resize(
            self.data,
            (self.data.shape[0] * magnification_rate, self.data.shape[1] * magnification_rate, 3),
            anti_aliasing=True
        ))
        
        width_block = int(image_expanded.data.shape[1] / n_horizon_block)
        hight_block = int(width_block / block_aspect)    
        n_vertical_block = image_expanded.data.shape[0] // hight_block

        image_expanded._crop_center(hight_block * n_vertical_block, width_block * n_horizon_block)
        image_separated = skimage.util.view_as_blocks(image_expanded.data, (hight_block, width_block, 3))
        return [[Image(image_separated[row][col][0])
                for col in range(n_horizon_block)] 
                for row in range(n_vertical_block)]

    def min_max_scaling(self):
        self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))

    @property
    def small_lab(self):
        return skimage.color.rgb2lab(skimage.transform.resize(self.data, (8, 8, 3), anti_aliasing=True))

    @property
    def rgb_mean(self):
        return np.array([np.mean(self.data[:, :, i]) for i in range(3)])

    
