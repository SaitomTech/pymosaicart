import os
from pathlib import Path
from typing import Union

import numpy as np
import skimage.io

RESIZE_RATIO = 0.1

class Image:
    def __init__(self, input_path: Union[Path, str]):
        self.data = skimage.io.imread(input_path)
    
    def resize(self):
        """画像データのリサイズ"""
        resize_image = skimage.transform.resize(
            self.data, 
            [self.data.shape[0] * RESIZE_RATIO, self.data.shape[1] * RESIZE_RATIO, 3]
        )
        self.data = (resize_image * 255).astype(np.uint8) 

    def save(self, output_dir: Union[Path, str], file_name: str) -> None:
        skimage.io.imsave(os.path.join(output_dir, file_name), self.data)
