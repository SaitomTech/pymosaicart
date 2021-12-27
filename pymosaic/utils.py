from glob import glob
import os
from pathlib import Path
from typing import Union, List

def get_image_paths(dir_path: Union[Path, str]) -> List[Path]:
    """ディレクトリ内の全画像ファイルPathを取得

    Args:
        dir_path: ディレクトリのパス
    """
    # TODO: 画像以外のPATHも取得しているので直す
    return glob(os.path.join(dir_path, "*"))
