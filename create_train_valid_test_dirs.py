import glob
import os
import random
import math
import shutil
import argparse
import platform

from tqdm import tqdm

def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='Provide arguments.')
    parser.add_argument(
        '--pdata',
        type=str,
        default='../../../datasets/101_ObjectCategories/',
        help='Location of the original data.'
    )
    parser.add_argument(
        '--psave',
        type=str,
        default='./datasets/',
        help='Path where to copy the data.'
    )
    return parser.parse_args()

def create_directories(
    path_save: str,
    directories: list,
    classes: list
    ) -> None:
    """Creates class directories in train, valid and test folders."""
    for dir_ in tqdm(directories):
        path_ = path_save + dir_ + '/'
        for cls_ in tqdm(classes):
            if not os.path.isdir(path_ + cls_):
                os.mkdir(path_ + cls_)

def copy_files_to_directories(
    path_data: str,
    path_save: str,
    classes: list,
    train_size: float = 0.5,
    valid_size: float = 0.25,
    test_size: float = 0.25
    ) -> None:
    """
    """

    plat = platform.platform()
    if 'Windows' in plat:
        plat = 'Windows'
    if 'Linux' in plat:
        plat = 'Linux'

    path_train_save = path_save + 'train/'
    path_valid_save = path_save + 'valid/'
    path_test_save = path_save + 'test/'

    for cls_ in tqdm(classes):

        files = glob.glob(path_data + cls_ + '/*.jpg')
        random.shuffle(files)

        train_i = math.floor(train_size * len(files))
        valid_i = train_i + math.floor(valid_size * len(files))

        train_files = files[:train_i]
        valid_files = files[train_i:valid_i]
        test_files = files[valid_i:]

        for file_ in train_files:

            if plat == 'Windows':
                filename = file_.split('\\')[-1]
            if plat == 'Linux':
                filename = file_.split('/')[-1]

            dst_ = path_train_save + cls_ + '/' + filename
            shutil.copyfile(file_, dst_)

        for file_ in valid_files:

            if plat == 'Windows':
                filename = file_.split('\\')[-1]
            if plat == 'Linux':
                filename = file_.split('/')[-1]

            dst_ = path_valid_save + cls_ + '/' + filename
            shutil.copyfile(file_, dst_)

        for file_ in test_files:

            if plat == 'Windows':
                filename = file_.split('\\')[-1]
            if plat == 'Linux':
                filename = file_.split('/')[-1]

            dst_ = path_test_save + cls_ + '/' +  filename
            shutil.copyfile(file_, dst_)


if __name__ == '__main__':

    args = parse_args()
    path_data = args.pdata
    path_save = args.psave

    classes = os.listdir(path_data)
    directories = os.listdir(path_save)

    create_directories(path_save, directories, classes)
    copy_files_to_directories(path_data, path_save, classes)
