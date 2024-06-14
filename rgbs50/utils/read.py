import os

import cv2
import numpy as np


def img_filter(imgs_list: [str, list], extension_filter: str = r'.jpg') -> list:
    """
    Description
        img_filter retains items in the specified format in the input list
    Params:
        extension_filter:   default is '.jpg'
    """
    return list(filter(lambda file: file.endswith(extension_filter), imgs_list))

def imread(filename: str) -> np.array:
    """
    Description
        imread is an easy extension of cv2.imread, which returns RGB images
    """
    try:
        image = cv2.imread(filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        raise print(f'{filename} is wrong')


def seqread(file: str, imgs_type='.jpg'):
    """
    Description
        Seqread reads all image items in the file and sorts them by numerical name
        It returns a list containing the absolute addresses of the images

        Sorting only supports two types, '*/1.jpg' and '*/*_1.jpg'

    Params:
        file:       images' file
        imgs_type:  default is '.jpg'

    Return:
        List of absolute paths of sorted images

    """

    try:
        output_list = sorted(img_filter(os.listdir(file), imgs_type), key=lambda x: int(x.split('.')[-2]))
    except ValueError:
        output_list = sorted(img_filter(os.listdir(file), imgs_type),
                             key=lambda x: int(x.split('.')[-2].split('_')[-1]))

    return [os.path.join(file, item) for item in output_list]


def txtread(filename: str, delimiter: [str, list] = None) -> np.ndarray:
    """
    Description
        txtread is an extension of np.loadtxt, support ',' and '\t' delimiter.
        The original implementation method is in the pytracking library at https://github.com/visionml/pytracking
    """

    if delimiter is None:
        delimiter = [',', '\t']

    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(filename, delimiter=d, dtype=np.float64)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(filename))
    else:
        ground_truth_rect = np.loadtxt(filename, delimiter=delimiter, dtype=np.float64)
        return ground_truth_rect
