import torch.utils.data as data
from os.path import join
from os import listdir
from torchvision.transforms import Compose, ToTensor,RandomCrop
from PIL import Image
import numpy as np


def img_modcrop(image, modulo):
    sz = image.size
    w = np.int32(sz[0] / modulo) * modulo
    h = np.int32(sz[1] / modulo) * modulo
    out = image.crop((0, 0, w, h))
    return out


def np2tensor():
    return Compose([
        ToTensor(),
    ])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg"])


def load_image(filepath):
    return Image.open(filepath).convert('RGB')


class DatasetFromFolderVal(data.Dataset):
    def __init__(self, hr_dir, lr_dir, upscale):
        super(DatasetFromFolderVal, self).__init__()
        self.hr_image_path = hr_dir
        self.lr_image_path = lr_dir
        self.upscale = upscale

    def __getitem__(self, index):
        input_image = load_image(self.lr_image_path)
        target_image = load_image(self.hr_image_path)

        # 转换为张量
        input_tensor = np2tensor()(input_image)
        target_tensor = np2tensor()(img_modcrop(target_image, self.upscale))

        return input_tensor

    def __len__(self):
        return len(self.lr_filenames)
