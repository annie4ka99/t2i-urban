import random
import numpy as np
from PIL import Image


def resize_and_crop(image):
    crop_size = 512
    width, height = image.size
    new_width, new_height = width, height
    if height >= width != crop_size:
        new_width = crop_size
        new_height = int((new_width / width) * height)
        image = image.resize((new_width, new_height), Image.BICUBIC)
    elif width >= height != crop_size:
        new_height = crop_size
        new_width = int((new_height / height) * width)
        image = image.resize((new_width, new_height), Image.BICUBIC)

    # crop
    crop_x = random.randint(0, np.maximum(0, new_width - crop_size))
    crop_y = random.randint(0, np.maximum(0, new_height - crop_size))
    image = image.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))

    return image
