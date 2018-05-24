import numpy as np


def process_image(image):
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))
    image = np.array(image)
    image = image / 255.

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))

    return image.astype(np.float32)


def print_elapsed_time(total_time):
    hh = int(total_time / 3600)
    mm = int((total_time % 3600) / 60)
    ss = int((total_time % 3600) % 60)
    print("\n** Total Elapsed Runtime: {:0>2}:{:0>2}:{:0>2}".format(hh, mm, ss))
