import cv2
import numpy as np
import random
import os

STATS = [0., 0.]

def load():
    '''loads RGB image'''
    def im_load(path):
        return cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)[:,:,[2,1,0]]
    return im_load


def normalize(mean, std):
    ''' subtracts the specified mean from the image and later divides by specified std'''
    def im_normalize(im):
        return np.divide(np.subtract(im, mean, im), std, im)
    return im_normalize


def random_crop(size, area_range, aspect_range):

    area_from, area_to = area_range
    aspect_from, aspect_to = aspect_range

    fallback = square_centrer_crop_resize(size)

    def im_random_crop(im):
        for attempt in xrange(10):
            initial_area = float(im.shape[0] * im.shape[1])

            area_fraction = random.random()*(area_to - area_from) + area_from
            aspect = random.random() * (aspect_to - aspect_from) + aspect_from

            requested_area = initial_area * area_fraction
            h = int((requested_area / aspect)**0.5)
            w = int((requested_area * aspect)**0.5)

            if random.random() < 0.5:  # in case aspect_range has uneven distance distribution around 1
                h, w = w, h

            x_space = im.shape[1] - w
            y_space = im.shape[0] - h

            if x_space < 0 or y_space < 0:
                continue # better luck cropping next time!

            x_off = random.randrange(x_space + 1)
            y_off = random.randrange(y_space + 1)

            cropped = im[y_off:y_off+h, x_off:x_off+w]
            return cv2.resize(cropped, size, interpolation=cv2.INTER_LINEAR)

        # a strange image, just take a largest square central crop and resize to required shape (implemented by fallback)...
        return fallback(im)

    return im_random_crop



def square_centrer_crop_resize(size):
    def square_centrer_crop_resize_op(im):
        short_edge = min(im.shape[:2])
        yy = int((im.shape[0] - short_edge) / 2)
        xx = int((im.shape[1] - short_edge) / 2)

        max_square = im[yy: yy + short_edge, xx: xx + short_edge]
        return cv2.resize(max_square, size, interpolation=cv2.INTER_LINEAR)
    return square_centrer_crop_resize_op


def crop_at(crop_size, position='CC'):
    '''crops the required patch of crop_size size at specified position.
       Position can be specified in terms of C (center), L(left) right (R).
       Eg. To crop top centre patch use 'LC' to crop bottom left use 'RL'.

        NOTE: input image has to have size not smaller than crop_size!
       '''
    def crop_at_op(im):
        im_size = im.shape
        assert im_size[0] >= crop_size[0] and im_size[1] >= crop_size[1]
        specs = []
        for i in xrange(2):
            if position[i]=='L':
                s = 0
                e = crop_size[i]
            elif position[i]=='C':
                s = ( im_size[i] - crop_size[i] ) / 2
                e = s + crop_size[i]
            elif position[i]=='R':
                s = im_size[i] - crop_size[i]
                e = im_size[i]
            else:
                raise ValueError('Invalid crop position letters must be in LCR ')
            specs.append((s, e))
        return im[specs[0][0]:specs[0][1], specs[1][0]:specs[1][1], :]

    return crop_at_op


def resize(size):
    def resize_op(im):
        return cv2.resize(im, size, interpolation=cv2.INTER_LINEAR) # simple linear resize
    return resize_op


def random_lighting(alphastd, eigval, eigvec):
    def im_random_lighting_op(im):

        alpha = np.random.normal(0, alphastd, (3,))

        rgb_shift = np.sum(eigval * eigvec * alpha, axis=1)

        return np.add(im, rgb_shift, im)
    return im_random_lighting_op




def get_images_op(folder, endings=('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP')):
    dirs = [p for p in os.listdir(folder) if os.path.isdir(os.path.join(folder, p))]
    return [os.path.join(folder, p, e)  for p in dirs for e in os.listdir(os.path.join(folder, p)) if any(e.endswith(ending) for ending in endings)]


def horizontal_flip():
    ''' flips the image horizontally must be H,W,C'''
    def im_horizontal_flip(im):
        return np.fliplr(im)
    return im_horizontal_flip




