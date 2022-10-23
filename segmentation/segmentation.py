import os
import random
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torchvision

import mmcv
import mmseg
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.apis import set_random_seed, inference_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def get_model(cfg_path, ckpt_path, multiscale=False):
    cfg = Config.fromfile(cfg_path)
    if multiscale:
        cfg.test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    # resize image to multiple of 32, improve SegFormer by 0.5-1.0 mIoU.
                    dict(type='ResizeToMultiple', size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **cfg_ade.img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
    model = init_segmentor(
        cfg,
        checkpoint=ckpt_path,
        device='cuda:0')
    return model


labels = [([90, 128, 21, 60, 26, 69, 76, 103, 104, 109, 113, 132, 138], [], [], []),
 ([1, 48, 25, 84, 79, 88, 148], [], [], []),
 ([], [], [2], [2]),
 ([], [], [], [3]),
 ([], [], [], [4]),
 ([43, 100, 123], [], [], [5]),
 ([], [], [6], [6]),
 ([42], [], [7], [7]),
 ([], [], [], [8]),
 ([], [], [], [9]),
 ([86, 106, 114], [], [9], [10]),
 ([53, 59, 96, 121], [], [], [11]),
 ([149], [], [], [12]),
 ([], [0], [], []),
 ([140], [1], [], []),
 ([0], [3], [], []),
 ([32, 61], [4], [], []),
 ([93, 87], [5], [], []),
 ([136], [6], [], []),
 ([], [7], [], []),
 ([4, 17, 66, 72], [], [], []),
 ([9, 13, 16, 29, 34, 46, 52, 94, 68, 91], [], [], []),
 ([2], [], [], []),
 ([], [11], [], []),
 ([], [12], [], []),
 ([], [13], [], []),
 ([122], [14], [], []),
 ([], [15], [], []),
 ([], [16], [], []),
 ([], [17], [], []),
 ([], [18], [], [])]

label_order = [21, 1, 2, 3, 4, 6,
               # 7,
               8, 9, 10, 20, 12, 13, 16, 5, 14, 11, 15, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 22, 0]


model_ade = get_model('segmentation/configs/segformer/segformer_mit-b5_640x640_160k_ade20k.py',
                      'segmentation/checkpoints/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth')
model_cityscapes = get_model('segmentation/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py',
                             'segmentation/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth')
model_facades = get_model('segmentation/configs/segformer/segformer_mit-b5_512x512_160k_facades_ade.py',
                          'segmentation//work_dirs/facades_ade_segformer/latest.pth')
model_urban = get_model('segmentation/configs/segformer/segformer_mit-b5_512x512_160k_ade_urban.py',
                        'segmentation//work_dirs/ade_urban_segformer/latest.pth')

semantic_nc = 31
back_color = 0

window_no_replace = np.array(
    [[1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 2, 3, 3, 3, 3, 3, 2, 1],
     [1, 2, 3, 4, 4, 4, 3, 2, 1],
     [1, 2, 3, 4, 0, 4, 3, 2, 1],
     [1, 2, 3, 4, 4, 4, 3, 2, 1],
     [1, 2, 3, 3, 3, 3, 3, 2, 1],
     [1, 2, 2, 2, 2, 2, 2, 2, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1]
     ])

window_in_place = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 2, 2, 2, 2, 2, 2, 2, 1],
                            [1, 2, 3, 3, 3, 3, 3, 2, 1],
                            [1, 2, 3, 4, 4, 4, 3, 2, 1],
                            [1, 2, 3, 4, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]])


def seg_inpainting(input_img, inp_pixels, inp_color, in_place=False, paint_back=False):
    window_weights = window_in_place if in_place else window_no_replace
    finished = True
    passed_pixels = []

    img = input_img.copy()
    h, w = img.shape
    label_maps = []
    for i in range(semantic_nc):
        label_maps.append((img == i).astype(img.dtype))
    for (i, j) in inp_pixels:
        prev_c = img[i, j]
        if prev_c != inp_color:
            continue
        counts = []
        up = 0 if i >= r else r - i
        down = 2 * r + 1 if i + r < h else 2 * r + 1 - ((i + r) - h + 1)
        left = 0 if j >= r else r - j
        right = 2 * r + 1 if j + r < w else 2 * r + 1 - ((j + r) - w + 1)
        weights = window_weights[up:down, left:right]

        for cl in range(semantic_nc):
            counts.append(
                (label_maps[cl][max(i - r, 0):min(i + r + 1, h), max(j - r, 0):min(j + r + 1, w)] * weights).sum())
        ind = np.argsort(-np.array(counts))[:4]

        paint_c = ind[0]
        if paint_c == inp_color:
            for k in range(1, 4):
                if counts[ind[k]] != 0 and ind[k] != inp_color:
                    paint_c = ind[k]
                    break

        img[i, j] = paint_c
        if paint_c == inp_color:
            passed_pixels.append((i, j))
            finished = False
            continue
        if in_place:
            label_maps[prev_c][i, j] = 0
            label_maps[paint_c][i, j] = 1
    return img, finished, passed_pixels


def segmentation_inpainting(seg):
    back_label = 0
    back_pixels = []
    people_pixels = []
    transport_pixels = []
    big_transport_pixels = []
    h, w = seg.shape
    res = []
    for i in range(h):
        for j in range(w):
            lab = seg[i, j]
            if lab == back_label:
                back_pixels.append((i, j))
            elif lab >= 23 and lab <= 24:
                people_pixels.append((i, j))
                continue
            elif lab == 25 or 29 <= lab <= 30:
                transport_pixels.append((i, j))
            elif lab >= 26 and lab <= 27:
                big_transport_pixels.append((i, j))
                continue

    inp_pixels = back_pixels
    total_pixels = h * w
    max_pixels = int(total_pixels * 0.15)
    #     print("max:", max_pixels)
    #     print("init:", len(inp_pixels))
    #     print("people:", len(people_pixels))
    #     print("transport:", len(transport_pixels))

    for ps in [people_pixels, transport_pixels, big_transport_pixels]:
        extra_pixels = ps.copy()
        if len(inp_pixels) + len(extra_pixels) < max_pixels:
            inp_pixels += extra_pixels
            for (i, j) in extra_pixels:
                seg[i, j] = back_label
        else:
            break

    start_time = time.time()

    passes = 0
    max_passes = 40
    finished = False
    while (not finished) and passes < max_passes:
        seg, finished, inp_pixels = seg_inpainting(seg, inp_pixels, back_label)
        passes += 1

    total_time = time.time() - start_time
    print('avg time per pass:{0:.2f} s'.format((total_time / passes)))
    print('total time:{0:.2f} s'.format(total_time))
    print('passes:', passes)

    return seg


def create_segmentation(imgs):
    results = []
    for i, img in enumerate(imgs):
        result_ade = inference_segmentor(model_ade, img)[0]
        result_cityscapes = inference_segmentor(model_cityscapes, img)[0]
        result_facades = inference_segmentor(model_facades, img)[0]
        result_urban = inference_segmentor(model_urban, img)[0]
        assert (result_ade.shape == result_cityscapes.shape and
                result_cityscapes.shape == result_facades.shape and
                result_facades.shape == result_urban.shape)

        result = np.zeros(result_ade.shape, np.int32)
        for cl in label_order:
            (_, cityscapes, _, _) = labels[cl]
            for model_cl in cityscapes:
                result[result_cityscapes == model_cl] = cl

        for cl in label_order:
            (ade, _, facades, urban) = labels[cl]
            for model_cl in ade:
                result[result_ade == model_cl] = cl
            for model_cl in facades:
                result[result_facades == model_cl] = cl
            for model_cl in urban:
                result[result_urban == model_cl] = cl

        im = Image.fromarray(result).convert('P')
        results.append(im)

    return results
