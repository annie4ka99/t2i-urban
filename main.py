import os
import time
import numpy as np
import torch
from rudalle.pipelines import super_resolution
from rudalle import get_realesrgan
import clip

from utils.location import get_locations
from search.unsplash import get_unsplash_images
from search.despositphotos import get_deposit_photos_images
from utils.image import resize_and_crop
from utils.queries import get_query
from segmentation.segmentation import create_segmentation
from generation.generate import generate_images


IMG_NUM = 10
SAMPLES_PER_IMG = 1

RES_DIR = "results"
ade_palette = np.loadtxt('segmentation/ade_palette.txt', dtype=np.uint8)
realesrgan = get_realesrgan('x2', device='cuda:0')


device = "cuda"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


def super_res_x2(pil_imgs):
    return super_resolution(pil_imgs, realesrgan)


def get_best_images_clip(pil_images, en_text, count):
    processed_images = []
    for img in pil_images:
        image = clip_preprocess(img).to(device)
        processed_images.append(image)
    images_tensors = torch.stack(processed_images)
    text = clip.tokenize([en_text]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = clip_model(images_tensors, text)
        scores = logits_per_text.cpu().numpy()[0]

    best_idx = np.argsort(-scores)[:count]
    best_images = []
    for i in best_idx:
        best_images.append(pil_images[i])
    return best_images


def prepare_res_dir():
    os.makedirs(RES_DIR, exist_ok=True)
    for f in os.listdir(RES_DIR):
        os.remove(os.path.join(RES_DIR, f))


def save_images(images, prefix):
    sr_images = super_res_x2(images)
    for i, image in enumerate(sr_images):
        suffix = str((i // 6) * 2 + i % 2) + '_' + str((i // 2) % 3)
        # suffix = str(i//4 * 2 + i % 2) + '_' + str((i // 2) % 2)
        image.save(os.path.join(RES_DIR, f'{prefix}_{suffix}.png'))


def save_segmentations(segs, prefix):
    for i, seg in enumerate(segs):
        seg_copy = seg.copy()
        seg_copy.putpalette(ade_palette[:31])
        seg_copy.save(os.path.join(RES_DIR, f'{prefix}_{i}.png'))


def save_originals(images, prefix):
    for i, image in enumerate(images):
        image.save(os.path.join(RES_DIR, f'{prefix}_{i}.png'))


def text_to_images(text, save_res=False, count=10):
    # locations = get_locations(text)
    locations = [text]
    results = []
    best_results = []

    if save_res:
        prepare_res_dir()

    start_time = time.time()
    for location in locations:
        location_results = []
        best_location_results = []

        query = get_query(location)
        images = get_unsplash_images(query, IMG_NUM)
        images += get_deposit_photos_images(query, IMG_NUM)

        images = [resize_and_crop(img) for img in images]
        numpy_images = [np.asarray(img) for img in images]
        segmentations = create_segmentation(numpy_images)
        if save_res:
            save_segmentations(segmentations, f"{location}_lab")

        for i in range(len(segmentations) // 2):
            seg_batch = [segmentations[2 * i], segmentations[2 * i + 1]]
            for _ in range(SAMPLES_PER_IMG):
                res_batch = generate_images(seg_batch)
                location_results += res_batch
        results += location_results

        if save_res:
            save_originals(images, f"{location}_orig")
            save_images(location_results, f"{location}_gen")

        best_location_results = super_res_x2(get_best_images_clip(results, text, count))
        best_results += best_location_results
        if save_res:
            for i, img in enumerate(best_results):
                img.save(f'{RES_DIR}/{location}_best_{i}.png')

    time_spent = (time.time() - start_time)
    print('total time spent for generating: %.3f s' % time_spent)
    print('avg time for generating 1 image: %.3f s' % (time_spent / len(results)))

    return best_results
