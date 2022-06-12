import requests
import re
from PIL import Image


def get_deposit_photos_images(query, count):
    dp_query = "-".join(query.split(' '))
    url = f'https://depositphotos.com/stock-photos/{dp_query}.html?filter=photos'
    response = requests.post(url)
    text = response.text
    urls_dict = dict()
    for m in re.finditer(r'https://st2\.depositphotos\.com/[0-9]+/[0-9]+/[a-z]/([0-9]+)/.*?\.(jpg|png)', text):
        size = int(m.group(1))
        url = m.group(0)
        name = url.split('/')[-1]
        if size >= 450 and (name not in urls_dict or urls_dict[name][0] < size):
            urls_dict[name] = (size, url)
    imgs = []
    for i, (k, img_url) in enumerate(list(urls_dict.values())[:min(count, len(urls_dict))]):
        imgs.append(Image.open(requests.get(img_url, stream=True).raw).convert('RGB'))
    return imgs