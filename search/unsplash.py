from pyunsplash import PyUnsplash
import requests
import time
from PIL import Image


UNSPLASH_ACCESS_KEY = 'Ie1P5akw6t7SsUUUlpIc5jluRJoPre2hqhtaIP8Yavw'

pu = PyUnsplash(api_key=UNSPLASH_ACCESS_KEY)


def get_unsplash_images(query, count):
    photos = pu.photos(type_='random', count=count, featured=True, query=query)
    photos = photos.entries

    photos_urls = [photo.link_download for photo in photos]

    # request_time = 0.0
    results = []
    for url in photos_urls:
        start_time = time.time()
        try:
            resized_im = Image.open(requests.get(url + "&fit=crop&h=1024&w=1024", stream=True).raw).convert('RGB')
            # request_time += (time.time() - start_time)
            results.append(resized_im)
        except:
            pass

    # print('avg request time per image to Unsplash:{0:.2f} s'.format((request_time / count)))
    # print('total request time to Unsplash:{0:.2f} s'.format(request_time))

    if len(results) > 0 and len(results) % 2 != 0:
        results.append(results[-1])

    return results
