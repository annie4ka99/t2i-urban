import requests
import os
from io import BytesIO
import base64
from PIL import Image


def send_api_request(text):
    try:
        url = 'http://{0}:{1}/get-images'.format('localhost', 8080)
        return requests.post(url, json={'text': text})
    except requests.exceptions.RequestException as e:
        print(e)


if __name__ == '__main__':
    res_dir = 'results_from_remote_api'
    os.makedirs(res_dir, exist_ok=True)
    for f in os.listdir(res_dir):
        os.remove(os.path.join(res_dir, f))

    text = "Он шел по набережной канавы, и недалеко уж оставалось ему. Но, дойдя до моста, он приостановился и вдруг повернул на мост, в сторону, и прошел на Сенную."
    resp = send_api_request(text).json()

    resp.raise_for_status()
    ans = resp.json()
    for i, encoded in enumerate(resp.json()['result']):
        im = Image.open(BytesIO(base64.b64decode(encoded)))
        im.save(os.path.join(res_dir, f'result_{i}.png'))
