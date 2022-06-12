import base64
from io import BytesIO


def encode_images_base64(images):
    result = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
        result.append(img_str)
    return result
