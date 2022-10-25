## Overview

This project is my bachelor's thesis on the topic: "Generation of images of 
urban landscapes in high resolution from a textual description using deep 
learning methods."

For a text that takes place in a city, it generates images of the corresponding urban location in 1024x1024 resolution.

The service works as follows:
1) The service accepts a request containing text in Russian. The text is translated into English.
2) Next, a text is sent to the third party service, which predicts its location using NLP methods. 
It is assumed that location is urban.
3) Images corresponding to the text description of the obtained location are loaded from free photo stocks ([Depositphotos](https://ru.depositphotos.com/), [Unsplash](https://unsplash.com/)).
4) The uploaded images are segmented into classes corresponding to the objects of the urban environment (road, sidewalk, building, etc.). For this, [SegFormer](https://github.com/NVlabs/SegFormer) is used.
5) For each obtained segmentation, several images are generated using [OASIS](https://github.com/boschresearch/OASIS).
6) The most appropriate to textual description generated images are selected using [CLIP](https://github.com/openai/CLIP) score.
7) Finally, x2 super resolution is performed on each image using [Real-ESRGAN](https://github.com/ai-forever/Real-ESRGAN).

## Installation
### Download checkpoints
#### Segmentation checkpoints
Download weights for SegFormer model from [here](https://drive.google.com/file/d/1LAXhIA3oSipTg7eSvx1ImNqB1CvJmpGu/view?usp=sharing) and unpack it into [segmentation/](./segmentation/).

#### Generation checkpoints
Download weights for OASIS model from [here](https://drive.google.com/file/d/1MfDiqQehgabzhx0oeyT7SlHxlo5UF5iY/view?usp=sharing) and unpack it into [generation/OASIS/checkpoints](generation/OASIS/checkpoints).

### Docker build
```sh
docker build -t t2i-urban .
docker run -d -p 8080:8080 --name=[container-name] t2i-urban
```
### Install packages
```sh
docker start [container-name]
docker attach [container-name]
pip install -v -e .
```
## Usage

### Send request
Running docker container sets up a server (on localhost:8080) that accepts requests in the following format:
- method: `get-images`
- Content-Type: `application/json`
- request format: `{"text": string}`
- answer format: `{"result": list<base64-string>}` (list of base64-encoded images)

### Manually
If you want to generate images manually run [main.py](./main.py). Usage: 
```
python main.py --text TEXT [--save-dir DIR] [--samples SAMPLES] [--from-text]

--text TEXT         - input text
--save-dir DIR      - results will be saved here (default is './results')
--samples SAMPLES   - number of images to be generated (default is 10)
--from-text         - with this option enabled images are generated directly from the text 
                      instead of generating from text location
                      (thus text must contain a description of the urban location)
```

#### Important:
Third party service, which predicts text's location can be unavailable. In this case use `--from-text` option to generate images directly from input text. Thus text must contain a description of the urban location.