import requests
from googletrans import Translator

translator = Translator()
url = 'http://{0}:{1}/extract-infer-location'.format('109.188.135.85', 8080)


def get_locations(ru_text):
    text = translator.translate(ru_text, src='ru').text
    print(f'translated text: {text}')

    response = requests.post(url, json={'text': text})
    resp_body = response.json()
    print("status:", response.status_code)
    locations = []
    if response.status_code != 200:
        print(resp_body['error'])
    else:
        extracted = resp_body['locations']['extracted']
        inferred = resp_body['locations']['inferred']
        locations = inferred if len(extracted) == 0 else extracted

    if len(locations) == 0:
        print("no location detected")
    else:
        print("found locations:", locations)
    return locations
