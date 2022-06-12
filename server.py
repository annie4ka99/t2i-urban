import json
import os
import time
import cherrypy
from cherrypy import log
import yaml
import psutil

from main import text_to_images
from utils.encode import encode_images_base64
process = psutil.Process(os.getpid())  # for monitoring and debugging purposes

config = yaml.safe_load(open("config.yml"))


def process_api_request(body):
    text = body.get('text')

    start = time.time()

    results = encode_images_base64(text_to_images(text))

    time_spent = time.time() - start
    log("Completed api call.Time spent {0:.3f} s".format(time_spent))

    result = {
        'result': results
    }
    return json.dumps(result)


class ApiServerController(object):
    @cherrypy.expose('/get-images')
    def get_images(self):
        cl = cherrypy.request.headers['Content-Length']
        raw = cherrypy.request.body.read(int(cl))
        body = json.loads(raw)
        return process_api_request(body).encode("utf-8")


if __name__ == '__main__':
    cherrypy.tree.mount(ApiServerController(), '/')

    cherrypy.config.update({
        'server.socket_port': config["app"]["port"],
        'server.socket_host': config["app"]["host"],
        'server.thread_pool': config["app"]["thread_pool"],
        'log.access_file': "access1.log",
        'log.error_file': "error1.log",
        'log.screen': True,
        'tools.response_headers.on': True,
        'tools.encode.encoding': 'utf-8',
        'tools.response_headers.headers': [('Content-Type', 'application/json;encoding=utf-8')],
    })

    try:
        cherrypy.engine.start()
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()
