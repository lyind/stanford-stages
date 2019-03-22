import os
from bottle import Bottle, route, request, run, static_file, response, install
from zipfile import ZipFile
import uuid

# initialize site.USER_SITE package to circumvent error in tensorflow's __init__.py when it is None
from site import getusersitepackages
getusersitepackages()

# avoid tkinter (Tk toolkit support for python is not available on windows embedded python)
import matplotlib
matplotlib.use('agg')

import inf_narco_app

app = Bottle()


NARCO_CONFIG = {
    "show": {
        "diagnosis": False
    },
    "save": {
        "plot": True,
        "hypnogram": True,
        "hypnodensity": True,
        "diagnosis": False,
        "hynogram_anl": True
    },
    "appConfig": {
        "segsize": 120, # overrides auto-detection via model name, segsize = 4*epoch_length
        "cpu_max": 7    # maximum amount of cpu cores assigned for multi-thread tasks
    }
}

def copy_filelike_to_filelike(src, dst, bufsize=16384):
    while True:
        buf = src.read(bufsize)
        if not buf:
            break
        dst.write(buf)


def do_cleanup(id):
    to_clean = [ '.edf', '.pkl', '.hypno_pkl','.anl','.hypnodensity.png','.hypnodensity.txt','.hypnogram.txt','.zip']
    for suffix in to_clean:
        try:
            os.remove(id + suffix)
        except:
            continue


def reply(filename):
    return static_file(filename, root='.', download=True)


@app.route('/api', method='POST')
def evaluation():
    src = request.body
    id = str(uuid.uuid1())
    
    @app.hook('after_request')
    def cleaning_up():
        print('cleaning files')
        do_cleanup(id)
    
    with open(id + ".edf", "wb") as upload:
        copy_filelike_to_filelike(src, upload)
    
    inf_narco_app.main(id + ".edf", NARCO_CONFIG)
    with ZipFile(id + ".zip", mode='w') as returnzip:
        for suffix in ['.anl','.hypnogram.txt','.hypnodensity.png','.hypnodensity.txt']:
            returnzip.write(id + suffix)
    
    print('returning results for ' + id)
    return reply(id + '.zip')


run(app, host='0.0.0.0', port=80, debug=True)

