import os
from bottle import Bottle, route, request, run, static_file, response, install
import inf_narco_app
from zipfile import ZipFile
import uuid

app = Bottle()

def copy_filelike_to_filelike(src, dst, bufsize=16384):
    while True:
        buf = src.read(bufsize)
        if not buf:
            break
        dst.write(buf)


def do_cleanup(file_uuid):
    to_clean = ['.edf','.pkl','.hypno_pkl','.anl','.png','.hypnogram','.zip']
    for i in to_clean:
        try:
            os.remove(file_uuid + i)
        except:
            continue


def reply(filename):
    return static_file(filename, root='.', download=True)


@app.route('/api', method='POST')
def evaluation():
    src = request.body
    random_name = str(uuid.uuid1())
    
    @app.hook('after_request')
    def cleaning_up():
        print('cleaning files')
        do_cleanup(random_name)
    
    with open(random_name + ".edf", "wb") as upload:
        copy_filelike_to_filelike(src, upload)
    
    inf_narco_app.main(random_name + ".edf", {})
    with ZipFile(random_name + ".zip", mode='w') as returnzip:
        returnzip.write(random_name + ".anl")
        returnzip.write(random_name + ".png")
        returnzip.write(random_name + ".hypnogram")
    
    print('returning results')
    return reply(random_name + '.zip')


run(app, host='0.0.0.0', port=80, debug=True)

