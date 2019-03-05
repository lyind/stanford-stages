import os
from bottle import Bottle, route, request, run, static_file
import inf_narco_app

app = Bottle()

def copy_filelike_to_filelike(src, dst, bufsize=16384):
    while True:
        buf = src.read(bufsize)
        if not buf:
            break
        dst.write(buf)

@app.route('/api', method='POST')
def evaluation():
    src = request.body
    with open("to_evaluate.edf", "wb") as upload:
        copy_filelike_to_filelike(src, upload)
    form = request.query.form
    if form == "txt":
        result = "to_evaluate.hypnogram"
        dlname = "hypnogram.txt"
        inf_narco_app.main("to_evaluate.edf", {"save":{"hypnogram":True}})
    elif form == "png":
        result = "to_evaluate.png"
        dlname = "hypnodensity.png"
        inf_narco_app.main("to_evaluate.edf", {"save":{"plot":True}})
    else:
        result = "to_evaluate.anl"
        dlname = "profil.anl"
        inf_narco_app.main("to_evaluate.edf", {})
    os.remove("to_evaluate.edf")
    os.remove("to_evaluate.pkl")
    os.remove("to_evaluate.hypno_pkl")
    return static_file(result, root='.', download=dlname)


run(app, host='0.0.0.0', port=80, debug=True)

