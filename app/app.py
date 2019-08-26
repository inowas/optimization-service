from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import json

SCHEMA_SERVER_URL = 'https://schema.inowas.com'

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        return json.dumps({
            'status': 200
        })

    if request.method == 'GET':
        return render_template('upload.html')


if __name__ == '__main__':
    app.secret_key = '2349978342978342907889709154089438989043049835890'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True, host='0.0.0.0')
