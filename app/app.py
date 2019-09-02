from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, render_template, flash
from flask_cors import CORS, cross_origin
import json
from models import db
# Import of the models
# https://www.compose.com/articles/using-postgresql-through-sqlalchemy/

DATABASE_URL = "postgresql+psycopg2://root:root@postgres:5432/optimization"

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

db.init_app(app)

with app.app_context():
    db.create_all()
# db.session.commit()


@app.route("/upload", methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        flash("Job successfully created. Redirected to optimization table.")
        return json.dumps({
            'status': 200
        })

    if request.method == 'GET':
        if request.content_type == "application/json":
            return json.dumps({
                'message': "test"
            })
        return render_template('upload.html')


if __name__ == "__main__":
    app.secret_key = '2349978342978342907889709154089438989043049835890'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, host='0.0.0.0')
