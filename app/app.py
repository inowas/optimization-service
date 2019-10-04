from flask import Flask
from flask_cors import CORS
from views import optimization_blueprint
from config import DATABASE_URL
from models import Base, OptimizationTask, CalculationTaskEvolutionaryOptimization, CalculationTaskLinearOptimization
from db import engine
from time import sleep
# https://www.compose.com/articles/using-postgresql-through-sqlalchemy/

sleep(10)


# Create a flask app
app = Flask(__name__)
# Configs for flask app
app.config["DEBUG"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Enable cross origin resource sharing
app.register_blueprint(optimization_blueprint)

CORS(app)

tables = [OptimizationTask.__table__]

Base.metadata.create_all(bind=engine,
                         tables=tables,
                         checkfirst=True)

if __name__ == "__main__":
    app.secret_key = '2349978342978342907889709154089438989043049835890'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, host='0.0.0.0')
