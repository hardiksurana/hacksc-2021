# contains app and routes

from src.backend.api.resources.quiz import GenerateQuizAPI
import os
import logging

from src.config import app_config
import flask
from flask import Flask
from flask_restful import Api

# set up configurations
app = Flask(__name__, instance_relative_config=True)
config_name = os.getenv("FLASK_ENV", "development")
app.config.from_object(app_config[config_name])

api = Api(app)
approot = app.config['APPLICATION_ROOT']

# expose endpoints here
api.add_resource(GenerateQuizAPI, approot + '/quiz', endpoint='api.quiz')
