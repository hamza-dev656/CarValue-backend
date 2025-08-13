from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config.settings import Config
from flask_cors import CORS
from flask_caching import Cache

cache = Cache()
db = SQLAlchemy()

def create_app():
    """Application factory"""
    app = Flask(__name__)
    app.config.from_object(Config)

    return app