# app/__init__.py (snippet)

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config.settings import Config
from flask_cors import CORS
from flask_caching import Cache
import os

cache = Cache()
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Configure cache (Redis with fallback to SimpleCache)
    from app.config.model_config import CACHE_REDIS_URL
    try:
        # Try Redis first
        app.config["CACHE_TYPE"] = "RedisCache"
        app.config["CACHE_REDIS_URL"] = CACHE_REDIS_URL
        app.config["CACHE_DEFAULT_TIMEOUT"] = 3600
        print(f"📦 Using Redis cache: {CACHE_REDIS_URL}")
    except Exception as e:
        # Fallback to SimpleCache if Redis is not available
        print(f"⚠️ Redis not available ({e}), falling back to SimpleCache")
        app.config["CACHE_TYPE"] = "SimpleCache"
        app.config["CACHE_DEFAULT_TIMEOUT"] = 3600

    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

    db.init_app(app)
    
    try:
        cache.init_app(app)
        print(f"✅ Cache initialized successfully: {app.config['CACHE_TYPE']}")
    except Exception as e:
        print(f"❌ Cache initialization failed: {e}")
        # Emergency fallback
        app.config["CACHE_TYPE"] = "SimpleCache"
        cache.init_app(app)
        print("✅ Emergency fallback to SimpleCache")

    from app.controllers.vehicle_controller import vehicle_bp
    app.register_blueprint(vehicle_bp, url_prefix="/api")

    # Initialize ML training asynchronously after app starts
    with app.app_context():
        is_main = os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug
        if is_main:
            try:
                from app.services.bootstrap.bootstrap import initialize_models_async
                initialize_models_async(app)
            except Exception as e:
                print(f"⚠️ Failed to schedule background training: {e}")

    return app
