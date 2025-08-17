#!/usr/bin/env python3
"""
Pytest configuration and fixtures for integration tests
"""
import os
import sys
import tempfile
import shutil
import pytest
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app, db, cache
from app.models.vehicle import Vehicle


@pytest.fixture(scope="session")
def app():
    """Create application instance for testing"""
    # Set environment variables for test database before creating app
    test_env = {
        'DATABASE_URL': 'sqlite:///:memory:',
        'CACHE_TYPE': 'SimpleCache', 
        'TESTING': '1',
        'SECRET_KEY': 'test-secret-key',
        'WERKZEUG_RUN_MAIN': 'false'  # Disable background training during tests
    }
    
    with patch.dict(os.environ, test_env, clear=False):
        app = create_app()
        
        # Force override the database URI
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['TESTING'] = True
        app.config['CACHE_TYPE'] = 'SimpleCache'
        app.config['CACHE_DEFAULT_TIMEOUT'] = 300
        app.config['WTF_CSRF_ENABLED'] = False
        
        with app.app_context():
            # Create all tables
            db.create_all()
            
            # Create test data
            _create_test_vehicles()
            
            yield app
            
            # Cleanup
            db.drop_all()


@pytest.fixture(scope="function")
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture(scope="function")
def test_cache(app):
    """Get cache instance for testing"""
    with app.app_context():
        cache.clear()
        yield cache
        cache.clear()


@pytest.fixture(scope="session")
def temp_artifacts_dir():
    """Create temporary artifacts directory for testing"""
    temp_dir = tempfile.mkdtemp(prefix="vinaudit_test_")
    
    # Patch the artifact directory
    with patch.dict(os.environ, {
        'CARVALUE_ARTIFACT_DIR': temp_dir,
        'CARVALUE_DATA_DIR': os.path.join(temp_dir, 'data')
    }):
        os.makedirs(os.path.join(temp_dir, 'data'), exist_ok=True)
        yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def _create_test_vehicles():
    """Create test vehicle data"""
    test_vehicles = [
        # 2015 Toyota Camry data (for local training test)
        Vehicle(
            vin="1HGBH41JXMN109101", year=2015, make="Toyota", model="Camry",
            trim="LE", listing_price=15000, listing_mileage=80000,
            dealer_state="CA", exterior_color="Silver", used=True
        ),
        Vehicle(
            vin="1HGBH41JXMN109102", year=2015, make="Toyota", model="Camry",
            trim="SE", listing_price=16500, listing_mileage=75000,
            dealer_state="CA", exterior_color="Black", used=True
        ),
        Vehicle(
            vin="1HGBH41JXMN109103", year=2015, make="Toyota", model="Camry",
            trim="LE", listing_price=14800, listing_mileage=90000,
            dealer_state="TX", exterior_color="White", used=True
        ),
        # Add more vehicles for different makes/models
        Vehicle(
            vin="1HGBH41JXMN109104", year=2018, make="Honda", model="Civic",
            trim="LX", listing_price=18000, listing_mileage=45000,
            dealer_state="NY", exterior_color="Blue", used=True
        ),
        Vehicle(
            vin="1HGBH41JXMN109105", year=2020, make="Ford", model="F-150",
            trim="XLT", listing_price=35000, listing_mileage=25000,
            dealer_state="TX", exterior_color="Red", used=True
        ),
    ]
    
    # Create more vehicles for Toyota Camry to enable local training
    for i in range(50):  # Need minimum 50 for local training
        vin_suffix = f"{109106 + i:06d}"
        vehicle = Vehicle(
            vin=f"1HGBH41JXMN{vin_suffix}",
            year=2015,
            make="Toyota", 
            model="Camry",
            trim="LE" if i % 2 == 0 else "SE",
            listing_price=14000 + (i * 100),  # Price variation
            listing_mileage=70000 + (i * 1000),  # Mileage variation
            dealer_state="CA" if i % 3 == 0 else "TX",
            exterior_color=["Silver", "Black", "White", "Red"][i % 4],
            used=True
        )
        test_vehicles.append(vehicle)
    
    # Add all vehicles to database
    for vehicle in test_vehicles:
        db.session.add(vehicle)
    
    db.session.commit()
    print(f"✅ Created {len(test_vehicles)} test vehicles")


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache for testing cache operations"""
    cache_data = {}
    
    class MockCache:
        def get(self, key):
            return cache_data.get(key)
        
        def set(self, key, value, timeout=None):
            cache_data[key] = value
            return True
        
        def delete(self, key):
            cache_data.pop(key, None)
            return True
        
        def clear(self):
            cache_data.clear()
            return True
    
    return MockCache()
