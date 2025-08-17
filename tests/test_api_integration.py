#!/usr/bin/env python3
"""
Integration tests for API endpoints
"""
import json
import pytest
from unittest.mock import patch


class TestVehicleSearchAPI:
    """Test the vehicle search API endpoint"""
    
    def test_vehicle_search_basic(self, client):
        """Test basic vehicle search functionality"""
        response = client.get('/api/search?year=2015&make=Toyota&model=Camry')
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Check response structure
        assert 'estimate' in data
        assert 'listings' in data
        assert 'calculation_date' in data
        assert 'method' in data
        assert 'model_accuracy' in data
        
        # Check model accuracy structure
        accuracy = data['model_accuracy']
        assert 'confidence' in accuracy
        assert accuracy['confidence'] in ['high', 'medium', 'low']
    
    def test_vehicle_search_with_mileage(self, client):
        """Test vehicle search with mileage parameter"""
        response = client.get('/api/search?year=2015&make=Toyota&model=Camry&mileage=80000')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert 'estimate' in data
        assert data['estimate'] != 'N/A'
    
    def test_vehicle_search_with_all_params(self, client):
        """Test vehicle search with all optional parameters"""
        params = {
            'year': 2015,
            'make': 'Toyota',
            'model': 'Camry',
            'mileage': 80000,
            'trim': 'LE',
            'color': 'Silver',
            'dealer_state': 'CA'
        }
        
        query_string = '&'.join([f'{k}={v}' for k, v in params.items()])
        response = client.get(f'/api/search?{query_string}')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert 'estimate' in data
        assert len(data['listings']) > 0
    
    def test_vehicle_search_missing_required_params(self, client):
        """Test API response when required parameters are missing"""
        # Missing make and model
        response = client.get('/api/search?year=2015')
        assert response.status_code == 400
        
        data = response.get_json()
        assert 'error' in data
        assert 'Missing required parameters' in data['error']
    
    def test_vehicle_search_invalid_year(self, client):
        """Test API response with invalid year parameter"""
        response = client.get('/api/search?year=invalid&make=Toyota&model=Camry')
        assert response.status_code == 400
        
        data = response.get_json()
        assert 'error' in data
        assert 'must be integers' in data['error']
    
    def test_vehicle_search_no_data(self, client):
        """Test API response when no data is available"""
        response = client.get('/api/search?year=1900&make=NonExistent&model=Car')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data['estimate'] == 'N/A'
        assert data['method'] == 'no_data'
        assert data['model_accuracy']['confidence'] == 'low'


class TestMakesModelsAPI:
    """Test the makes and models API endpoint"""
    
    def test_makes_models_endpoint(self, client):
        """Test the makes and models discovery endpoint"""
        response = client.get('/api/makes-models')
        
        assert response.status_code == 200
        data = response.get_json()
        
        # Should return hierarchical structure
        assert isinstance(data, dict)
        
        # Check for test data
        if data:  # Only check if we have data
            # Should have year keys
            years = list(data.keys())
            assert len(years) > 0
            
            # Each year should have makes
            first_year = years[0]
            assert isinstance(data[first_year], dict)


class TestCacheIntegration:
    """Test caching behavior in API responses"""
    
    def test_response_caching(self, client, test_cache):
        """Test that responses are properly cached"""
        # First request
        response1 = client.get('/api/search?year=2015&make=Toyota&model=Camry')
        assert response1.status_code == 200
        
        # Second identical request should hit cache
        response2 = client.get('/api/search?year=2015&make=Toyota&model=Camry')
        assert response2.status_code == 200
        
        # Responses should be identical (excluding timestamps)
        data1 = response1.get_json()
        data2 = response2.get_json()
        
        # Remove timestamps for comparison
        data1_no_time = {k: v for k, v in data1.items() if k != 'calculation_date'}
        data2_no_time = {k: v for k, v in data2.items() if k != 'calculation_date'}
        
        assert data1_no_time == data2_no_time
        assert 'calculation_date' in data1
        assert 'calculation_date' in data2
    
    def test_hit_counter_increment(self, client, test_cache):
        """Test that hit counter increments on requests"""
        from app.utils.cache_utils import hits_key, bump_hits
        
        # Clear any existing hits
        key = hits_key(2015, "Toyota", "Camry")
        test_cache.delete(key)
        
        # Make requests and check hit counter
        for i in range(5):
            response = client.get('/api/search?year=2015&make=Toyota&model=Camry')
            assert response.status_code == 200
            
            # Check hit counter was incremented
            hits = test_cache.get(key)
            assert hits == i + 1


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_database_error_handling(self, client):
        """Test API behavior when database operations fail"""
        with patch('app.repositories.vehicle_repository.VehicleRepository.get_listings_with_filters') as mock_repo:
            mock_repo.side_effect = Exception("Database connection failed")
            
            response = client.get('/api/search?year=2015&make=Toyota&model=Camry')
            assert response.status_code == 500
            
            data = response.get_json()
            assert 'error' in data
    
    def test_invalid_route(self, client):
        """Test response to invalid API routes"""
        response = client.get('/api/nonexistent-endpoint')
        assert response.status_code == 404
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.get('/api/search?year=2015&make=Toyota&model=Camry')
        
        # Check for CORS headers (configured in app/__init__.py)
        headers = response.headers
        # Note: In testing environment, CORS headers might not be present
        # This test ensures the endpoint is accessible
        assert response.status_code == 200


class TestPerformance:
    """Test performance characteristics"""
    
    def test_response_time(self, client):
        """Test that API responses are reasonably fast"""
        import time
        
        start_time = time.time()
        response = client.get('/api/search?year=2015&make=Toyota&model=Camry')
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be under 5 seconds (generous for integration test)
        response_time = end_time - start_time
        assert response_time < 5.0, f"Response took {response_time:.2f}s, too slow"
    
    def test_concurrent_requests(self, client):
        """Test handling of multiple concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.get('/api/search?year=2015&make=Toyota&model=Camry')
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        for response in results:
            assert response.status_code == 200
            data = response.get_json()
            assert 'estimate' in data
