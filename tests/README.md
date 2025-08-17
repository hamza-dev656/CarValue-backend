# VinAudit Backend Integration Tests

This directory contains comprehensive integration tests for the VinAudit backend system, covering all major components and workflows.

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── test_api_integration.py        # API endpoint integration tests
├── test_ml_integration.py         # ML pipeline and model tests
├── test_cache_integration.py      # Cache and hit counter tests
├── test_background_training.py    # Background training workflow tests
└── README.md                      # This file
```

## Test Categories

### 🌐 API Integration Tests (`test_api_integration.py`)
- **Vehicle Search API**: Complete request-response flow testing
- **Parameter Validation**: Required/optional parameter handling
- **Error Handling**: Invalid inputs, missing data scenarios
- **CORS Configuration**: Cross-origin request support
- **Performance**: Response time benchmarks
- **Concurrent Requests**: Multi-threading safety

### 🤖 ML Integration Tests (`test_ml_integration.py`)
- **Global Model Training**: Complete training pipeline with TSV export
- **Local Model Training**: Segment-specific model creation
- **Model Registry**: Save/load operations with metadata
- **Prediction Pipeline**: Feature preparation and model inference
- **Data Export Logic**: TSV reuse and force re-export behavior
- **Feature Consistency**: Training vs prediction feature alignment

### ⚡ Cache Integration Tests (`test_cache_integration.py`)
- **Hit Counter System**: Request frequency tracking
- **Training Locks**: Concurrent training prevention
- **Cache Operations**: Get/set/delete with error handling
- **TTL Behavior**: Time-based cache expiration
- **Key Generation**: Consistent cache key creation
- **Error Resilience**: Graceful cache failure handling

### 🔄 Background Training Tests (`test_background_training.py`)
- **Background Workflows**: Asynchronous training execution
- **Flask Context Management**: Proper app context in threads
- **External Service Integration**: HTTP calls to training services
- **Fallback Mechanisms**: Local training when external services fail
- **Thread Management**: Proper cleanup and resource management
- **Bootstrap Process**: Server startup model initialization

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-cov

# Ensure virtual environment is activated
source vin/bin/activate
```

### Basic Test Execution

```bash
# Run all tests
python run_tests.py

# Run specific test category
python run_tests.py --category api
python run_tests.py --category ml
python run_tests.py --category cache
python run_tests.py --category background

# Run with verbose output
python run_tests.py --verbose

# Run with coverage reporting
python run_tests.py --coverage

# Quick tests only (exclude slow tests)
python run_tests.py --category quick
```

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_api_integration.py

# Run specific test class
pytest tests/test_api_integration.py::TestVehicleSearchAPI

# Run specific test method
pytest tests/test_api_integration.py::TestVehicleSearchAPI::test_vehicle_search_basic

# Run with markers
pytest -m "not slow"
pytest -m "api"
```

### Test Setup Verification

```bash
# Check test environment setup
python run_tests.py --setup-only
```

## Test Configuration

### Environment Variables

Tests automatically configure the following environment variables:

```bash
TESTING=1                              # Enable test mode
CACHE_TYPE=SimpleCache                 # Use simple cache for testing
SQLALCHEMY_DATABASE_URI=sqlite:///:memory:  # In-memory database
CARVALUE_REUSE_TSV=1                  # Enable TSV reuse
CARVALUE_FORCE_REEXPORT=0             # Disable forced re-export
CARVALUE_ARTIFACT_DIR=./test_artifacts # Test artifacts directory
```

### Test Data

Tests use an in-memory SQLite database with pre-populated test data:

- **55+ Toyota Camry records** (2015) - Enables local model training tests
- **Multiple vehicle types** - Honda Civic, Ford F-150, etc.
- **Varied attributes** - Different trims, colors, mileage, states
- **Price variations** - Realistic price ranges for testing

## Test Fixtures

### Core Fixtures (`conftest.py`)

- **`app`** - Flask application instance with test configuration
- **`client`** - Test client for API requests
- **`test_cache`** - Simple cache instance for testing
- **`temp_artifacts_dir`** - Temporary directory for model artifacts
- **`mock_redis_cache`** - Mock cache for isolated testing

### Usage Example

```python
def test_my_feature(app, client, test_cache):
    """Test example using fixtures"""
    with app.app_context():
        # Test logic here
        response = client.get('/api/search?year=2015&make=Toyota&model=Camry')
        assert response.status_code == 200
```

## Test Data Requirements

### Minimum Data for Tests

- **Local Training**: Requires 50+ records for the same year/make/model
- **Global Training**: Uses all available test data
- **Hit Counter**: Tests with configurable thresholds
- **Cache Operations**: Uses temporary cache instances

### Creating Additional Test Data

```python
# Add more test vehicles in conftest.py _create_test_vehicles()
test_vehicles.append(Vehicle(
    vin="UNIQUE_VIN_HERE",
    year=2020,
    make="Tesla",
    model="Model3",
    listing_price=45000,
    listing_mileage=15000
))
```

## Performance Benchmarks

### Expected Response Times

- **API Requests**: < 5 seconds (integration test tolerance)
- **Local Training**: < 30 seconds with test data
- **Global Training**: Varies with data size
- **Cache Operations**: < 100ms

### Concurrent Request Testing

Tests verify the system can handle:
- 5 concurrent API requests
- Multiple background training jobs
- Concurrent cache operations

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

2. **Database Issues**
   ```bash
   # Tests use in-memory SQLite - no persistent database needed
   # Check that SQLAlchemy is properly configured
   ```

3. **Cache Errors**
   ```bash
   # Tests use SimpleCache - no Redis required
   # Check Flask-Caching installation
   ```

4. **Missing Dependencies**
   ```bash
   # Install all test dependencies
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```

### Debug Mode

```python
# Add to test for debugging
import pdb; pdb.set_trace()

# Or use pytest debugging
pytest --pdb tests/test_api_integration.py::test_specific_function
```

## Coverage Reports

When running with `--coverage`, reports are generated:

- **Terminal**: Summary coverage report
- **HTML**: Detailed coverage report in `htmlcov/`

```bash
# Generate coverage report
python run_tests.py --coverage

# View HTML report
open htmlcov/index.html
```

## Continuous Integration

### Test Commands for CI

```bash
# Basic test run
python run_tests.py --category quick

# Full test suite with coverage
python run_tests.py --coverage

# Performance/stress tests
python run_tests.py --category slow
```

### Test Markers

Tests are marked for easy filtering:

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (ML training, etc.)
- `@pytest.mark.api` - API-specific tests
- `@pytest.mark.cache` - Cache-related tests

## Best Practices

### Writing New Tests

1. **Use descriptive test names**: `test_vehicle_search_with_mileage_parameter`
2. **Test both success and failure cases**
3. **Use appropriate fixtures**: Don't reinvent test setup
4. **Mock external dependencies**: HTTP calls, file operations
5. **Assert on specific behaviors**: Not just "doesn't crash"
6. **Clean up resources**: Use fixtures that auto-cleanup

### Test Organization

1. **Group related tests in classes**
2. **Use setup/teardown methods when needed**
3. **Keep tests independent**: One test shouldn't depend on another
4. **Use meaningful assertions**: Assert on business logic, not implementation details

## Contributing

When adding new features, ensure:

1. **Integration tests cover the feature**
2. **Both success and error paths are tested**
3. **Tests pass in isolation and as part of the suite**
4. **Documentation is updated to reflect new test coverage**
