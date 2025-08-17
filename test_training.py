#!/usr/bin/env python3
"""
Direct test of local training functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import create_app
from app.services.training_service import TrainingService

def test_local_training():
    """Test local training directly"""
    app = create_app()
    
    with app.app_context():
        print("🧪 Testing local training for 2015 Toyota Camry...")
        
        try:
            result = TrainingService.train_local_from_db(2015, "Toyota", "Camry", "test_v1.0")
            
            if result:
                print(f"✅ Training successful!")
                print(f"   📈 RMSE: {result.rmse_pct:.2f}%")
                print(f"   📊 Training samples: {result.n_train}")
                print(f"   💾 Model saved to: {result.path}")
            else:
                print("❌ Training returned None")
                
        except Exception as e:
            print(f"❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_local_training()


