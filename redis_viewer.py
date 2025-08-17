#!/usr/bin/env python3
"""
Redis Cache Viewer - View human-readable values from Flask-Caching Redis store
"""

import redis
import pickle
import json
from datetime import datetime

def view_redis_cache():
    """View all cached values in human-readable format"""
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        
        # Get all Flask-Cache keys
        keys = r.keys('flask_cache*')
        
        if not keys:
            print("No Flask-Cache keys found in Redis")
            return
        
        print("=" * 60)
        print("REDIS CACHE VIEWER - Human Readable Values")
        print("=" * 60)
        
        for key in keys:
            key_str = key.decode('utf-8')
            raw_value = r.get(key)
            
            print(f"\n🔑 Key: {key_str}")
            print("-" * 40)
            
            try:
                if raw_value and raw_value.startswith(b'!\x80\x05'):
                    # This is pickled data from Flask-Caching
                    # Skip the first byte (Flask-Caching prefix) and unpickle
                    pickle_data = raw_value[1:]  # Remove the '!' prefix
                    unpickled = pickle.loads(pickle_data)
                    
                    # Format the output nicely
                    if isinstance(unpickled, dict):
                        print(f"📄 Type: Dictionary")
                        print(f"📝 Value: {json.dumps(unpickled, indent=2, default=str)}")
                    elif isinstance(unpickled, (list, tuple)):
                        print(f"📄 Type: {type(unpickled).__name__}")
                        print(f"📝 Value: {unpickled}")
                    elif hasattr(unpickled, '__dict__'):
                        # Custom object (like ModelMeta)
                        print(f"📄 Type: {type(unpickled).__name__}")
                        print(f"📝 Value: {unpickled.__dict__}")
                    else:
                        print(f"📄 Type: {type(unpickled).__name__}")
                        print(f"📝 Value: {unpickled}")
                        
                else:
                    # Raw string or other data
                    print(f"📄 Type: Raw data")
                    print(f"📝 Value: {raw_value}")
                    
            except Exception as e:
                print(f"❌ Error unpickling: {e}")
                print(f"📄 Raw bytes: {raw_value[:100]}...")
                
            # Get TTL info
            ttl = r.ttl(key)
            if ttl > 0:
                print(f"⏰ TTL: {ttl} seconds")
            elif ttl == -1:
                print(f"⏰ TTL: No expiration")
            else:
                print(f"⏰ TTL: Expired")
                
        print("\n" + "=" * 60)
        
    except redis.ConnectionError:
        print("❌ Could not connect to Redis. Is Redis running?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    view_redis_cache()
