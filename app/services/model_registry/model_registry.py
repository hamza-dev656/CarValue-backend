# app/services/model_registry.py
"""
Model registry for managing ML model metadata and artifacts.
Handles CatBoost model storage, loading, and metadata management.
"""

import os
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional

from app.config.model_config import ARTIFACT_DIR, LOCAL_MODEL_TTL_SEC, GLOBAL_MODEL_TTL_SEC


@dataclass
class ModelMeta:
    """Metadata for a trained ML model"""
    kind: str                    # "global" | "local"
    segment: Optional[str]       # e.g. "2015:Toyota:Camry" for local models, None for global
    data_version: str           # data version used for training
    n_train: int                # number of training samples
    rmse_pct: float            # RMSE percentage on validation set
    path: str                  # file path to model artifact
    created_at: float          # timestamp when model was created
    model_type: str = "catboost_cbm"  # model format type
    sha256: Optional[str] = None       # file integrity hash
    size_bytes: Optional[int] = None   # file size in bytes

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelMeta':
        """Create instance from dictionary"""
        return cls(**data)


class ModelArtifactManager:
    """Handles physical model file operations"""
    
    @staticmethod
    def save_catboost_model(model, filename: str) -> tuple[str, str, int]:
        """
        Save CatBoost model to disk and return metadata.
        
        Returns:
            Tuple of (file_path, sha256_hash, file_size)
        """
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        path = os.path.join(ARTIFACT_DIR, filename)
        
        # Save model
        model.save_model(path)
        
        # Compute integrity information
        sha256_hash = ModelArtifactManager._compute_file_hash(path)
        file_size = os.path.getsize(path)
        
        return path, sha256_hash, file_size
    
    @staticmethod
    def load_catboost_model(path: str):
        """Load CatBoost model from disk"""
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(path)
        return model
    
    @staticmethod
    def _compute_file_hash(path: str) -> str:
        """Compute SHA256 hash of file"""
        hash_obj = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()


class ModelMetadataRegistry:
    """Manages model metadata storage and retrieval using cache/Redis"""
    
    def __init__(self, cache_store):
        """
        Initialize with cache store (Flask-Caching or Redis-like interface)
        
        Args:
            cache_store: Object with get/set methods for caching
        """
        self.cache = cache_store
    
    def register_model(self, meta: ModelMeta) -> None:
        """Register a model's metadata with appropriate TTL"""
        key = self._get_cache_key(meta.kind, meta.segment)
        
        # Set TTL based on model type
        if meta.kind == "local":
            ttl = LOCAL_MODEL_TTL_SEC
        elif meta.kind == "global":
            ttl = GLOBAL_MODEL_TTL_SEC
        else:
            ttl = LOCAL_MODEL_TTL_SEC  # Default fallback
            
        self.cache.set(key, json.dumps(meta.to_dict()), timeout=ttl)
    
    def get_global_model_meta(self) -> Optional[ModelMeta]:
        """Get metadata for the global model"""
        key = self._get_cache_key("global", None)
        raw_data = self.cache.get(key)
        return ModelMeta.from_dict(json.loads(raw_data)) if raw_data else None
    
    def get_local_model_meta(self, year: int, make: str, model_name: str) -> Optional[ModelMeta]:
        """Get metadata for a local model"""
        segment = f"{year}:{make}:{model_name}"
        key = self._get_cache_key("local", segment)
        raw_data = self.cache.get(key)
        return ModelMeta.from_dict(json.loads(raw_data)) if raw_data else None
    
    @staticmethod
    def _get_cache_key(kind: str, segment: Optional[str]) -> str:
        """Generate cache key for model metadata"""
        if kind == "global":
            return "model_registry:global"
        elif kind == "local" and segment:
            return f"model_registry:local:{segment}"
        else:
            raise ValueError(f"Invalid kind/segment combination: {kind}/{segment}")


class ModelRegistry:
    """
    Unified interface for model management.
    Combines metadata registry and artifact management.
    """
    
    def __init__(self, cache_store):
        """Initialize with cache store"""
        self.metadata = ModelMetadataRegistry(cache_store)
        self.artifacts = ModelArtifactManager()
    
    # High-level model loading
    def get_local_model(self, year: int, make: str, model_name: str) -> tuple[Optional[ModelMeta], Optional[object]]:
        """
        Get local model metadata and loaded model.
        
        Returns:
            Tuple of (metadata, loaded_model) or (None, None) if not found
        """
        meta = self.metadata.get_local_model_meta(year, make, model_name)
        
        if meta and self._model_file_exists(meta.path):
            model = self.artifacts.load_catboost_model(meta.path)
            return meta, model
            
        return None, None
    
    def get_global_model(self) -> tuple[Optional[ModelMeta], Optional[object]]:
        """
        Get global model metadata and loaded model.
        
        Returns:
            Tuple of (metadata, loaded_model) or (None, None) if not found
        """
        meta = self.metadata.get_global_model_meta()
        
        if meta and self._model_file_exists(meta.path):
            model = self.artifacts.load_catboost_model(meta.path)
            return meta, model
            
        return None, None
    
    def _model_file_exists(self, path: str) -> bool:
        """Check if model file exists and is readable"""
        try:
            import os
            return os.path.exists(path) and os.path.getsize(path) > 0
        except Exception:
            return False
    

    
    # Convenience methods for backward compatibility
    def get_global_meta(self) -> Optional[ModelMeta]:
        """Get global model metadata only"""
        return self.metadata.get_global_model_meta()
    
    def get_local_meta(self, year: int, make: str, model_name: str) -> Optional[ModelMeta]:
        """Get local model metadata only"""
        return self.metadata.get_local_model_meta(year, make, model_name)
    
    def save_catboost(self, model, filename: str) -> tuple[str, str, int]:
        """Save CatBoost model and return metadata"""
        return self.artifacts.save_catboost_model(model, filename)
    
    def load_catboost(self, path: str):
        """Load CatBoost model from path"""
        return self.artifacts.load_catboost_model(path)
    
    def register_model(self, meta: ModelMeta) -> None:
        """Register model metadata"""
        self.metadata.register_model(meta)
    
    def cleanup_expired_models(self) -> dict:
        """
        Clean up model files that no longer have metadata in cache (expired TTL).
        
        Returns:
            Dict with cleanup statistics
        """
        import glob
        import logging
        
        log = logging.getLogger("model_cleanup")
        stats = {"removed_files": 0, "total_checked": 0, "errors": 0, "size_freed_mb": 0.0}
        
        try:
            # Find all local model files
            local_pattern = os.path.join(ARTIFACT_DIR, "local_*.cbm")
            local_files = glob.glob(local_pattern)
            
            for file_path in local_files:
                stats["total_checked"] += 1
                
                try:
                    # Extract year, make, model from filename
                    # Example: local_2015_Toyota_Camry_v00000000.cbm
                    filename = os.path.basename(file_path)
                    parts = filename.replace("local_", "").replace(".cbm", "").split("_")
                    
                    if len(parts) >= 3:
                        year, make, model = parts[0], parts[1], parts[2]
                        
                        # Check if metadata still exists in cache
                        meta = self.metadata.get_local_model_meta(int(year), make, model)
                        
                        if not meta:
                            # Metadata expired, remove the file
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            stats["removed_files"] += 1
                            stats["size_freed_mb"] += file_size / (1024 * 1024)
                            log.info(f"Removed expired model file: {filename} ({file_size/1e6:.1f}MB)")
                        
                except Exception as e:
                    stats["errors"] += 1
                    log.error(f"Error processing {file_path}: {e}")
                    
        except Exception as e:
            log.error(f"Cleanup failed: {e}")
            
        if stats["removed_files"] > 0:
            log.info(f"Cleanup completed: {stats['removed_files']} files removed, {stats['size_freed_mb']:.1f}MB freed")
        
        return stats
    
    def schedule_periodic_cleanup(self, interval_hours: int = 24):
        """
        Schedule periodic cleanup of expired model files.
        
        Args:
            interval_hours: How often to run cleanup (default: 24 hours)
        """
        import threading
        import time
        import logging
        
        log = logging.getLogger("model_cleanup")
        
        def cleanup_worker():
            while True:
                try:
                    time.sleep(interval_hours * 3600)  # Convert hours to seconds
                    log.info("Starting scheduled model cleanup...")
                    stats = self.cleanup_expired_models()
                    log.info(f"Scheduled cleanup completed: {stats}")
                except Exception as e:
                    log.error(f"Scheduled cleanup failed: {e}")
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        log.info(f"Scheduled model cleanup every {interval_hours} hours")
    
        """Legacy method for compatibility"""
        return f"model_registry:local:{year}:{make}:{model_name}"