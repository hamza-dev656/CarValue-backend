import sys
import os
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy.exc import IntegrityError
from app import create_app, db
from app.models.vehicle import Vehicle
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataImporter:
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize raw data"""
        numeric_cols = ["year", "listing_price", "listing_mileage"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        bool_cols = ["used", "certified"]
        for col in bool_cols:
            df[col] = df[col].astype(str).str.upper() == "TRUE"

        date_cols = ["first_seen_date", "last_seen_date", "dealer_vdp_last_seen_date"]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        
        return df.where(pd.notnull(df), None)


    @staticmethod
    def prepare_vehicle_dict(row: pd.Series) -> dict:
        """Convert DataFrame row to vehicle dictionary with robust type safety and length checks"""
        
        def safe_str(value, max_length=None, default=None):
            """Safely convert to string with length limit"""
            if pd.isna(value) or value is None:
                return default
            value = str(value)
            return value[:max_length] if max_length else value
        
        def safe_int(value, default=None):
            """Safely convert to integer"""
            try:
                return int(value) if pd.notna(value) else default
            except (ValueError, TypeError):
                return default
        
        def safe_float(value, default=None):
            """Safely convert to float"""
            try:
                return float(value) if pd.notna(value) else default
            except (ValueError, TypeError):
                return default
        
        def safe_bool(value, default=False):
            """Safely convert to boolean"""
            if pd.isna(value) or value is None:
                return default
            return bool(value)
        
        def safe_date(value, default=None):
            """Safely convert to datetime"""
            if pd.isna(value) or value is None:
                return default
            try:
                return value.to_pydatetime()
            except AttributeError:
                return default
            
        required = {
            'vin': safe_str(row.get('vin'), 17),
            'year': safe_int(row.get('year')),
            'make': safe_str(row.get('make'), 50),
            'model': safe_str(row.get('model'), 50)
        }
        
        if None in required.values():
            missing = [k for k,v in required.items() if v is None]
            logger.debug(f"Skipping VIN {row.get('vin')} - missing: {missing}")
            return None
        
        return {
            "vin": safe_str(row.get("vin"), 17),
            "year": safe_int(row.get("year")),
            "make": safe_str(row.get("make"), 50),
            "model": safe_str(row.get("model"), 50),
                        
            "trim": safe_str(row.get("trim"), 100),
            "dealer_name": safe_str(row.get("dealer_name"), 100),
            "dealer_street": safe_str(row.get("dealer_street"), 100),
            "dealer_city": safe_str(row.get("dealer_city"), 50),
            "dealer_state": safe_str(row.get("dealer_state"), 20),
            "dealer_zip": safe_str(row.get("dealer_zip"), 10),
            "listing_price": safe_float(row.get("listing_price")),
            "listing_mileage": safe_int(row.get("listing_mileage")),
            "used": safe_bool(row.get("used"), default=True),
            "certified": safe_bool(row.get("certified"), default=False),
            "style": safe_str(row.get("style"), 100),
            "driven_wheels": safe_str(row.get("driven_wheels"), 50),
            "engine": safe_str(row.get("engine"), 100),
            "fuel_type": safe_str(row.get("fuel_type"), 50),
            "exterior_color": safe_str(row.get("exterior_color"), 50),
            "interior_color": safe_str(row.get("interior_color"), 50),
            "seller_website": safe_str(row.get("seller_website"), 255),
            "first_seen_date": safe_date(row.get("first_seen_date")),
            "last_seen_date": safe_date(row.get("last_seen_date")),
            "dealer_vdp_last_seen_date": safe_date(row.get("dealer_vdp_last_seen_date")),
            "listing_status": safe_str(row.get("listing_status"), 50)
        }

    @staticmethod
    def import_from_txt(file_path: str, batch_size: int = 1000) -> dict:
        """Import data from pipe-delimited text file"""
        stats = {
            "total_rows": 0,
            "imported": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": datetime.now()
        }

        app = create_app()
        
        with app.app_context():
            try:
                db.session.execute(text("SELECT 1"))
                logger.info("Database connection successful")
                
                chunks = pd.read_csv(
                    file_path,
                    delimiter='|',
                    chunksize=batch_size,
                    encoding='utf-8',
                    on_bad_lines='warn'
                )

                for chunk in tqdm(chunks, desc="Importing batches"):
                    stats["total_rows"] += len(chunk)
                    cleaned_chunk = DataImporter.clean_data(chunk)
                    
                    vehicles = [
                        DataImporter.prepare_vehicle_dict(row) 
                        for _, row in cleaned_chunk.iterrows()
                        if pd.notna(row["vin"]) 
                    ]
                    
                    vehicles = [v for v in vehicles if v is not None]

                    try:
                        db.session.bulk_insert_mappings(Vehicle, vehicles)
                        db.session.commit()
                        stats["imported"] += len(vehicles)
                        logger.debug(f"Inserted {len(vehicles)} records")
                    except Exception as e:
                        db.session.rollback()
                        logger.error(f"Batch insert failed: {str(e)}")
                        stats["errors"] += len(vehicles)

            except Exception as e:
                logger.error(f"Import failed: {str(e)}")
                raise
            finally:
                stats["end_time"] = datetime.now()
                stats["duration"] = stats["end_time"] - stats["start_time"]

        return stats

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Import vehicle data from TXT file')
    parser.add_argument('--file-path', required=True, help='Path to the text file')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Records per batch (default: 1000)')
    
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        db.create_all()
        logger.info(f"Starting import from {args.file_path}")
        result = DataImporter.import_from_txt(args.file_path, args.batch_size)
        
        logger.info("\nImport Summary:")
        logger.info(f"Total rows processed: {result['total_rows']}")
        logger.info(f"Successfully imported: {result['imported']}")
        logger.info(f"Skipped (duplicates): {result['skipped']}")
        logger.info(f"Errors: {result['errors']}")
        logger.info(f"Duration: {result['duration']}")