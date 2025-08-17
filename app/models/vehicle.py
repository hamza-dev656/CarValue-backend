from datetime import datetime
from sqlalchemy import Column, Integer, String, Numeric, Boolean, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from app import db

Base = declarative_base()

class Vehicle(db.Model):
    __tablename__ = 'vehicles'

    id = Column(Integer, primary_key=True)
    vin = Column(String(17), unique=True, nullable=False)
    year = Column(Integer, nullable=False, index=True)
    make = Column(String(50), nullable=False, index=True)
    model = Column(String(50), nullable=False, index=True)
    trim = Column(String(100))
    dealer_name = Column(String(100))
    dealer_street = Column(String(100))
    dealer_city = Column(String(50))
    dealer_state = Column(String(20))
    dealer_zip = Column(String(10))
    listing_price = Column(Numeric(10, 2))
    listing_mileage = Column(Integer)
    used = Column(Boolean, default=True)
    certified = Column(Boolean, default=False)
    style = Column(String(100))
    driven_wheels = Column(String(50))
    engine = Column(String(100))
    fuel_type = Column(String(50))
    exterior_color = Column(String(50))
    interior_color = Column(String(50))
    seller_website = Column(String(255))
    first_seen_date = Column(DateTime)
    last_seen_date = Column(DateTime)
    dealer_vdp_last_seen_date = Column(DateTime)
    listing_status = Column(String(50))

    __table_args__ = (
        Index('idx_year_make_model', 'year', 'make', 'model'),
        Index('idx_make_model_year', 'make', 'model', 'year'),
        Index('idx_price_mileage', 'listing_price', 'listing_mileage'),
    )

    def to_dict(self):
      """Convert model instance to dictionary"""
      return {
          'id': self.id,
          'vin': self.vin,
          'year': self.year,
          'make': self.make,
          'model': self.model,
          'trim': self.trim,
          'dealer_name': self.dealer_name,
          'dealer_location': f"{self.dealer_city}, {self.dealer_state}",
          'price': float(self.listing_price) if self.listing_price else None,
          'mileage': self.listing_mileage,
          'used': self.used,
          'certified': self.certified,
          'style': self.style,
          'driven_wheels': self.driven_wheels,
          'engine': self.engine,
          'fuel_type': self.fuel_type,
          'exterior_color': self.exterior_color,
          'interior_color': self.interior_color
      }