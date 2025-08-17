from sqlalchemy import func, and_, between, or_
from sqlalchemy.orm import noload
from app.models.vehicle import Vehicle
from app import db
from typing import List, Optional, Iterable


class VehicleRepository:
    """Data access layer for vehicle operations"""

    @staticmethod
    def get_listings_with_filters(filters: dict, limit: int = 100) -> List[Vehicle]:
        """Get vehicle listings matching multiple filters"""
        query = db.session.query(Vehicle).filter(
            and_(
                Vehicle.year == filters["year"],
                Vehicle.make == filters["make"],
                Vehicle.model == filters["model"],
                Vehicle.listing_price.isnot(None)
            )
        )

        if filters.get("trim"):
            query = query.filter(Vehicle.trim == filters["trim"])
        if filters.get("color"):
            query = query.filter(Vehicle.exterior_color == filters["color"])
        if filters.get("dealer_state"):
            query = query.filter(Vehicle.dealer_state == filters["dealer_state"])
        if filters.get("mileage"):
            lower_bound = int(filters["mileage"] * 0.8)
            upper_bound = int(filters["mileage"] * 1.2)
            query = query.filter(between(Vehicle.listing_mileage, lower_bound, upper_bound))

        query = query.order_by(Vehicle.listing_mileage)
        
        if limit:
            query = query.limit(limit)

        return query.all()

    @staticmethod
    def get_average_price_with_filters(filters: dict) -> Optional[float]:
        """Calculate average price with multiple filters"""
        query = db.session.query(func.avg(Vehicle.listing_price)).filter(
            and_(
                Vehicle.year == filters["year"],
                Vehicle.make == filters["make"],
                Vehicle.model == filters["model"],
                Vehicle.listing_price.isnot(None),
            )
        )

        if filters.get("trim"):
            query = query.filter(Vehicle.trim == filters["trim"])
        if filters.get("color"):
            query = query.filter(Vehicle.exterior_color == filters["color"])
        if filters.get("dealer_state"):
            query = query.filter(Vehicle.dealer_state == filters["dealer_state"])
        if filters.get("mileage"):
            lower_bound = int(filters["mileage"] * 0.8)
            upper_bound = int(filters["mileage"] * 1.2)
            query = query.filter(
                between(Vehicle.listing_mileage, lower_bound, upper_bound)
            )

        result = query.scalar()
        return float(result) if result else None

    @staticmethod
    def get_makes_by_year(year: int) -> List[str]:
        """Get distinct makes for a given year"""
        return [
            row[0]
            for row in db.session.query(Vehicle.make)
            .filter(Vehicle.year == year)
            .distinct()
            .all()
        ]

    @staticmethod
    def get_models_by_make_year(year: int, make: str) -> List[str]:
        """Get distinct models for a given make and year"""
        return [
            row[0]
            for row in db.session.query(Vehicle.model)
            .filter(and_(Vehicle.year == year, Vehicle.make == make))
            .distinct()
            .all()
        ]

    @staticmethod
    def stream_training_rows(
        batch_size: int = 5000,
        year: Optional[int] = None,
        make: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Iterable[Vehicle]:
        """
        Stream Vehicle rows in DB-sized chunks for low-RAM training/export.

        Yields ORM Vehicle instances with the fields used by TrainingService:
        listing_price, year, make, model, trim, listing_mileage, dealer_state,
        exterior_color, style, driven_wheels, fuel_type, interior_color,
        used, certified, listing_status, last_seen_date.

        Usage:
            for v in VehicleRepository.stream_training_rows(batch_size=5000):
                ...

            for v in VehicleRepository.stream_training_rows(5000, 2015, "Toyota", "Camry"):
                ...
        """
        q = (
            db.session.query(Vehicle)
            .options(noload("*"))                 # avoid eager-loading relationships
            .filter(
                Vehicle.listing_price.isnot(None),
                Vehicle.year.isnot(None),
                Vehicle.make.isnot(None),
                Vehicle.model.isnot(None),
            )
        )

        if year is not None:
            q = q.filter(Vehicle.year == year)
        if make is not None:
            q = q.filter(Vehicle.make == make)
        if model is not None:
            q = q.filter(Vehicle.model == model)

        # Deterministic streaming helps with reproducibility
        q = (
            q.order_by(Vehicle.id.asc())
             .yield_per(batch_size)
             .enable_eagerloads(False)
             .execution_options(stream_results=True)
        )

        for v in q:
            yield v