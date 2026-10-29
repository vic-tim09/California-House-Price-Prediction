"""Pydantic schemas validate incoming API data."""

from pydantic import BaseModel, ConfigDict, Field


class HouseFeatures(BaseModel):
    """One California census block group's housing features."""

    # Extra keys are rejected so misspelled feature names don't go unnoticed.
    model_config = ConfigDict(extra="forbid")

    MedInc: float = Field(gt=0, le=20, description="Median income, in tens of thousands of USD")
    HouseAge: float = Field(gt=0, le=100, description="Median house age in years")
    AveRooms: float = Field(gt=0, le=100, description="Average rooms per household")
    AveBedrms: float = Field(gt=0, le=25, description="Average bedrooms per household")
    Population: float = Field(gt=0, le=100000, description="Block-group population")
    AveOccup: float = Field(gt=0, le=1000, description="Average household occupancy")
    Latitude: float = Field(ge=32, le=42, description="Latitude in California dataset range")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude in California dataset range")


class PredictionResponse(BaseModel):
    """Consistent response returned for a single prediction."""

    estimated_median_house_value_usd: float
    currency: str = "USD"
    target_description: str = "Estimated median house value for a census block group"
    model: str = "RandomForestRegressor"


class BatchPredictionRequest(BaseModel):
    """A bounded batch prevents accidentally huge requests."""

    houses: list[HouseFeatures] = Field(min_length=1, max_length=50)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
