from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from src.api.routers import artifact_router
from src.api.models import Metric, ModelRating

#edit fastapi title and desc later, for now llm generated
app = FastAPI(
    title="ECE 461 - Trustworthy Model Registry API",
    description="Baseline API implementation for the Trustworthy Model Registry.",
    version="3.4.2",
    contact={
        "name": "Your Team Name",
    },
    # Disable default tags to use custom ones defined in the router
    openapi_tags=[{"name": "Artifacts", "description": "Operations on Models, Datasets, and Code."}]
)

# deal with cors if issues come up

# expose url

# Include the router for artifact-related endpoints
app.include_router(artifact_router.router)

# Run command (for local testing):
# uvicorn src.main:app --reload
