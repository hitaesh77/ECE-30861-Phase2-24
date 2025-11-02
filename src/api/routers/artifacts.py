import time
import uuid
from datetime import datetime

from fastapi import APIRouter, status, HTTPException
from pydantic import HttpUrl

from src.api.models.artifact import Artifact, ArtifactData, ArtifactType, ModelRating, Metric

# Create a FastAPI router instance for artifact-related endpoints
router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


def mock_metrics_calculation(url: HttpUrl) -> ModelRating:
    """
    MOCK FUNCTION: Simulates calling the refactored 'metrics.py' logic.
    In a real app, this would download the file from S3 and run the analysis.
    """
    print(f"MOCK: Running metrics analysis on content from {url}")
    time.sleep(1) # Simulate computation time

    # Return some mock but realistic data
    metrics = [
        Metric(name="net_score", value=0.85, unit="ratio"),
        Metric(name="size_mb", value=1024.5, unit="MB"),
        Metric(name="license_compliance", value=1.0, unit="ratio"),
    ]
    return ModelRating(metrics=metrics)


def mock_ingest_and_persist(data: ArtifactData, artifact_id: str) -> Artifact:
    """
    MOCK FUNCTION: Simulates the core data flow logic (S3 upload + DynamoDB write).

    1. Downloads content from data.url. (Skipped in mock)
    2. Uploads binary content to S3.
    3. Writes metadata to DynamoDB.
    """
    # 1. Simulate S3 Ingestion and Key generation
    s3_key = f"s3://my-registry-bucket/{data.artifact_type.value}/{artifact_id}.zip"

    # 2. Simulate Metrics Computation
    rating_data = mock_metrics_calculation(data.url)

    # 3. Simulate Persistence (DynamoDB write)
    # The artifact name is derived from the URL path if not provided
    derived_name = data.name if data.name else data.url.path.split('/')[-1]

    new_artifact = Artifact(
        artifact_id=artifact_id,
        artifact_type=data.artifact_type,
        name=derived_name,
        original_url=str(data.url),
        s3_key=s3_key,
        ingest_date=datetime.now(),
        rating_data=rating_data,
        lineage_graph={"nodes": 1, "edges": 0}, # Mock Lineage
        cost_data={"compute_usd": 0.05, "storage_usd": 0.001} # Mock Cost
    )
    print(f"MOCK: Saved artifact {artifact_id} metadata to DynamoDB.")
    return new_artifact


@router.post(
    "/model",
    response_model=Artifact,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a new model artifact (BASELINE)"
)
async def ingest_model(data: ArtifactData):
    """
    Receives a URL pointing to a model, triggers download, metrics computation,
    S3 storage, and DynamoDB metadata persistence.
    """
    if data.artifact_type != ArtifactType.model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Endpoint /model only accepts artifact_type='model', received '{data.artifact_type.value}'"
        )

    # Core Logic: Generate ID and perform ingestion
    artifact_id = str(uuid.uuid4().int)[:10] # Generate a 10-digit unique ID
    
    try:
        new_artifact = mock_ingest_and_persist(data, artifact_id)
        return new_artifact
    except Exception as e:
        # In a real app, handle specific S3/DynamoDB/Download errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed for ID {artifact_id}: {str(e)}"
        )
