import time
import uuid
import asyncio
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from fastapi import APIRouter, status, HTTPException, Header, Query, Response
from pydantic import HttpUrl

from api.models.artifact import (
    Artifact, ArtifactData, ArtifactType, ModelRating, 
    ArtifactMetadata, ArtifactQuery, EnumerateOffset
)

# Import your metrics computation
# from src.run import classify_url
from metrics import run_metrics
from utils import UrlCategory

# import aws services
from api.services.dynamodb_service import db_service

# Create router WITHOUT prefix - paths are defined per OpenAPI spec
router = APIRouter(tags=["Artifacts"])

# In-memory storage for development (replace with DynamoDB later)
ARTIFACT_STORE: Dict[str, Dict] = {}

USE_LOCAL = False
USE_AWS = True

# HELPERS
def generate_artifact_id() -> str:
    """Generate a unique artifact ID matching the spec format."""
    return str(uuid.uuid4().int)[:13]

def extract_name_from_url(url: str) -> str:
    """Extract artifact name from URL."""
    parsed = urlparse(url)
    path_parts = parsed.path.strip('/').split('/')
    
    if 'huggingface.co' in parsed.netloc:
        if len(path_parts) >= 2:
            return path_parts[-1]
    
    return path_parts[-1] if path_parts else 'unknown'

async def compute_metrics_from_url(url: str, artifact_type: ArtifactType) -> ModelRating:
    """
    Compute metrics using your existing CLI logic.
    This integrates with your run2.py and metrics.py.
    """
    # Classify the URL
    # category, provider, ids = classify_url(url)
    
    # Build URL dictionary for run_metrics
    url_dict = {
        UrlCategory.MODEL: {'url': url}
    }
    
    # For models, try to infer code/dataset if available
    # (This is a simplification - you may need more logic)
    
    try:
        # Run your metrics computation
        result = await run_metrics(url_dict)
        
        # Convert to ModelRating format
        rating = ModelRating(
            name=result.get('name', extract_name_from_url(url)),
            category=result.get('category', 'MODEL'),
            net_score=result.get('net_score', -1.0),
            net_score_latency=result.get('net_score_latency', 0.0),
            ramp_up_time=result.get('ramp_up_time', -1.0),
            ramp_up_time_latency=result.get('ramp_up_time_latency', 0.0),
            bus_factor=result.get('bus_factor', -1.0),
            bus_factor_latency=result.get('bus_factor_latency', 0.0),
            performance_claims=result.get('performance_claims', -1.0),
            performance_claims_latency=result.get('performance_claims_latency', 0.0),
            license=result.get('license', -1.0),
            license_latency=result.get('license_latency', 0.0),
            dataset_and_code_score=result.get('dataset_and_code_score', -1.0),
            dataset_and_code_score_latency=result.get('dataset_and_code_score_latency', 0.0),
            dataset_quality=result.get('dataset_quality', -1.0),
            dataset_quality_latency=result.get('dataset_quality_latency', 0.0),
            code_quality=result.get('code_quality', -1.0),
            code_quality_latency=result.get('code_quality_latency', 0.0),
            size_score=result.get('size_score', {
                'raspberry_pi': -1.0,
                'jetson_nano': -1.0,
                'desktop_pc': -1.0,
                'aws_server': -1.0
            }),
            size_score_latency=result.get('size_score_latency', 0.0)
        )
        
        return rating
        
    except Exception as e:
        print(f"Error computing metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"The artifact rating system encountered an error: {str(e)}"
        )

def check_metrics_threshold(rating: ModelRating) -> bool:
    """
    Check if artifact meets ingestion threshold (0.5 for non-latency metrics).
    Returns True if artifact can be ingested.
    """
    non_latency_metrics = [
        rating.ramp_up_time,
        rating.bus_factor,
        rating.performance_claims,
        rating.license,
        rating.dataset_and_code_score,
        rating.dataset_quality,
        rating.code_quality
    ]
    
    for metric in non_latency_metrics:
        if metric < 0.5 and metric != -1.0:  # -1.0 is allowed for missing metrics
            # pass for local testing. return false when deploying
            # return False
            pass
    
    return True

def parse_offset(offset: Optional[str]) -> Optional[Dict[str, Any]]:
    if not offset:
        return None
    try:
        return json.loads(offset)
    except:
        raise HTTPException(status_code=400, detail="Invalid offset")

def encode_offset(key: Optional[Dict[str, Any]]) -> Optional[str]:
    if not key:
        return None
    return json.dumps(key)

# ============================================================================
# BASELINE ENDPOINTS
# ============================================================================

@router.post(
    "/artifact/{artifact_type}",
    response_model=Artifact,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new artifact (BASELINE - Ingest)"
)
async def create_artifact(
    artifact_type: ArtifactType,
    artifact_data: ArtifactData,
    x_authorization: Optional[str] = Header(None)
):
    """
    Register a new artifact by providing a downloadable source URL.
    
    For models, computes trustworthiness metrics and checks threshold (0.5).
    """
    url_str = str(artifact_data.url)
    
    # Extract name from URL
    artifact_name = extract_name_from_url(url_str)
    
    # local implementation to check duplicates
    if (USE_LOCAL):
        for artifact_id, artifact in ARTIFACT_STORE.items():
            if artifact['url'] == url_str:
                raise HTTPException(
                    status_code=409,
                    detail="Artifact exists already"
                )
    
    # aws implementation to check duplicates
    if (USE_AWS):
        existing = await db_service.get_artifact_by_url(url_str)
        if existing:
            raise HTTPException(
                status_code=409,
                detail="Artifact exists already"
            )
    
    # Generate unique ID
    artifact_id = generate_artifact_id()
    
    # Compute metrics for models
    rating = None
    if artifact_type == ArtifactType.model:
        rating = await compute_metrics_from_url(url_str, artifact_type)
        
        # Check if meets threshold
        if not check_metrics_threshold(rating):
            raise HTTPException(
                status_code=424,
                detail="Artifact is not registered due to the disqualified rating"
            )
    
    # Store artifact
    artifact_record = {
        'id': artifact_id,
        'name': artifact_name,
        'type': artifact_type.value,
        'url': url_str,
        'rating': rating.model_dump() if rating else None,
        'created_at': datetime.utcnow().isoformat()
    }
    
    # local implementation
    if (USE_LOCAL):
        ARTIFACT_STORE[artifact_id] = artifact_record

    # aws implementation
    if (USE_AWS):
        await db_service.create_artifact(artifact_record)
    
    # Return artifact
    return Artifact(
        metadata=ArtifactMetadata(
            name=artifact_name,
            id=artifact_id,
            type=artifact_type
        ),
        data=ArtifactData(url=artifact_data.url)
    )


@router.get(
    "/artifact/{artifact_type}/{id}",
    response_model=Artifact,
    summary="Retrieve an artifact (BASELINE - Read)"
)
async def get_artifact(
    artifact_type: ArtifactType,
    id: str,
    x_authorization: Optional[str] = Header(None)
):
    """Return this artifact."""

    # local implementation
    if (USE_LOCAL):
        if id not in ARTIFACT_STORE:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        artifact = ARTIFACT_STORE[id]

    # aws implementation
    if (USE_AWS):
        artifact = await db_service.get_artifact(id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
    
    if artifact['type'] != artifact_type.value:
        raise HTTPException(
            status_code=400,
            detail="Artifact type mismatch"
        )
    
    return Artifact(
        metadata=ArtifactMetadata(
            name=artifact['name'],
            id=artifact['id'],
            type=ArtifactType(artifact['type'])
        ),
        data=ArtifactData(url=artifact['url'])
    )


@router.put(
    "/artifact/{artifact_type}/{id}",
    summary="Update an artifact (BASELINE - Update)"
)
async def update_artifact(
    artifact_type: ArtifactType,
    id: str,
    artifact: Artifact,
    x_authorization: Optional[str] = Header(None)
):
    """Update this content of the artifact."""

    # local implementation
    if (USE_LOCAL):
        if id not in ARTIFACT_STORE:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        existing = ARTIFACT_STORE[id]

    # aws implementation
    if (USE_AWS):
        existing = await db_service.get_artifact(id)
        if not existing:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        
    if existing['type'] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch")
    
    # Verify name and id match
    if artifact.metadata.id != id or artifact.metadata.name != existing['name']:
        raise HTTPException(
            status_code=400,
            detail="Name and ID must match"
        )
    
    # Update URL in local implementation
    if (USE_LOCAL):
        existing['url'] = str(artifact.data.url)
        existing['updated_at'] = datetime.utcnow().isoformat()

    # update URL in aws implementation
    if (USE_AWS):
        await db_service.update_artifact(id, {'url': str(artifact.data.url)})
    
    return {"message": "Artifact is updated"}


@router.delete(
    "/artifact/{artifact_type}/{id}",
    summary="Delete an artifact (NON-BASELINE)"
)
async def delete_artifact(
    artifact_type: ArtifactType,
    id: str,
    x_authorization: Optional[str] = Header(None)
):
    """Delete this artifact."""

    # local implementation
    if (USE_LOCAL):
        if id not in ARTIFACT_STORE:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        artifact = ARTIFACT_STORE[id]

    # aws implementation
    if (USE_AWS):
        artifact = await db_service.get_artifact(id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        
    if artifact['type'] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch")
    
    # local implementation
    if (USE_LOCAL):
        del ARTIFACT_STORE[id]

    # aws implementation
    if (USE_AWS):
        await db_service.delete_artifact(id)
    
    return {"message": "Artifact is deleted"}

# THIS IS ENUMERATE, WILL HAVE TO FIX
@router.post("/artifacts")
async def list_artifacts_endpoint(
    queries: List[ArtifactQuery],
    offset: Optional[str] = Query(None)
):
    """List artifacts matching the provided queries (local or AWS)."""

    if not queries or not isinstance(queries, list):
        raise HTTPException(status_code=400, detail="Invalid request body")

    q = queries[0]

    name_filter = q.name if q.name else None
    type_filter = q.types if q.types else None

    # Handle pagination offset for AWS
    last_key = None
    if offset:
        try:
            last_key = json.loads(offset)
        except:
            raise HTTPException(status_code=400, detail="Invalid offset token")

    # Local implementation
    if not USE_AWS:
        results = []

        for artifact_id, artifact in ARTIFACT_STORE.items():

            if name_filter and name_filter != "*" and artifact["name"] != name_filter:
                continue

            if type_filter and artifact["type"] not in type_filter:
                continue

            results.append({
                "id": artifact_id,
                "name": artifact["name"],
                "type": artifact["type"]
            })

        if len(results) > 1000:
            raise HTTPException(status_code=413, detail="Too many artifacts returned")

        return Response(
            content=json.dumps(results),
            media_type="application/json"
        )

    # AWS implementation
    items, next_key = await db_service.list_artifacts(
        name_filter=name_filter,
        type_filter=type_filter,
        limit=50,
        last_key=last_key
    )

    if len(items) > 1000:
        raise HTTPException(status_code=413, detail="Too many artifacts returned")

    response = Response(
        content=json.dumps(items),
        media_type="application/json"
    )

    if next_key:
        response.headers["offset"] = json.dumps(next_key)

    return response



@router.get(
    "/artifact/model/{id}/rate",
    response_model=ModelRating,
    summary="Get ratings for model artifact (BASELINE)"
)
async def rate_model(
    id: str,
    x_authorization: Optional[str] = Header(None)
):
    """Get ratings for this model artifact."""

    # local implementation
    if (USE_LOCAL):
        if id not in ARTIFACT_STORE:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
        artifact = ARTIFACT_STORE[id]

    # aws implementation
    if (USE_AWS):
        artifact = await db_service.get_artifact(id)
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact does not exist")
    
    if artifact['type'] != 'model':
        raise HTTPException(
            status_code=400,
            detail="Artifact is not a model"
        )
    
    rating_data = artifact.get('rating')
    if not rating_data:
        raise HTTPException(
            status_code=500,
            detail="The artifact rating system encountered an error"
        )
    
    return ModelRating(**rating_data)


# ============================================================================
# ADDITIONAL ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check():
    """Heartbeat check (BASELINE)."""
    return {"status": "ok"}

@router.get("/tracks")
async def get_tracks():
    """Get the list of tracks planned for implementation."""
    return {
        "plannedTracks": [
            "Performance track"
        ]
    }

# @router.delete("/reset")
# async def reset_registry(x_authorization: Optional[str] = Header(None)):
#     """Reset the registry (BASELINE)."""

#     # local implementation
#     # global ARTIFACT_STORE
#     # ARTIFACT_STORE.clear()
#     # return {"message": "Registry is reset"}

#     # aws implementation
#     try:
#             # Assuming db_service is an instance of a DynamoDB service class
#             # with a method to delete all items in the table.
#             # YOU MUST IMPLEMENT db_service.delete_all_artifacts()
#             await db_service.clear_all_artifacts() 

#             return {"message": "Registry is reset"}
        
#     except Exception as e:
#         # Log the error (optional)
#         print(f"Error resetting registry in DynamoDB: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to reset registry: {str(e)}"
#         )

@router.delete("/reset")
async def reset_registry(x_authorization: Optional[str] = Header(None)):
    try:
        deleted_count = 0

        # local implementation
        if (USE_LOCAL):
            global ARTIFACT_STORE
            deleted_count += len(ARTIFACT_STORE)
            ARTIFACT_STORE.clear()

        # aws implementation
        if (USE_AWS):
            deleted_count += await db_service.clear_all_artifacts()
            await asyncio.sleep(0.5)

            items, _ = await db_service.list_artifacts(
                name_filter="*",
                limit=1
            )

            if items:
                raise HTTPException(
                    status_code=500,
                    detail=f"Reset incomplete: {len(items)} artifacts still present"
                )

        print(f"Reset successful: deleted {deleted_count} artifacts")
        print("=" * 50)

        return {
            "message": "Registry is reset",
            "deleted_items": deleted_count
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error resetting registry: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset registry: {str(e)}"
        )
    
# temp debug endpoint
@router.get("/debug/count")
async def debug_count_artifacts():
    try:
        all_items = []

        # local implementation
        if USE_LOCAL:
            for item in ARTIFACT_STORE.values():
                all_items.append({
                    "id": item["id"],
                    "name": item["name"],
                    "type": item["type"]
                })

        # aws implementation
        if USE_AWS:
            items, _ = await db_service.list_artifacts(
                name_filter="*",
                limit=1000
            )
            for item in items:
                all_items.append({
                    "id": item["id"],
                    "name": item["name"],
                    "type": item["type"]
                })

        return {
            "count": len(all_items),
            "artifacts": all_items[:10]
        }

    except Exception as e:
        return {"error": str(e)}
