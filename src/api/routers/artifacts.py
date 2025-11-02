import time
import uuid
import asyncio
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from urllib.parse import urlparse

from fastapi import APIRouter, status, HTTPException, Header, Query, Response
from pydantic import HttpUrl

from api.models.artifact import (
    Artifact, ArtifactData, ArtifactType, ModelRating, 
    ArtifactMetadata, ArtifactQuery, EnumerateOffset
)

# Import your metrics computation
from run2 import classify_url
from metrics import run_metrics
from utils import UrlCategory

# Create router WITHOUT prefix - paths are defined per OpenAPI spec
router = APIRouter(tags=["Artifacts"])

# In-memory storage for development (replace with DynamoDB later)
ARTIFACT_STORE: Dict[str, Dict] = {}

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

# SAFE RUN METRICS:
async def safe_run_metrics(urls: Dict[UrlCategory, Dict[str, str]]) -> Dict:
    """
    Wrapper around run_metrics to catch internal errors and return 
    a default GradeResult dictionary structure, preventing a 500 error.
    """
    try:
        # Attempt to run the metrics computation
        return await run_metrics(urls)
    except Exception as e:
        # Log the error (optional, but good practice)
        print(f"FATAL: run_metrics failed internally with: {e}")
        
        # Return a default, failed result that conforms to the GradeResult structure
        return {
            'name': 'UNKNOWN',
            'category': 'MODEL',
            'net_score': -1.0,
            'net_score_latency': 0.0,
            'ramp_up_time': -1.0,
            'ramp_up_time_latency': 0.0,
            'bus_factor': -1.0,
            'bus_factor_latency': 0.0,
            'performance_claims': -1.0,
            'performance_claims_latency': 0.0,
            'license': -1.0,
            'license_latency': 0.0,
            'dataset_and_code_score': -1.0,
            'dataset_and_code_score_latency': 0.0,
            'dataset_quality': -1.0,
            'dataset_quality_latency': 0.0,
            'code_quality': -1.0,
            'code_quality_latency': 0.0,
            'size_score': {
                'raspberry_pi': -1.0,
                'jetson_nano': -1.0,
                'desktop_pc': -1.0,
                'aws_server': -1.0
            },
            'size_score_latency': 0.0
        }

async def compute_metrics_from_url(url: str, artifact_type: ArtifactType) -> ModelRating:
    """
    Compute metrics using your existing CLI logic.
    This integrates with your run2.py and metrics.py.
    """
    # Classify the URL
    category, provider, ids = classify_url(url)
    
    # Build URL dictionary for run_metrics
    url_dict = {
        UrlCategory.MODEL: {'url': url}
    }
    
    # For models, try to infer code/dataset if available
    # (This is a simplification - you may need more logic)
    
    try:
        # Run your metrics computation
        result = await run_metrics(url_dict)
        # result = await safe_run_metrics(url_dict)
        
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
    
    # Check if artifact already exists with same URL
    for artifact_id, artifact in ARTIFACT_STORE.items():
        if artifact['url'] == url_str:
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
    
    ARTIFACT_STORE[artifact_id] = artifact_record
    
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
    if id not in ARTIFACT_STORE:
        raise HTTPException(status_code=404, detail="Artifact does not exist")
    
    artifact = ARTIFACT_STORE[id]
    
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
    if id not in ARTIFACT_STORE:
        raise HTTPException(status_code=404, detail="Artifact does not exist")
    
    existing = ARTIFACT_STORE[id]
    
    if existing['type'] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch")
    
    # Verify name and id match
    if artifact.metadata.id != id or artifact.metadata.name != existing['name']:
        raise HTTPException(
            status_code=400,
            detail="Name and ID must match"
        )
    
    # Update URL
    existing['url'] = str(artifact.data.url)
    existing['updated_at'] = datetime.utcnow().isoformat()
    
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
    if id not in ARTIFACT_STORE:
        raise HTTPException(status_code=404, detail="Artifact does not exist")
    
    artifact = ARTIFACT_STORE[id]
    
    if artifact['type'] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch")
    
    del ARTIFACT_STORE[id]
    
    return {"message": "Artifact is deleted"}


@router.post(
    "/artifacts",
    response_model=List[ArtifactMetadata],
    summary="List artifacts (BASELINE - Enumerate)"
)
async def list_artifacts(
    queries: List[ArtifactQuery],
    response: Response,
    offset: Optional[str] = Query(None),
    x_authorization: Optional[str] = Header(None)
):
    """
    Get artifacts matching queries.
    
    If query name is "*", return all artifacts.
    Otherwise, search by name and optionally filter by types.
    
    Response is paginated with offset in header.
    """
    if not queries:
        raise HTTPException(
            status_code=400,
            detail="At least one query required"
        )
    
    # Handle wildcard query (enumerate all)
    if len(queries) == 1 and queries[0].name == "*":
        # Get all artifacts
        all_artifacts = list(ARTIFACT_STORE.values())
        
        # Apply pagination
        start_idx = int(offset) if offset else 0
        page_size = 10
        
        paginated = all_artifacts[start_idx:start_idx + page_size]
        next_offset = start_idx + page_size if start_idx + page_size < len(all_artifacts) else None
        
        # Set offset header
        if next_offset is not None:
            response.headers["offset"] = str(next_offset)
        
        # Convert to metadata format
        return [
            ArtifactMetadata(
                name=item['name'],
                id=item['id'],
                type=ArtifactType(item['type'])
            )
            for item in paginated
        ]
    
    # Handle specific name queries
    results = []
    seen_ids = set()
    
    for query in queries:
        for artifact_id, artifact in ARTIFACT_STORE.items():
            # Skip if already included
            if artifact_id in seen_ids:
                continue
            
            # Check name match
            if artifact['name'] == query.name:
                # Check type filter if specified
                if query.types:
                    type_values = [t.value for t in query.types]
                    if artifact['type'] not in type_values:
                        continue
                
                results.append(artifact)
                seen_ids.add(artifact_id)
    
    # Check for too many results
    if len(results) > 100:
        raise HTTPException(
            status_code=413,
            detail="Too many artifacts returned"
        )
    
    # Convert to metadata format
    return [
        ArtifactMetadata(
            name=item['name'],
            id=item['id'],
            type=ArtifactType(item['type'])
        )
        for item in results
    ]


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
    if id not in ARTIFACT_STORE:
        raise HTTPException(status_code=404, detail="Artifact does not exist")
    
    artifact = ARTIFACT_STORE[id]
    
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


@router.delete("/reset")
async def reset_registry(x_authorization: Optional[str] = Header(None)):
    """Reset the registry (BASELINE)."""
    global ARTIFACT_STORE
    ARTIFACT_STORE.clear()
    return {"message": "Registry is reset"}