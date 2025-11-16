from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

from api.routers import artifacts

app = FastAPI(
    title="ECE 461 - Trustworthy Model Registry API",
    description="Baseline API implementation for the Trustworthy Model Registry.",
    version="3.4.2",
    contact={
        "Group": "Group 24",
    },
)

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router - NO PREFIX, paths defined in router
app.include_router(artifacts.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Trustworthy Model Registry API",
        "version": "3.4.2",
        "status": "operational",
        "aws_region": os.getenv('AWS_REGION', 'not configured')
    }