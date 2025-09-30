# api/temp_main.py - Temporary version without database for API key generation

import secrets
import hashlib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Attrahere ML Code Analysis API (Temp)",
    version="1.0.0",
    description="Temporary version for API key generation"
)

class CreateUserRequest(BaseModel):
    email: str
    name: str
    organization_slug: Optional[str] = "default"

class CreateUserResponse(BaseModel):
    user_id: str
    email: str
    name: str
    api_key: str
    organization: str

@app.get("/health")
def read_health():
    """Health check endpoint."""
    return {"status": "ok", "message": "Temporary API for key generation"}

@app.post("/api/v1/users", response_model=CreateUserResponse)
def create_user_temp(request: CreateUserRequest):
    """
    Temporary endpoint to create API keys without database.
    """
    # Generate API key
    api_key = f"ak_{secrets.token_urlsafe(40)}"
    
    # Generate a temporary user ID
    user_id = secrets.token_urlsafe(16)
    
    return CreateUserResponse(
        user_id=user_id,
        email=request.email,
        name=request.name,
        api_key=api_key,
        organization="Default Organization"
    )

@app.get("/")
def root():
    """Root endpoint with instructions."""
    return {
        "message": "Temporary Attrahere API for generating API keys",
        "instructions": "POST to /api/v1/users with email and name to generate an API key",
        "example": {
            "email": "user@example.com",
            "name": "Test User"
        }
    }