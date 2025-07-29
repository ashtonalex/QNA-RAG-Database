"""
Main FastAPI application entry point.
Includes API routers, CORS, and Celery integration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import documents
from celery_worker import test_celery
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code (initialize resources, DB connections, etc.)
    yield
    # Shutdown code (cleanup resources)


app = FastAPI(title="Document Processing API", lifespan=lifespan)

# CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(documents.router)

# Include RAG pipeline router
from app.api import rag_pipeline
app.include_router(rag_pipeline.router)


# Startup/shutdown events for resource management
# Remove @app.on_event("startup") and @app.on_event("shutdown")


@app.get("/test-celery")
def run_test_celery():
    task = test_celery.delay()
    return JSONResponse({"task_id": task.id})
