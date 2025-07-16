"""
Celery worker entry point for background document processing tasks.
"""

from celery import Celery

# Configure Celery broker and backend (adjust as needed)
celery_app = Celery(
    "document_processor",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

# Import tasks to register them with Celery
# from app.services.document_processor import test_celery


@celery_app.task(name="test_celery")
def test_celery():
    return "Celery is working!"


if __name__ == "__main__":
    celery_app.start()
