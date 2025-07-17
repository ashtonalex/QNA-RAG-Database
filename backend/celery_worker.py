import os
import ssl
from celery import Celery
from dotenv import load_dotenv

"""
Celery worker entry point for background document processing tasks.
"""


load_dotenv()

print("Using Redis URL:", os.environ.get("REDIS_URL"))

celery_app = Celery(
    "worker",
    broker=os.environ.get("REDIS_URL"),
    backend=os.environ.get("REDIS_URL"),
)

# Secure rediss:// support for Upstash
redis_url = os.environ.get("REDIS_URL")
if redis_url and redis_url.startswith("rediss://"):
    import ssl

    celery_app.conf.broker_use_ssl = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
    celery_app.conf.redis_backend_use_ssl = {"ssl_cert_reqs": ssl.CERT_REQUIRED}

# Import tasks to register them with Celery
# from app.services.document_processor import test_celery


@celery_app.task(name="test_celery")
def test_celery():
    return "Celery is working!"


if __name__ == "__main__":
    celery_app.start()
