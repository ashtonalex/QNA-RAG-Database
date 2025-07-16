"""
Celery worker entry point for background document processing tasks.
"""

import os
import ssl
from celery import Celery

redis_url = os.getenv("REDIS_URL", "")
celery_app = Celery("document_processor", broker=redis_url, backend=redis_url)

# Secure rediss:// support for Upstash
if redis_url.startswith("rediss://"):
    celery_app.conf.broker_use_ssl = {"ssl_cert_reqs": ssl.CERT_NONE}
    celery_app.conf.redis_backend_use_ssl = {"ssl_cert_reqs": ssl.CERT_NONE}

# Import tasks to register them with Celery
# from app.services.document_processor import test_celery


@celery_app.task(name="test_celery")
def test_celery():
    return "Celery is working!"


if __name__ == "__main__":
    celery_app.start()
