web: uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 4
worker: celery -A app.worker worker --loglevel=info --concurrency=4
