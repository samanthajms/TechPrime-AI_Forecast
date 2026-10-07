"""WSGI alias so Render's default start command (`gunicorn app:app`) works. The real service is forecast_service.py."""
from forecast_service import app  # noqa: F401
