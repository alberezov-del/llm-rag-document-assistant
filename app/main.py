from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A local RAG assistant for PDF, TXT, and Markdown documents.",
)
app.include_router(router, prefix=settings.api_prefix)

