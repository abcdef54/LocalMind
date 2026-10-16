"""Register implemented conversation and submission endpoints under the v1 router."""

from fastapi import APIRouter

from src.backend.api.v1.conversations import router as conversation_router
from src.backend.api.v1.submissions import router as submission_router
from src.backend.api.v1.indexing import router as indexing_router

router = APIRouter()
router.include_router(conversation_router)
router.include_router(submission_router)

router.include_router(indexing_router)
