from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def get_users():
    return {"status": 200, "message": "OK"}
