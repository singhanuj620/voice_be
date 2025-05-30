from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import voice, health, reportfile
from routes.dbcheck import router as dbcheck_router

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice.router)
app.include_router(health.router)
app.include_router(reportfile.router)
app.include_router(dbcheck_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("index:app", host="127.0.0.1", port=8000, reload=True)
