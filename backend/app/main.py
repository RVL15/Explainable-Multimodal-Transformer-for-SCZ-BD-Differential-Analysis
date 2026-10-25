from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.eeg.synthetic import populate_demo_cohort
from app.routers import health, datasets, signals, training, evaluation, xai, reports, demo, multimodal, history

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB tables
    init_db()
    # Check if demo cohort exists, if not generate initial demo subjects for immediate demonstration
    from app.database import get_db
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM subjects WHERE dataset_id = 'demo'")
        if cursor.fetchone()[0] == 0:
            populate_demo_cohort(num_scz=16, num_bd=16, num_hc=10)
    yield

app = FastAPI(
    title="Explainable Multimodal Transformer for SCZ-BD Differential Analysis",
    description="Research prototype exploring deep learning differential analysis of Schizophrenia (SCZ) vs Bipolar Disorder (BD) from EEG signals.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(signals.router)
app.include_router(training.router)
app.include_router(evaluation.router)
app.include_router(xai.router)
app.include_router(reports.router)
app.include_router(demo.router)
app.include_router(multimodal.router)
app.include_router(history.router)
app.include_router(history.activity_router)

@app.get("/")
def root():
    return {
        "project": "Explainable Multimodal Transformer for SCZ-BD Differential Analysis",
        "role": "College Semester 7 Computer Engineering Project",
        "status": "online",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
