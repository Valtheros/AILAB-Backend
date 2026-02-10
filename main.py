from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.training_service import TrainingService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

training_service = TrainingService()

class TrainRequest(BaseModel):
    model_size: str # n, s, m, l, x
    epochs: int
    batch_size: int = 16
    project_name: str = "train_run"

@app.post("/api/train")
def start_train(request: TrainRequest):
    model_name = f"yolo11{request.model_size}"
    try:
        container_id = training_service.start_training_container(
            model_name=model_name,
            epochs=request.epochs,
            batch_size=request.batch_size,
            project_name=request.project_name
        )
        return {"status": "success", "container_id": container_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{container_id}")
def get_status(container_id: str):
    status = training_service.get_container_status(container_id)
    return {"container_id": container_id, "status": status}

@app.get("/api/logs/{container_id}")
def get_logs(container_id: str):
    logs = training_service.get_container_logs(container_id)
    return {"container_id": container_id, "logs": logs}

@app.get("/api/metrics/{project_name}")
def get_metrics(project_name: str):
    metrics = training_service.get_training_metrics(project_name)
    # If metrics is empty list, it means no data yet
    if not metrics:
        return {"status": "no_data"}
    return {"status": "success", "metrics": metrics}

@app.post("/api/stop/{container_id}")
def stop_train(container_id: str):
    try:
        training_service.stop_training_container(container_id)
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "YOLO Training Backend is running"}
