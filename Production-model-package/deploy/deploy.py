import uvicorn
from joblib import load
import pandas as pd
from fastapi import APIRouter, FastAPI, Request

app = FastAPI(title="Remote Community Power Usage")


@app.get("/")
async def root():
    return {"message": "Remote Community Power Usage"}

@app.post("/predict", status_code=200)
async def predict(fh: int):
    """
    Make predictions with the model
    """
    model = load("train/lgbm_forecaster.pickle")
    results = model.predict(fh=fh)

    return results



if __name__ == "__main__":
    # Use this for debugging purposes only
    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
