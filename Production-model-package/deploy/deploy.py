import uvicorn
from joblib import load
from fastapi import FastAPI

app = FastAPI(title="Remote Community Power Usage")

model_unvailable_msg="Model NOT Available"

@app.get("/")
async def root():
    return {"message": "Remote Community Power Usage"}



@app.post("/predict", status_code=200)
async def predict(fh: int):
    """
    Make predictions with the model
    """
    try:
        model = load("../train/lgbm_forecaster.pickle")
    except:
        print("Model not available")
        return model_unvailable_msg
    results = model.predict(fh=fh)

    return results



if __name__ == "__main__":
    # Use this for debugging purposes only
    # reload=True appears to be broken in uvicorn.run() 
    uvicorn.run(app, host="localhost", port=8001, log_level="debug", reload=False)
