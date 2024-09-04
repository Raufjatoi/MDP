from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import joblib
import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

## Load models 
#heart_model = joblib.load("models/heart_model.pkl")
#diabetes_model = joblib.load("models/diabetes_model.pkl")
#flu_model = joblib.load("models/flu_model.pkl")
#cold_model = joblib.load("models/cold_model.pkl")
#allergy_model = joblib.load("models/allergy_model.pkl")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/heart")
async def heart(request: Request):
    return templates.TemplateResponse("heart.html", {"request": request})

@app.get("/diabetes")
async def diabetes(request: Request):
    return templates.TemplateResponse("diabetes.html", {"request": request})

@app.get("/flu")
async def flu(request: Request):
    return templates.TemplateResponse("flu.html", {"request": request})

@app.get("/cold")
async def cold(request: Request):
    return templates.TemplateResponse("cold.html", {"request": request})

@app.get("/allergy")
async def allergy(request: Request):
    return templates.TemplateResponse("allergy.html", {"request": request})

@app.get("/back")
async def back():
    return RedirectResponse(url="/")

## Add prediction endpoints for each disease
#@app.post("/predict/heart")
#async def predict_heart(data: dict):
    # Process input data and make prediction
    # Return prediction result

#@app.post("/predict/diabetes")
#async def predict_diabetes(data: dict):
    # Process input data and make prediction
    # Return prediction result

#@app.post("/predict/flu")
#async def predict_flu(data: dict):
    # Process input data and make prediction
    # Return prediction result

#@app.post("/predict/cold")
#async def predict_cold(data: dict):
    # Process input data and make prediction
    # Return prediction result

#@app.post("/predict/allergy")
#async def predict_allergy(data: dict):
    # Process input data and make prediction
    # Return prediction result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)