from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List , Dict

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

## Load models 
heart_model = joblib.load("models/random_forest_model.pkl")
diabetes_model = joblib.load("models/RF_DIABETES.pkl")
flu_model = joblib.load("models/LR_FLU_ALLERGY.pkl")
#cold_model = joblib.load("models/cold_model.pkl")
allergy_model = joblib.load("models/LR_FLU_ALLERGY.pkl")

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
@app.get("/lung")
async def lung(request: Request):
    return templates.TemplateResponse("lung.html", {"request": request})
@app.get("/allergy")
async def allergy(request: Request):
    return templates.TemplateResponse("allergy.html", {"request": request})
@app.get("/allergy")
async def allergy(request: Request):
    return templates.TemplateResponse("allergy.html", {"request": request})
@app.get("/brain")
async def brain(request: Request):
    return templates.TemplateResponse("brain.html", {"request": request})
@app.get("/pneumonia")
async def pneumonia(request: Request):
    return templates.TemplateResponse("pneumonia.html", {"request": request})
@app.get("/back")
async def back():
    return RedirectResponse(url="/")


class Response(BaseModel):
    answers: dict 

@app.post("/heart_submit_responses")
async def heart_submit_responses(response: Response):
    feature_vector = [
        response.answers.get("Do you have a cough?", "no") == "yes",
        response.answers.get("Do you have muscle aches?", "no") == "yes",
        response.answers.get("Do you feel tired?", "no") == "yes",
        response.answers.get("Do you have a sore throat?", "no") == "yes",
        response.answers.get("Do you have a runny nose?", "no") == "yes",
        response.answers.get("Do you have a stuffy nose?", "no") == "yes",
        response.answers.get("Do you have a fever?", "no") == "yes",
        response.answers.get("Do you experience nausea?", "no") == "yes",
        response.answers.get("Do you experience vomiting?", "no") == "yes",
        response.answers.get("Do you have diarrhea?", "no") == "yes",
        response.answers.get("Do you experience shortness of breath?", "no") == "yes",
        response.answers.get("Do you have difficulty breathing?", "no") == "yes",
        response.answers.get("Have you lost your sense of taste?", "no") == "yes",
        response.answers.get("Have you lost your sense of smell?", "no") == "yes",
        response.answers.get("Do you have an itchy nose?", "no") == "yes",
        response.answers.get("Do you have itchy eyes?", "no") == "yes",
        response.answers.get("Do you have an itchy mouth?", "no") == "yes",
        response.answers.get("Do you have an itchy inner ear?", "no") == "yes",
        response.answers.get("Do you experience sneezing?", "no") == "yes",
        response.answers.get("Do you have pink eye?", "no") == "yes"
    ]
    prediction = heart_model.predict([feature_vector])

    if prediction[0] == 1: 
        analysis_result = "Possible heart symptoms detected."
    else:
        analysis_result = "No heart symptoms detected."

    return {"analysis": analysis_result}

class FluResponse(BaseModel):
    answers: dict  

@app.post("/flu_submit_responses")
async def flu_submit_responses(response: FluResponse):
    feature_vector = [
        response.answers.get("Do you have a cough?", "no") == "yes",
        response.answers.get("Do you have muscle aches?", "no") == "yes",
        response.answers.get("Do you feel tired?", "no") == "yes",
        response.answers.get("Do you have a sore throat?", "no") == "yes",
        response.answers.get("Do you have a runny nose?", "no") == "yes",
        response.answers.get("Do you have a stuffy nose?", "no") == "yes",
        response.answers.get("Do you have a fever?", "no") == "yes",
        response.answers.get("Do you experience nausea?", "no") == "yes",
        response.answers.get("Do you experience vomiting?", "no") == "yes",
        response.answers.get("Do you have diarrhea?", "no") == "yes",
        response.answers.get("Do you experience shortness of breath?", "no") == "yes",
        response.answers.get("Do you have difficulty breathing?", "no") == "yes",
        response.answers.get("Have you lost your sense of taste?", "no") == "yes",
        response.answers.get("Have you lost your sense of smell?", "no") == "yes",
        response.answers.get("Do you have an itchy nose?", "no") == "yes",
        response.answers.get("Do you have itchy eyes?", "no") == "yes",
        response.answers.get("Do you have an itchy mouth?", "no") == "yes",
        response.answers.get("Do you have an itchy inner ear?", "no") == "yes",
        response.answers.get("Do you experience sneezing?", "no") == "yes",
        response.answers.get("Do you have pink eye?", "no") == "yes"
    ]
    with open("models/LR_FLU_ALLERGY.pkl") as m:
        model1 = pickle.load(m)
    def f_predictions(cough,muscle_aches,tiredness,sore_throat,runny_nose,stuffy_nose,fever,nausea,vomiting,diarrhea,shortness_of_breath,difficulty_breathing,loss_of_taste,loss_of_smell,itchy_nose,itchy_eyes,itchy_mouth,itchy_inner_ear,sneezing,pink_eye):
        loss_of_senses = (loss_of_taste + loss_of_smell)/2
        itchiness = (itchy_nose + itchy_eyes + itchy_mouth + itchy_inner_ear)/4
        common_symptoms = (fever + nausea + vomiting + diarrhea)/4
        breathing = (shortness_of_breath + difficulty_breathing)/2
        x = np.array([[cough,muscle_aches,tiredness,sore_throat,runny_nose,stuffy_nose,fever,nausea,vomiting,diarrhea,shortness_of_breath,difficulty_breathing,loss_of_taste,loss_of_smell,itchy_nose,itchy_eyes,itchy_mouth,itchy_inner_ear,sneezing,pink_eye,loss_of_senses,itchiness,common_symptoms,breathing]])
        prediction = model1.predict(x)
        if prediction[0] == 0:
            prediction = "Allergy"
        else:
            prediction = "Flu"
        return prediction
    prediction = f_predictions(feature_values[0],feature_values[1],feature_values[2],feature_values[3],feature_values[4],feature_values[5],feature_values[6],feature_values[7],feature_values[8],feature_values[9],feature_values[10],feature_values[11],feature_values[12],feature_values[13],feature_values[14],feature_values[15],feature_values[16],feature_values[17],feature_values[18],feature_values[19])

    return {"analysis": prediction}



class allergyResponse(BaseModel):
    answers: dict  


@app.post("/allergy_submit_responses")
async def allergy_submit_responses(response: allergyResponse):
    feature_vector = [
        response.answers.get("Do you have a cough?", "no") == "yes",
        response.answers.get("Do you have muscle aches?", "no") == "yes",
        response.answers.get("Do you feel tired?", "no") == "yes",
        response.answers.get("Do you have a sore throat?", "no") == "yes",
        response.answers.get("Do you have a runny nose?", "no") == "yes",
        response.answers.get("Do you have a stuffy nose?", "no") == "yes",
        response.answers.get("Do you have a fever?", "no") == "yes",
        response.answers.get("Do you experience nausea?", "no") == "yes",
        response.answers.get("Do you experience vomiting?", "no") == "yes",
        response.answers.get("Do you have diarrhea?", "no") == "yes",
        response.answers.get("Do you experience shortness of breath?", "no") == "yes",
        response.answers.get("Do you have difficulty breathing?", "no") == "yes",
        response.answers.get("Have you lost your sense of taste?", "no") == "yes",
        response.answers.get("Have you lost your sense of smell?", "no") == "yes",
        response.answers.get("Do you have an itchy nose?", "no") == "yes",
        response.answers.get("Do you have itchy eyes?", "no") == "yes",
        response.answers.get("Do you have an itchy mouth?", "no") == "yes",
        response.answers.get("Do you have an itchy inner ear?", "no") == "yes",
        response.answers.get("Do you experience sneezing?", "no") == "yes",
        response.answers.get("Do you have pink eye?", "no") == "yes"
    ]
    prediction = allergy_model.predict([feature_vector])

    if prediction[0] == 1: 
        analysis_result = "Possible flu symptoms detected."
    else:
        analysis_result = "No flu symptoms detected."

    return {"analysis": analysis_result}



# Define the request model to accept answers as a dict
class DiabetesResponse(BaseModel):
    answers: Dict[str, float]  # Assuming all values are numeric, adjust as needed

# Assuming you have a preloaded diabetes prediction model
# For example, a scikit-learn model loaded elsewhere
# diabetes_model = some_pretrained_model

@app.post("/diabetes_submit_responses")
async def diabetes_submit_responses(response: DiabetesResponse):
    with open("models/RF_DIABETES.pkl",'rb') as file:
        model = pickle.load(file)
    def d_prediction(gender,age,hypertension,heart_disease,smoking_history,bmi,hba1c_level,blood_glucose_level):
        f1 = (hba1c_level + blood_glucose_level)/age
        f2 = (hypertension)/bmi
        f3 = hba1c_level ** 2
        f4 = blood_glucose_level ** 2
        f5 = age ** 0.5
        x = np.array([gender,age,hypertension,heart_disease,smoking_history,bmi,hba1c_level,blood_glucose_level,f1,f2,f3,f4,f5])
        prediction = model.predict([x])
        if prediction[0] == 0:
            prediction = "Not Diabetic"
        else:
            prediction = "Diabetic"
        return prediction
    answers_dict = response.answers

    # Convert the answers dict into a list of values for prediction
    # Assuming the model requires the values in a specific order
    feature_values = list(answers_dict.values())

    # Convert to the correct format for your model, e.g., 2D array for sklearn
    prediction = d_prediction(feature_values[0],feature_values[1],feature_values[2],feature_values[3],feature_values[4],feature_values[5],feature_values[6],feature_values[7])

    return {"prediction": prediction}  # Assuming single prediction

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
    uvicorn.run(app, host="127.0.0.1", port=8000)


#import cv2
#def load_img(path):
#   img = cv2.imread(path)
#   img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#   img = cv2.resize(img,(size,size)) ## SIZE IS THE SIZE OF THE TENSOR WHICH THE MODEL EXPECTS
#   img = np.expand_dims(img,axis=0)
#   img = img / 255.0
#   return img