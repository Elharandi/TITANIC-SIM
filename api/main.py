from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import joblib
import pandas as pd

# Load pipeline once at startup
model = joblib.load("titanic_pipeline.joblib")

app = FastAPI(title="Titanic Survival API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Features(BaseModel):
    Pclass:     int   = Field(..., ge=1, le=3)
    Age:        float = Field(..., ge=0, le=120)
    Fare:       float = Field(..., ge=0)
    FamilySize: int   = Field(..., ge=1, le=11)
    Has_Cabin:  int   = Field(..., ge=0, le=1)
    Sex_male:   int   = Field(..., ge=0, le=1)
    Title_Miss: int   = Field(..., ge=0, le=1)
    Title_Mr:   int   = Field(..., ge=0, le=1)
    Title_Mrs:  int   = Field(..., ge=0, le=1)
    Title_Rare: int   = Field(..., ge=0, le=1)

    @model_validator(mode='after')
    def check_title_exclusivity(self):
        total = (self.Title_Miss + self.Title_Mr +
                 self.Title_Mrs + self.Title_Rare)
        if total > 1:
            raise ValueError("Only one title flag can be active")
        return self

@app.get("/")
def home():
    return {"message": "Titanic Survival API is running. Visit /docs"}

@app.post("/predict")
def predict(data: Features):
    try:
        input_data = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_data).tolist()[0]
        survival_proba = float(model.predict_proba(input_data)[0, 1])
        return {
            "survival_probability": survival_proba,
            "prediction": prediction,
            "verdict": "survived" if prediction == 1 else "did not survive",
            "confidence": "high" if abs(survival_proba - 0.5) > 0.3 else "uncertain"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
