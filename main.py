from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
model_path = "C:/FakeNews_detector/Backend/Fakenews/fakenewsmodel.pkl"
model = joblib.load(model_path)

# Load the TfidfVectorizer
vectorizer_path = "C:/FakeNews_detector/Backend/Fakenews/vectorizer.pkl"
vectorizer = joblib.load(vectorizer_path)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input format
class NewsInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Fake News Detector API is running"}

@app.post("/predict")
def predict_fake_news(data: NewsInput):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    logging.info(f"Received input: {data.text}")
    try:
        # Transform the input text
        transformed_text = vectorizer.transform([data.text])
        # Make a prediction
        prediction = model.predict(transformed_text)
        logging.info(f"Prediction: {prediction[0]}")
        return {"prediction": prediction[0]}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": str(e)}
