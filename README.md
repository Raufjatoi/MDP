# Multiple Disease Predictor (MDP)

A FastAPI-based web application for predicting multiple diseases.

## File Structure

MDP/    
├── app.py     
├── templates/     
│   ├── index.html     
│   ├── heart.html     
│   ├── diabetes.html     
│   ├── flu.html      
│   ├── cold.html     
│   ├── allergy.html      
├── requirements.txt      
├── static/       
├── README.md       
├── LICENSE       
└── models/         
     ├── heart_model.pkl       
     ├── diabetes_model.pkl     
     ├── flu_model.pkl      
     ├── cold_model.pkl      
     └── allergy_model.pkl      
     ├── heart_model.pkl     
     ├── diabetes.py    
     ├── flu.py    
     ├── cold.py   
     └── allergy.py    
     ├── flu.py     
   

## Setup and Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py 
   or 
   uvicorn app:app --reload
   ```

3. Access the app at `http://localhost:8000`

## Contributors

- [Rauf](https://rauf-psi.vercel.app/)    
- [Mudassir](#)      
- [peter](#)   
- [Abdullah](#)    

