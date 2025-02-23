from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List  # ✅ Import de List pour la compatibilité avec Python <3.9
from pipeline import prepare_data, train_model, save_model 

# Charger le modèle entraîné
model = joblib.load("model.joblib")

# Initialiser l'application FastAPI
app = FastAPI()

# Définir le format des données d'entrée
class InputData(BaseModel):
    features: List[float]  # ✅ Utilisation de List[float] au lieu de list[float]

# Définir la route de prédiction
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convertir les données d'entrée en tableau NumPy
        input_array = np.array(data.features).reshape(1, -1)
        
        # Faire la prédiction
        prediction = model.predict(input_array)
        
        # Retourner le résultat
        return {"Customer will retain" if prediction.tolist()==0 else "Customer will churn"}
    
    except Exception as e:
        return {"error": str(e)}

# 📌 Endpoint pour le réentraînement du modèle
@app.post("/retrain")
def retrain():
    try:
        # 🎯 1. Recharger les données
        x_train, x_test, y_train, y_test = prepare_data()

        # 🎯 2. Début d'une nouvelle expérience MLflow
        #mlflow.set_experiment("Model Retraining")

        #with mlflow.start_run():
            # 🎯 3. Réentraîner le modèle
        new_model = train_model(x_train, y_train)

            # 🎯 4. Sauvegarder le modèle mis à jour
        save_model(new_model)

            # 🎯 5. Enregistrer dans MLflow
            #mlflow.sklearn.log_model(new_model, "updated_model")
            #mlflow.log_param("retrain", "True")

            # 🎯 6. Recharger le modèle en mémoire
        global model
        model = new_model

        return {"message": "Model retrained successfully."}

    except Exception as e:
        return {"error": str(e)}
# Point d'entrée principal pour exécuter l'API avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
