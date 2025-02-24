const API_URL = "http://127.0.0.1:8000"; // Assurez-vous que FastAPI tourne à cette adresse

// 📌 Fonction pour envoyer une prédiction
async function predict() {
    const featuresInput = document.getElementById("features").value;
    
    if (!featuresInput) {
        alert("Veuillez entrer des caractéristiques !");
        return;
    }

    const featuresArray = featuresInput.split(",").map(Number);

    const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ features: featuresArray })
    });

    const data = await response.json();
    document.getElementById("result").textContent = `Résultat : ${JSON.stringify(data)}`;
}

// 📌 Fonction pour réentraîner le modèle
async function retrain() {
    const response = await fetch(`${API_URL}/retrain`, { method: "POST" });

    const data = await response.json();
    document.getElementById("retrain-status").textContent = data.message || "Réentraînement terminé.";
}
