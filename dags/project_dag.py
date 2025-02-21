from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import subprocess
import sys
import os

# Ajouter le répertoire parent au chemin Python pour pouvoir importer 'pipeline.py'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline import load_model, prepare_data

# Définir le DAG
dag = DAG(
    'ml_project_dag',  # Nom du DAG
    description='DAG pour orchestrer les étapes du projet ML',
    schedule_interval=None,  # Aucun horaire (peut être modifié selon vos besoins)
    start_date=datetime(2025, 2, 20),  # Date de démarrage
    catchup=False,  # Ne pas exécuter les DAGs passés
)

# Tâches du DAG

# 1. Vérifier le formatage du code avec Black
def check_formatting():
    subprocess.run(['black', '--check', '.'])

format_check_task = PythonOperator(
    task_id='check_formatting',
    python_callable=check_formatting,
    dag=dag,
)

# 2. Exécuter les tests avec pytest
def run_tests():
    subprocess.run(['pytest', 'tests/'])

test_task = PythonOperator(
    task_id='run_tests',
    python_callable=run_tests,
    dag=dag,
)

# 3. Préparer les données (préparation des données)
def prepare_data_task():
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Training data: {X_train.shape[0]} samples, Test data: {X_test.shape[0]} samples")

data_preparation_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data_task,
    dag=dag,
)

# 4. Entraîner le modèle
def train_model_task():
    X_train, _, y_train, _ = prepare_data()  # Utiliser les données préparées
    model = train_model(X_train, y_train)  # Assurez-vous que 'train_model' est défini dans 'pipeline.py'
    print(f"Modèle entraîné : {model}")

model_training_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model_task,
    dag=dag,
)

# 5. Sauvegarder le modèle
def save_model_task():
    model = load_model()  # Charger le modèle déjà formé
    save_model(model)  # Sauvegarder le modèle formé

model_saving_task = PythonOperator(
    task_id='save_model',
    python_callable=save_model_task,
    dag=dag,
)

# 6. Évaluer le modèle
def evaluate_model_task():
    _, X_test, _, y_test = prepare_data()  # Utiliser les données de test
    evaluate_model(X_test, y_test)

model_evaluation_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model_task,
    dag=dag,
)

# Définir les dépendances entre les tâches
format_check_task >> test_task >> data_preparation_task >> model_training_task >> model_saving_task >> model_evaluation_task
