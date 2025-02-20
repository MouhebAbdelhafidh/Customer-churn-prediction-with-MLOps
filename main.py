import argparse
import joblib
from pipeline import (
    prepare_data,
    train_model,
    save_model,
    evaluate_model,
    load_model,
)

def main():
    parser = argparse.ArgumentParser(description="Model training pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")

    args = parser.parse_args()

    if args.prepare:
        X_train, X_test, y_train, y_test = prepare_data()
        print("✅ Data prepared.")

    if args.train:
        X_train, X_test, y_train, y_test = prepare_data()
        model = train_model(X_train, y_train)
        save_model(model)
        print("✅ Model trained and saved.")

    if args.evaluate:
        X_train, X_test, y_train, y_test = prepare_data()
        model = load_model()
        if model is None:
            print("⚠️ Erreur : No model was found. First train the model using --train.")
            return
        evaluate_model(X_test, y_test)
        print("✅ Model evaluation done.")

if __name__ == "__main__":
    main()
