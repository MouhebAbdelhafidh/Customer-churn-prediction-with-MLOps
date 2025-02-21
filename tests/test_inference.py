import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import load_model

import numpy as np


def test_model_inference():
    # Load the trained model
    model = load_model()

    # Create a new data point for prediction
    sample = np.array(
        [
            [
                18,
                117.0,
                408,
                0,
                0,
                184.500000,
                97.0,
                203.355322,
                80.0,
                215.8,
                90.0,
                8.7,
                4.0,
                1.0,
            ]
        ]
    )

    # Predict with the model
    prediction = model.predict(sample)

    # Assert that the prediction is a single value (since we're predicting one sample)
    assert len(prediction) == 1
