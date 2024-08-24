
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from house_prices_model.config.core import config
from house_prices_model.processing.features import OutlierHandler

def test_distance_variable_outlierhandler(sample_input_data):
    # Given
    encoder = OutlierHandler(variable = config.model_config.distance_var)
    q1, q3 = np.percentile(sample_input_data[0]['distance'], q=[25, 75])
    iqr = q3 - q1
    assert sample_input_data[0].loc[100, 'distance'] > q3 + (1.5 * iqr)

    # When
    subject = encoder.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[100, 'distance'] <= q3 + (1.5 * iqr)
