"""Tests for the distance_explainer
"""
import os
from typing import Callable

import numpy as np
import pytest

from distance_explainer.config import get_default_config
from distance_explainer.distance import DistanceExplainer

DUMMY_EMBEDDING_DIMENSIONALITY = 10


def test_dummy_data_exact_expected_output(set_all_the_seeds: Callable, dummy_model: Callable):
    """Code output should be identical to recorded output."""
    [expected_saliency, expected_value] = np.load('./test_data/test_dummy_data_exact_expected_output.npz').values()
    embedded_reference = np.random.randn(1, DUMMY_EMBEDDING_DIMENSIONALITY)
    input_arr = np.random.random((32, 32, 3))

    config = get_default_config()
    explainer = DistanceExplainer(mask_selection_range_max=config.mask_selection_range_max,
                                  mask_selection_range_min=config.mask_selection_range_min,
                                  mask_selection_negative_range_max=config.mask_selection_negative_range_max,
                                  mask_selection_negative_range_min=config.mask_selection_negative_range_min,
                                  n_masks=config.number_of_masks,
                                  axis_labels={2: 'channels'},
                                  preprocess_function=None,
                                  feature_res=config.feature_res,
                                  p_keep=config.p_keep)
    saliency, value = explainer.explain_image_distance(dummy_model, input_arr, embedded_reference)

    assert saliency.shape == (1,) + input_arr.shape[:2] + (1,)  # Has correct shape
    assert np.allclose(expected_saliency, saliency)  # Has correct value
    assert np.allclose(expected_value, value)  # Has correct value





@pytest.fixture()
def dummy_model() -> Callable:
    """Get a dummy model that returns a random embedding for every input in a batch."""
    return lambda x: np.random.randn(x.shape[0], DUMMY_EMBEDDING_DIMENSIONALITY)


@pytest.fixture()
def set_all_the_seeds(seed_value=0):
    """Set all necessary seeds."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
