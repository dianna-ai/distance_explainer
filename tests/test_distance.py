"""Tests for the distance_explainer."""
import dataclasses
import os
from typing import Callable
import numpy as np
import pytest
from numpy.typing import ArrayLike
from distance_explainer import DistanceExplainer
from tests.config import Config
from tests.config import get_default_config

DUMMY_EMBEDDING_DIMENSIONALITY = 10


@pytest.fixture(autouse=True)
def set_all_the_seeds(seed_value=0):
    """Set all necessary seeds."""
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)


@pytest.fixture()
def dummy_model() -> Callable:
    """Get a dummy model that returns a random embedding for every input in a batch."""
    return lambda x: np.random.randn(x.shape[0], DUMMY_EMBEDDING_DIMENSIONALITY)


@pytest.fixture
def dummy_data() -> tuple[ArrayLike, ArrayLike]:
    """Get random dummy data."""
    embedded_reference = np.random.randn(DUMMY_EMBEDDING_DIMENSIONALITY)
    input_arr = np.random.random((32, 32, 3))
    return embedded_reference, input_arr


def get_explainer(config: Config, axis_labels={2: 'channels'}, preprocess_function=None) -> DistanceExplainer:
    """Get explainer object."""
    explainer = DistanceExplainer(mask_selection_range_max=config.mask_selection_range_max,
                                  mask_selection_range_min=config.mask_selection_range_min,
                                  mask_selection_negative_range_max=config.mask_selection_negative_range_max,
                                  mask_selection_negative_range_min=config.mask_selection_negative_range_min,
                                  n_masks=config.number_of_masks,
                                  axis_labels=axis_labels,
                                  preprocess_function=preprocess_function,
                                  feature_res=config.feature_res,
                                  p_keep=config.p_keep)
    return explainer


def test_distance_explainer_saliency(dummy_data: tuple[ArrayLike, ArrayLike],
                                     dummy_model: Callable):
    """Code output should be identical to recorded saliency."""
    embedded_reference, input_arr = dummy_data
    explainer = get_explainer(get_default_config())
    expected_saliency, expected_value = np.load('./tests/test_data/test_dummy_data_exact_expected_output.npz').values()

    saliency = explainer.explain_image_distance(dummy_model, input_arr, embedded_reference)

    assert saliency.shape == (1,) + input_arr.shape[:2] + (1,)  # Has correct shape
    assert np.allclose(expected_saliency, saliency)  # Has correct saliency


@pytest.mark.parametrize("empty_side,expected_tag",
                         [({"mask_selection_range_max": 0.}, "pos_empty"),
                          ({"mask_selection_negative_range_min": 1.}, "neg_empty")])
def test_distance_explainer_one_sided_saliency(dummy_data: tuple[ArrayLike, ArrayLike],
                                               dummy_model: Callable,
                                               empty_side: dict[str, float],
                                               expected_tag: str):
    """Code output should be identical to recorded saliency."""
    embedded_reference, input_arr = dummy_data
    expected_saliency, expected_value = np.load(
        f'./tests/test_data/test_dummy_data_exact_expected_output_{expected_tag}.npz').values()
    config = dataclasses.replace(get_default_config(), **empty_side)
    explainer = get_explainer(config)

    saliency = explainer.explain_image_distance(dummy_model, input_arr, embedded_reference)
    assert saliency.shape == (1,) + input_arr.shape[:2] + (1,)  # Has correct shape
    assert np.allclose(expected_saliency, saliency)  # Has correct saliency
