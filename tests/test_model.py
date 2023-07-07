import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from src.training.data_preprocessing import drop_feat, get_data
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath

@pytest.fixture
def data():
    return pd.DataFrame({"X1": [1, 2], "X2": [3, 4],"X3": [3, 4], "Y": [0, 1]})

def test_drop_feat(data):
    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="main")
    res = drop_feat(data, config.feature.test)
    expected = pd.DataFrame({"X3": [3, 4], "Y": [0, 1]})
    assert_frame_equal(res, expected)

def test_get_data():
    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="main")
    data = get_data(abspath(config.unit_testing_data.path))
    X = data.drop('Y', axis=1)
    Y = data['Y']
    X_expected = pd.DataFrame({"X1": [1, 2], "X2": [3, 4], "X3": [3, 4]})
    Y_expected = pd.Series([0, 1], name="Y")
    assert_frame_equal(X, X_expected)
    assert_series_equal(Y, Y_expected)

