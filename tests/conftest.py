import pytest

from textforge_ai import dataset


@pytest.fixture(scope="session")
def small_dataset():
    return dataset.generate(150, seed=123)
