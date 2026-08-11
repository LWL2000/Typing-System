import logging

import pytest


@pytest.fixture(autouse=True)
def reset_package_logger():
    logger = logging.getLogger("pure_gaze_typing")
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    yield
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
