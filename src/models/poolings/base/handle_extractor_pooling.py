from contextlib import contextmanager

from models.extractors.base.forward_hook import StopForwardException


@contextmanager
def handle_extractor_pooling(poolings):
    assert isinstance(poolings, (tuple, list))
    for pooling in poolings:
        pooling.enable_hooks()
    try:
        yield
    except StopForwardException:
        # clear extractors from pooling
        for pooling in poolings:
            pooling.clear_extractor_outputs()
            pooling.disable_hooks()
        raise
    for pooling in poolings:
        pooling.disable_hooks()
