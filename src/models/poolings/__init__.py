from copy import deepcopy

from utils.factory import instantiate


def pooling_from_kwargs(kind, static_ctx, **kwargs):
    # static_ctx is a seperate variable because it shouldn't be deepcopied
    kwargs = deepcopy(kwargs)
    return instantiate(module_names=[f"models.poolings.{kind}"], type_names=[kind], static_ctx=static_ctx, **kwargs)
