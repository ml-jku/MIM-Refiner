from utils.factory import instantiate


def mask_generator_from_kwargs(kind, **kwargs):
    return instantiate(module_names=[f"models.vit.mask_generators.{kind}"], type_names=[kind], **kwargs)
