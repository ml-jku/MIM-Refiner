from kappadata import get_norm_transform
from kappadata.wrappers import XTransformWrapper
from kappadata.transforms.norm import KDImageRangeNorm


def get_norm_parameters_from_datacontainer(data_container, dataset_key=None):
    transform = get_norm_transform_from_datacontainer(data_container=data_container, dataset_key=dataset_key)
    if transform is None:
        return None, None
    if isinstance(transform, KDImageRangeNorm):
        return (0.5,), (0.5,)
    return transform.mean, transform.std


def get_norm_transform_from_datacontainer(data_container, dataset_key=None):
    ds, collator = data_container.get_dataset(key=dataset_key, mode="x")
    if collator is not None:
        raise NotImplementedError
    return get_norm_transform_from_dataset(ds)


def get_norm_transform_from_dataset(dataset):
    xtransform_wrapper = dataset.get_wrapper_of_type(XTransformWrapper)
    if xtransform_wrapper is None:
        return None
    return get_norm_transform(xtransform_wrapper.transform)
