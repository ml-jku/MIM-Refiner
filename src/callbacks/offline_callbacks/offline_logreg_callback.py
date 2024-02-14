import os
from functools import partial

import numpy as np
import torch
from kappadata.wrappers import ModeWrapper

from callbacks.base.periodic_callback import PeriodicCallback
from distributed.config import get_world_size
from models.extractors import extractor_from_kwargs
from models.extractors.base.forward_hook import StopForwardException
from utils.factory import create_collection
from utils.object_from_kwargs import objects_from_kwargs

try:
    from cyanure import MultiClassifier, preprocess
except ImportError:
    # cyanure is only available for linux -> mock on windows for development purposes
    assert os.name == "nt"


    class MultiClassifier:
        def __init__(self, *_, **__):
            self.w = None

        def fit(self, X, y, *_, **__):
            assert isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype == np.float32
            assert isinstance(y, np.ndarray) and y.ndim == 1 and y.dtype == np.int64
            self.w = np.ones((X.shape[1], y.max() + 1))

        @staticmethod
        def score(X, y, *_, **__):
            assert isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype == np.float32
            assert isinstance(y, np.ndarray) and y.ndim == 1 and y.dtype == np.int64
            return np.float64(0.1)


    # https://github.com/inria-thoth/cyanure/blob/master/cyanure/data_processing.py#L21
    # noinspection PyUnusedLocal
    def preprocess(X, centering=False, normalize=True, columns=False):
        return X


class OfflineLogregCallback(PeriodicCallback):
    def __init__(
            self,
            test_dataset_key,
            extractors,
            train_dataset_key=None,
            train_dataset_keys=None,
            forward_kwargs=None,
            predict_dataset_key=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_dataset_key = test_dataset_key
        self.predict_dataset_key = predict_dataset_key
        self.train_dataset_keys = []
        if train_dataset_key is not None:
            self.train_dataset_keys.append(train_dataset_key)
        if train_dataset_keys is not None:
            self.train_dataset_keys += train_dataset_keys
        self.extractors = extractors
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.__test_config_id = None
        self.__predict_config_id = None
        self.__train_config_ids = None
        self.out = self.path_provider.stage_output_path / "logits"

    def _register_sampler_configs(self, trainer):
        self.__test_config_id = self._register_sampler_config_from_key(
            key=self.test_dataset_key,
            mode=trainer.dataset_mode,
        )
        if self.predict_dataset_key is not None:
            self.__predict_config_id = self._register_sampler_config_from_key(
                key=self.predict_dataset_key,
                mode=trainer.dataset_mode
            )
        self.__train_config_ids = []
        for train_dataset_key in self.train_dataset_keys:
            config_id = self._register_sampler_config_from_key(key=train_dataset_key, mode=trainer.dataset_mode)
            self.__train_config_ids.append(config_id)

    def _before_training(self, model, **kwargs):
        self.extractors = create_collection(self.extractors, extractor_from_kwargs, static_ctx=model.static_ctx)
        for extractor in self.extractors:
            extractor.register_hooks(model)
            extractor.disable_hooks()
        self.out.mkdir(exist_ok=True)

    def _forward(self, batch, trainer_model, trainer):
        generator = trainer_model(batch=batch, **self.forward_kwargs)
        with trainer.autocast_context:
            try:
                next(generator)
            except (StopForwardException, StopIteration):
                pass
        features = {str(extractor): extractor.extract().cpu() for extractor in self.extractors}
        batch, _ = batch  # remove ctx
        target = ModeWrapper.get_item(mode=trainer.dataset_mode, item="class", batch=batch)
        return features, target.clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer_model, trainer, batch_size, data_iter, **_):
        assert get_world_size() == 1, "LogisticRegression is not implemented for multi-gpu"
        iterate = partial(
            self.iterate_over_dataset,
            forward_fn=partial(self._forward, trainer_model=trainer_model, trainer=trainer),
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # test_dataset forward
        test_features, test_y = iterate(config_id=self.__test_config_id, )
        test_y = test_y.numpy()

        # predict
        if self.predict_dataset_key is not None:
            pred_features, pred_y = iterate(config_id=self.__predict_config_id)
            pred_y = pred_y.to(model.device, non_blocking=True)
        else:
            pred_features = None
            pred_y = None

        # iterate over train datasets
        for train_dataset_key, train_config_id in zip(self.train_dataset_keys, self.__train_config_ids):
            # train_dataset foward
            train_features, train_y = iterate(config_id=train_config_id)
            train_y = train_y.numpy()

            # setup logistic regression
            classifier = MultiClassifier(loss="multiclass-logistic", penalty="l2", fit_intercept=False)

            # calculate/log metrics
            # check that len(train_features) == len(train_y) -> raises error when 2 views are propagated
            assert all(len(v) == len(train_y) for v in train_features.values())

            for feature_key in train_features.keys():
                train_x = train_features[feature_key].numpy()
                test_x = test_features[feature_key].numpy()
                assert len(test_x) == len(test_y)

                # sanity check
                assert len(train_x) == len(train_y)
                assert len(test_x) == len(test_y)
                assert train_x.ndim == 2
                assert test_x.ndim == 2

                # fit
                self.logger.info(f"fit logistic regression of {len(train_x)} samples")
                # https://github.com/facebookresearch/msn/blob/main/logistic_eval.py#L172
                preprocess(train_x, normalize=True, columns=False, centering=True)
                lamb = 0.00025 / len(train_x)
                classifier.fit(
                    train_x,
                    train_y,
                    it0=10,
                    lambd=lamb,
                    lambd2=lamb,
                    nthreads=10,
                    tol=1e-3,
                    solver="auto",
                    seed=0,
                    max_epochs=300,
                )

                # calculate accuracy
                self.logger.info(f"calculate logistic regression accuracy on test set")
                accuracy = float(classifier.score(test_x, test_y))
                key = f"accuracy1/logreg/{train_dataset_key}/{feature_key}"
                self.logger.info(f"{key}: {accuracy:.6f}")
                self.writer.add_scalar(key, accuracy, logger=self.logger, format_str=".6f")

                # save weights
                w_fname = f"logreg.{train_dataset_key}.{feature_key} cp={self.update_counter.cur_checkpoint}.th"
                torch.save(torch.from_numpy(classifier.w).float(), self.path_provider.checkpoint_path / w_fname)

                # predict
                if pred_features is not None:
                    self.logger.info(f"predict logistic regression on prediction set")
                    cur_pred = pred_features[feature_key].to(model.device, non_blocking=True)
                    w = torch.from_numpy(classifier.w).float().to(model.device, non_blocking=True)
                    predictions = cur_pred @ w
                    accuracy = (predictions.argmax(dim=1) == pred_y).sum() / len(predictions)
                    key = f"accuracy1/logreg/{train_dataset_key}/{self.predict_dataset_key}/{feature_key}/full"
                    self.logger.info(f"{key}: {accuracy:.6f}")
                    self.writer.add_scalar(key, accuracy, logger=self.logger, format_str=".6f")

                    # write predictions
                    if len(pred_features) == 1 and self.update_counter.end_checkpoint.update == 0:
                        fname = f"{train_dataset_key}.th"
                    else:
                        fname = f"{train_dataset_key}_{feature_key}_{self.update_counter.cur_checkpoint}.th"
                    cur_out = self.out / fname
                    self.logger.info(f"writing predicted logits to {cur_out}")
                    torch.save(predictions.cpu(), cur_out)

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
