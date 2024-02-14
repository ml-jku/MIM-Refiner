from torch import nn

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from models.poolings.base.handle_extractor_pooling import handle_extractor_pooling
from utils.factory import create, create_collection
from models.probe.batchnorm1d import BatchNorm1d
from models.probe.layernorm import LayerNorm
from models.identity import Identity

class ProbeModel(CompositeModelBase):
    def __init__(self, encoder, heads, norm_mode="batchnorm", **kwargs):
        super().__init__(**kwargs)
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            is_frozen=True,
        )
        heads = create_collection(
            heads,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            output_shape=self.output_shape,
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        if norm_mode == "batchnorm":
            norm_ctor = BatchNorm1d
        elif norm_mode == "layernorm":
            norm_ctor = LayerNorm
        elif norm_mode == "identity":
            norm_ctor = Identity
        else:
            raise NotImplementedError
        self.norms = nn.ModuleDict({
            str(pooling): norm_ctor(input_shape=pooling.get_output_shape(self.encoder.output_shape))
            for pooling in list(set([head.pooling for head in heads.values()]))
        })
        self.heads = nn.ModuleDict(heads)
        # register pooling hooks (required for ExtractorPooling)
        for head in self.heads.values():
            head.pooling.register_hooks(self.encoder)

    @property
    def submodels(self):
        return dict(
            encoder=self.encoder,
            **{f"norms.{key}": value for key, value in self.norms.items()},
            **{f"heads.{key}": value for key, value in self.heads.items()},
        )

    def forward(self, x):
        # forward student encoder
        poolings = [head.pooling for head in self.heads.values()]
        with handle_extractor_pooling(poolings):
            encoder_outputs = {}
            # encoder forward
            encoder_output = self.encoder(x)["main"]
            # pool + norm
            for head in self.heads.values():
                # only add if it wasn't already added (multiple heads can have the same pooling)
                if head.pooling not in encoder_outputs:
                    pooled = head.pooling(encoder_output)
                    encoder_outputs[head.pooling] = self.norms[head.pooling](pooled)
        # predict
        outputs = {
            head_name: head(encoder_outputs[head.pooling], apply_pooling=False)
            for head_name, head in self.heads.items()
        }
        return outputs

    def classify(self, x):
        return self(x)