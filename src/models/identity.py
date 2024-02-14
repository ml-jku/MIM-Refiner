from models.base.single_model_base import SingleModelBase


class Identity(SingleModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = self.input_shape

    @staticmethod
    def forward(x):
        return x

    def features(self, x):
        return self(x)

    def predict(self, x):
        return dict(main=self(x))
