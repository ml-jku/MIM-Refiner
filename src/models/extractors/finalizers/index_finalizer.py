class IndexFinalizer:
    def __init__(self, index):
        self.index = index

    def __call__(self, features):
        return features[self.index]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{type(self).__name__}({self.index})"
