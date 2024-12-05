from torch.nn import Identity

class Submod:
    """
    A Submod is a helper class which contains a submodule of a model, along with an optional LayerNorm module
    that is applied before the submodule.
    """
    def __init__(self, name, module, LN_pre=None, LN_post=None):
        self.name = name
        self.module = module
        self.LN_pre = LN_pre if LN_pre is not None else Identity()
        self.LN_post = LN_post if LN_post is not None else Identity()