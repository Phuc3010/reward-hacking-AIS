import torch

class GeometricMixtureWrapper:
    r"""
    Geometric Mixture generation wrapper that samples from the logits of two model's geometric mixture.

    Args:
        model (`PreTrainedModel`): The model to be wrapped.
        ref_model (`PreTrainedModel`): The reference model.
        generation_config (`GenerationConfig`): The generation config.
        mixture_coef (`float`, *optional* - default: 0.5): The mixture coefficient.
    """

    main_input_name = "input_ids"
    _supports_cache_class = False
    _supports_static_cache = False

    def __init__(self, model, ref_model, generation_config, mixture_coef=0.5, device=None):
        super().__init__()

        self.model = model.eval()
        self.config = model.config
        self.ref_model = ref_model.eval()
        self.generation_config = generation_config
        self.mixture_coef = mixture_coef
        self.device = device

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        model_outputs = self.model(*args, **kwargs)
        model_logits = model_outputs.logits
        ref_model_logits = self.ref_model(*args, **kwargs).logits

        model_outputs.logits = torch.nn.functional.log_softmax(
            self.mixture_coef * ref_model_logits + (1 - self.mixture_coef) * model_logits, dim=-1
        )

        return model_outputs