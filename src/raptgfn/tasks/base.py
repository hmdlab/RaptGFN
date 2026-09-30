"""Base class for sequence-generation tasks."""


class BaseTask:
    """Base interface for a sequence-generation task."""

    def __init__(self, tokenizer, obj_dim, max_len, transform=lambda x: x, batch_size=1, **kwargs):
        self.tokenizer = tokenizer
        self.obj_dim = obj_dim
        self.transform = transform
        self.batch_size = batch_size
        self.max_len = max_len

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate candidates; implemented by subclasses."""
        raise NotImplementedError

    def score(self, str_array):
        """Compute objective scores; implemented by subclasses."""
        raise NotImplementedError

    def task_setup(self, *args, **kwargs):
        """Initialize the task; implemented by subclasses."""
        raise NotImplementedError
