from abc import ABC, abstractmethod
import torch


class BasePipeline(ABC):
    """Abstract base class for inference pipelines.

    Every subclass receives an image and its ground-truth label at the
    original resolution, preprocesses both, runs the model, post-processes
    the logits, and returns a segmentation map at the original resolution.

    The ``gt_label`` argument is needed for oracle-mode models; pipelines
    that use a real trained model can ignore it.
    """

    @abstractmethod
    def run(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        model: torch.nn.Module,
        target_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError
