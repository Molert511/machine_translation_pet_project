from typing import Protocol, Any, Iterator

import torch

class ModelProtocol(Protocol):
    vocab_src: dict[str, int]
    vocab_tgt: dict[str, int]

    def __call__(
        self,
        src: torch.Tensor,
        src_lengths: list[int],
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def translate(self, src: torch.Tensor, src_lengths: list[int]) -> list[list[int]]:
        ...

    @property
    def device(self) -> torch.device:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        ...

    def eval(self) -> "ModelProtocol":
        ...

    def train(self) -> "ModelProtocol":
        ...
