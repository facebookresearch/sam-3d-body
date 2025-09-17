import torch
from typing import Optional, Union


class Base(torch.nn.Module):
    def __init__(self, seed_or_generator: Optional[Union[int, torch.Generator]] = None):
        super().__init__()

        if isinstance(seed_or_generator, torch.Generator):
            self.random_generator = seed_or_generator
        elif isinstance(seed_or_generator, int):
            self.seed = seed_or_generator
        elif seed_or_generator is None:
            self.random_generator = torch.default_generator
        else:
            raise RuntimeError(
                f"cannot use argument of type {type(seed_or_generator)} to set random generator"
            )

    @property
    def seed(self):
        raise AttributeError(f"Cannot read attribute 'seed'.")

    @seed.setter
    def seed(self, value: int):
        self._random_generator = torch.Generator().manual_seed(value)

    @property
    def random_generator(self):
        return self._random_generator

    @random_generator.setter
    def random_generator(self, generator: torch.Generator):
        self._random_generator = generator

    def forward(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        return self.generate(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        )

    def generate(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        for _, xt, _ in self.generate_iter(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            pass
        return xt

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        raise NotImplementedError

    def loss(self, x, *args_conditionals, **kwargs_conditionals):
        raise NotImplementedError
