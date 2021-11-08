# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Optional, Type, Union
from unittest.mock import Mock

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_deprecation


def is_overridden(
    method_name: str,
    instance: Optional[object] = None,
    parent: Optional[Type[object]] = None,
    model: Optional[Union["pl.LightningModule", "pl.LightningDataModule"]] = None,
) -> bool:
    if model is not None and instance is None:
        rank_zero_deprecation(
            "`is_overriden(model=...)` has been deprecated and will be removed in v1.6."
            "Please use `is_overriden(instance=...)`"
        )
        instance = model

    if instance is None:
        # if `self.lightning_module` was passed as instance, it can be `None`
        return False

    if parent is None:
        if isinstance(instance, (pl.LightningModule, pl.LightningDataModule, pl.Callback)):
            parent = instance.__class__.__bases__
        if parent is None:
            raise ValueError("Expected a parent")

    parent = (parent,) if not isinstance(parent, tuple) else parent

    instance_attr = getattr(instance, method_name, None)
    # `functools.wraps()` support
    if hasattr(instance_attr, "__wrapped__"):
        instance_attr = instance_attr.__wrapped__
    # `Mock(wraps=...)` support
    if isinstance(instance_attr, Mock):
        # access the wrapped function
        instance_attr = instance_attr._mock_wraps
    # `partial` support
    elif isinstance(instance_attr, partial):
        instance_attr = instance_attr.func
    if instance_attr is None:
        return False

    parent_attrs = [getattr(p, method_name, None) for p in parent]
    if all(parent_attr is None for parent_attr in parent_attrs):
        raise ValueError("The parent should define the method")

    return any(
        instance_attr.__code__ != parent_attr.__code__ for parent_attr in parent_attrs if parent_attr is not None
    )
