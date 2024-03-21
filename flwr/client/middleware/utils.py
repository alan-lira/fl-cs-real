# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
# ==============================================================================
"""Utility functions for middleware layers."""


from typing import List

from flwr.client.typing import FlowerCallable, Layer
from flwr.common.context import Context
from flwr.common.message import Message


def make_ffn(ffn: FlowerCallable, layers: List[Layer]) -> FlowerCallable:
    """."""

    def wrap_ffn(_ffn: FlowerCallable, _layer: Layer) -> FlowerCallable:
        def new_ffn(message: Message, context: Context) -> Message:
            return _layer(message, context, _ffn)

        return new_ffn

    for layer in reversed(layers):
        ffn = wrap_ffn(ffn, layer)

    return ffn
