# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Context."""


from dataclasses import dataclass

from .recordset import RecordSet


@dataclass
class Context:
    """State of your run.

    Parameters
    ----------
    state : RecordSet
        Holds records added by the entity in a given run and that will stay local.
        This means that the data it holds will never leave the system it's running from.
        This can be used as an intermediate storage or scratchpad when
        executing middleware layers. It can also be used as a memory to access
        at different points during the lifecycle of this entity (e.g. across
        multiple rounds)
    """

    state: RecordSet
