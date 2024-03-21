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
"""Message handler for the SecAgg+ protocol."""


import os
from dataclasses import dataclass, field
from logging import INFO, WARNING
from typing import Any, Callable, Dict, List, Tuple, cast

from flwr.client.typing import FlowerCallable
from flwr.common import ndarray_to_bytes, parameters_to_ndarrays
from flwr.common import recordset_compat as compat
from flwr.common.configsrecord import ConfigsRecord
from flwr.common.constant import TASK_TYPE_FIT
from flwr.common.context import Context
from flwr.common.logger import log
from flwr.common.message import Message, Metadata
from flwr.common.recordset import RecordSet
from flwr.common.secure_aggregation.crypto.shamir import create_shares
from flwr.common.secure_aggregation.crypto.symmetric_encryption import (
    bytes_to_private_key,
    bytes_to_public_key,
    decrypt,
    encrypt,
    generate_key_pairs,
    generate_shared_key,
    private_key_to_bytes,
    public_key_to_bytes,
)
from flwr.common.secure_aggregation.ndarrays_arithmetic import (
    factor_combine,
    parameters_addition,
    parameters_mod,
    parameters_multiply,
    parameters_subtraction,
)
from flwr.common.secure_aggregation.quantization import quantize
from flwr.common.secure_aggregation.secaggplus_constants import (
    KEY_ACTIVE_SECURE_ID_LIST,
    KEY_CIPHERTEXT_LIST,
    KEY_CLIPPING_RANGE,
    KEY_DEAD_SECURE_ID_LIST,
    KEY_DESTINATION_LIST,
    KEY_MASKED_PARAMETERS,
    KEY_MOD_RANGE,
    KEY_PUBLIC_KEY_1,
    KEY_PUBLIC_KEY_2,
    KEY_SAMPLE_NUMBER,
    KEY_SECURE_ID,
    KEY_SECURE_ID_LIST,
    KEY_SHARE_LIST,
    KEY_SHARE_NUMBER,
    KEY_SOURCE_LIST,
    KEY_STAGE,
    KEY_TARGET_RANGE,
    KEY_THRESHOLD,
    RECORD_KEY_CONFIGS,
    RECORD_KEY_STATE,
    STAGE_COLLECT_MASKED_INPUT,
    STAGE_SETUP,
    STAGE_SHARE_KEYS,
    STAGE_UNMASK,
    STAGES,
)
from flwr.common.secure_aggregation.secaggplus_utils import (
    pseudo_rand_gen,
    share_keys_plaintext_concat,
    share_keys_plaintext_separate,
)
from flwr.common.typing import ConfigsRecordValues, FitRes


@dataclass
# pylint: disable-next=too-many-instance-attributes
class SecAggPlusState:
    """State of the SecAgg+ protocol."""

    current_stage: str = STAGE_UNMASK

    sid: int = 0
    sample_num: int = 0
    share_num: int = 0
    threshold: int = 0
    clipping_range: float = 0.0
    target_range: int = 0
    mod_range: int = 0

    # Secret key (sk) and public key (pk)
    sk1: bytes = b""
    pk1: bytes = b""
    sk2: bytes = b""
    pk2: bytes = b""

    # Random seed for generating the private mask
    rd_seed: bytes = b""

    rd_seed_share_dict: Dict[int, bytes] = field(default_factory=dict)
    sk1_share_dict: Dict[int, bytes] = field(default_factory=dict)
    # The dict of the shared secrets from sk2
    ss2_dict: Dict[int, bytes] = field(default_factory=dict)
    public_keys_dict: Dict[int, Tuple[bytes, bytes]] = field(default_factory=dict)

    def __init__(self, **kwargs: ConfigsRecordValues) -> None:
        for k, v in kwargs.items():
            if k.endswith(":V"):
                continue
            new_v: Any = v
            if k.endswith(":K"):
                k = k[:-2]
                keys = cast(List[int], v)
                values = cast(List[bytes], kwargs[f"{k}:V"])
                if len(values) > len(keys):
                    updated_values = [
                        tuple(values[i : i + 2]) for i in range(0, len(values), 2)
                    ]
                    new_v = dict(zip(keys, updated_values))
                else:
                    new_v = dict(zip(keys, values))
            self.__setattr__(k, new_v)

    def to_dict(self) -> Dict[str, ConfigsRecordValues]:
        """Convert the state to a dictionary."""
        ret = vars(self)
        for k in list(ret.keys()):
            if isinstance(ret[k], dict):
                # Replace dict with two lists
                v = cast(Dict[str, Any], ret.pop(k))
                ret[f"{k}:K"] = list(v.keys())
                if k == "public_keys_dict":
                    v_list: List[bytes] = []
                    for b1_b2 in cast(List[Tuple[bytes, bytes]], v.values()):
                        v_list.extend(b1_b2)
                    ret[f"{k}:V"] = v_list
                else:
                    ret[f"{k}:V"] = list(v.values())
        return ret


def _get_fit_fn(
    msg: Message, ctxt: Context, call_next: FlowerCallable
) -> Callable[[], FitRes]:
    """Get the fit function."""

    def fit() -> FitRes:
        out_msg = call_next(msg, ctxt)
        return compat.recordset_to_fitres(out_msg.message, keep_input=False)

    return fit


def secaggplus_middleware(
    msg: Message,
    ctxt: Context,
    call_next: FlowerCallable,
) -> Message:
    """Handle incoming message and return results, following the SecAgg+ protocol."""
    # Ignore non-fit messages
    if msg.metadata.task_type != TASK_TYPE_FIT:
        return call_next(msg, ctxt)

    # Retrieve local state
    if RECORD_KEY_STATE not in ctxt.state.configs:
        ctxt.state.set_configs(RECORD_KEY_STATE, ConfigsRecord({}))
    state_dict = ctxt.state.get_configs(RECORD_KEY_STATE).data
    state = SecAggPlusState(**state_dict)

    # Retrieve incoming configs
    configs = msg.message.get_configs(RECORD_KEY_CONFIGS).data

    # Check the validity of the next stage
    check_stage(state.current_stage, configs)

    # Update the current stage
    state.current_stage = cast(str, configs.pop(KEY_STAGE))

    # Check the validity of the configs based on the current stage
    check_configs(state.current_stage, configs)

    # Execute
    if state.current_stage == STAGE_SETUP:
        res = _setup(state, configs)
    elif state.current_stage == STAGE_SHARE_KEYS:
        res = _share_keys(state, configs)
    elif state.current_stage == STAGE_COLLECT_MASKED_INPUT:
        fit = _get_fit_fn(msg, ctxt, call_next)
        res = _collect_masked_input(state, configs, fit)
    elif state.current_stage == STAGE_UNMASK:
        res = _unmask(state, configs)
    else:
        raise ValueError(f"Unknown secagg stage: {state.current_stage}")

    # Save state
    ctxt.state.set_configs(RECORD_KEY_STATE, ConfigsRecord(state.to_dict()))

    # Return message
    return Message(
        metadata=Metadata(0, "", "", "", TASK_TYPE_FIT),
        message=RecordSet(configs={RECORD_KEY_CONFIGS: ConfigsRecord(res, False)}),
    )


def check_stage(current_stage: str, configs: Dict[str, ConfigsRecordValues]) -> None:
    """Check the validity of the next stage."""
    # Check the existence of KEY_STAGE
    if KEY_STAGE not in configs:
        raise KeyError(
            f"The required key '{KEY_STAGE}' is missing from the input `named_values`."
        )

    # Check the value type of the KEY_STAGE
    next_stage = configs[KEY_STAGE]
    if not isinstance(next_stage, str):
        raise TypeError(
            f"The value for the key '{KEY_STAGE}' must be of type {str}, "
            f"but got {type(next_stage)} instead."
        )

    # Check the validity of the next stage
    if next_stage == STAGE_SETUP:
        if current_stage != STAGE_UNMASK:
            log(WARNING, "Restart from the setup stage")
    # If stage is not "setup",
    # the stage from `named_values` should be the expected next stage
    else:
        expected_next_stage = STAGES[(STAGES.index(current_stage) + 1) % len(STAGES)]
        if next_stage != expected_next_stage:
            raise ValueError(
                "Abort secure aggregation: "
                f"expect {expected_next_stage} stage, but receive {next_stage} stage"
            )


# pylint: disable-next=too-many-branches
def check_configs(stage: str, configs: Dict[str, ConfigsRecordValues]) -> None:
    """Check the validity of the configs."""
    # Check `named_values` for the setup stage
    if stage == STAGE_SETUP:
        key_type_pairs = [
            (KEY_SAMPLE_NUMBER, int),
            (KEY_SECURE_ID, int),
            (KEY_SHARE_NUMBER, int),
            (KEY_THRESHOLD, int),
            (KEY_CLIPPING_RANGE, float),
            (KEY_TARGET_RANGE, int),
            (KEY_MOD_RANGE, int),
        ]
        for key, expected_type in key_type_pairs:
            if key not in configs:
                raise KeyError(
                    f"Stage {STAGE_SETUP}: the required key '{key}' is "
                    "missing from the input `named_values`."
                )
            # Bool is a subclass of int in Python,
            # so `isinstance(v, int)` will return True even if v is a boolean.
            # pylint: disable-next=unidiomatic-typecheck
            if type(configs[key]) is not expected_type:
                raise TypeError(
                    f"Stage {STAGE_SETUP}: The value for the key '{key}' "
                    f"must be of type {expected_type}, "
                    f"but got {type(configs[key])} instead."
                )
    elif stage == STAGE_SHARE_KEYS:
        for key, value in configs.items():
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not isinstance(value[0], bytes)
                or not isinstance(value[1], bytes)
            ):
                raise TypeError(
                    f"Stage {STAGE_SHARE_KEYS}: "
                    f"the value for the key '{key}' must be a list of two bytes."
                )
    elif stage == STAGE_COLLECT_MASKED_INPUT:
        key_type_pairs = [
            (KEY_CIPHERTEXT_LIST, bytes),
            (KEY_SOURCE_LIST, int),
        ]
        for key, expected_type in key_type_pairs:
            if key not in configs:
                raise KeyError(
                    f"Stage {STAGE_COLLECT_MASKED_INPUT}: "
                    f"the required key '{key}' is "
                    "missing from the input `named_values`."
                )
            if not isinstance(configs[key], list) or any(
                elm
                for elm in cast(List[Any], configs[key])
                # pylint: disable-next=unidiomatic-typecheck
                if type(elm) is not expected_type
            ):
                raise TypeError(
                    f"Stage {STAGE_COLLECT_MASKED_INPUT}: "
                    f"the value for the key '{key}' "
                    f"must be of type List[{expected_type.__name__}]"
                )
    elif stage == STAGE_UNMASK:
        key_type_pairs = [
            (KEY_ACTIVE_SECURE_ID_LIST, int),
            (KEY_DEAD_SECURE_ID_LIST, int),
        ]
        for key, expected_type in key_type_pairs:
            if key not in configs:
                raise KeyError(
                    f"Stage {STAGE_UNMASK}: "
                    f"the required key '{key}' is "
                    "missing from the input `named_values`."
                )
            if not isinstance(configs[key], list) or any(
                elm
                for elm in cast(List[Any], configs[key])
                # pylint: disable-next=unidiomatic-typecheck
                if type(elm) is not expected_type
            ):
                raise TypeError(
                    f"Stage {STAGE_UNMASK}: "
                    f"the value for the key '{key}' "
                    f"must be of type List[{expected_type.__name__}]"
                )
    else:
        raise ValueError(f"Unknown secagg stage: {stage}")


def _setup(
    state: SecAggPlusState, configs: Dict[str, ConfigsRecordValues]
) -> Dict[str, ConfigsRecordValues]:
    # Assigning parameter values to object fields
    sec_agg_param_dict = configs
    state.sample_num = cast(int, sec_agg_param_dict[KEY_SAMPLE_NUMBER])
    state.sid = cast(int, sec_agg_param_dict[KEY_SECURE_ID])
    log(INFO, "Client %d: starting stage 0...", state.sid)

    state.share_num = cast(int, sec_agg_param_dict[KEY_SHARE_NUMBER])
    state.threshold = cast(int, sec_agg_param_dict[KEY_THRESHOLD])
    state.clipping_range = cast(float, sec_agg_param_dict[KEY_CLIPPING_RANGE])
    state.target_range = cast(int, sec_agg_param_dict[KEY_TARGET_RANGE])
    state.mod_range = cast(int, sec_agg_param_dict[KEY_MOD_RANGE])

    # Dictionaries containing client secure IDs as keys
    # and their respective secret shares as values.
    state.rd_seed_share_dict = {}
    state.sk1_share_dict = {}
    # Dictionary containing client secure IDs as keys
    # and their respective shared secrets (with this client) as values.
    state.ss2_dict = {}

    # Create 2 sets private public key pairs
    # One for creating pairwise masks
    # One for encrypting message to distribute shares
    sk1, pk1 = generate_key_pairs()
    sk2, pk2 = generate_key_pairs()

    state.sk1, state.pk1 = private_key_to_bytes(sk1), public_key_to_bytes(pk1)
    state.sk2, state.pk2 = private_key_to_bytes(sk2), public_key_to_bytes(pk2)
    log(INFO, "Client %d: stage 0 completes. uploading public keys...", state.sid)
    return {KEY_PUBLIC_KEY_1: state.pk1, KEY_PUBLIC_KEY_2: state.pk2}


# pylint: disable-next=too-many-locals
def _share_keys(
    state: SecAggPlusState, configs: Dict[str, ConfigsRecordValues]
) -> Dict[str, ConfigsRecordValues]:
    named_bytes_tuples = cast(Dict[str, Tuple[bytes, bytes]], configs)
    key_dict = {int(sid): (pk1, pk2) for sid, (pk1, pk2) in named_bytes_tuples.items()}
    log(INFO, "Client %d: starting stage 1...", state.sid)
    state.public_keys_dict = key_dict

    # Check if the size is larger than threshold
    if len(state.public_keys_dict) < state.threshold:
        raise ValueError("Available neighbours number smaller than threshold")

    # Check if all public keys are unique
    pk_list: List[bytes] = []
    for pk1, pk2 in state.public_keys_dict.values():
        pk_list.append(pk1)
        pk_list.append(pk2)
    if len(set(pk_list)) != len(pk_list):
        raise ValueError("Some public keys are identical")

    # Check if public keys of this client are correct in the dictionary
    if (
        state.public_keys_dict[state.sid][0] != state.pk1
        or state.public_keys_dict[state.sid][1] != state.pk2
    ):
        raise ValueError(
            "Own public keys are displayed in dict incorrectly, should not happen!"
        )

    # Generate the private mask seed
    state.rd_seed = os.urandom(32)

    # Create shares for the private mask seed and the first private key
    b_shares = create_shares(state.rd_seed, state.threshold, state.share_num)
    sk1_shares = create_shares(state.sk1, state.threshold, state.share_num)

    srcs, dsts, ciphertexts = [], [], []

    # Distribute shares
    for idx, (sid, (_, pk2)) in enumerate(state.public_keys_dict.items()):
        if sid == state.sid:
            state.rd_seed_share_dict[state.sid] = b_shares[idx]
            state.sk1_share_dict[state.sid] = sk1_shares[idx]
        else:
            shared_key = generate_shared_key(
                bytes_to_private_key(state.sk2),
                bytes_to_public_key(pk2),
            )
            state.ss2_dict[sid] = shared_key
            plaintext = share_keys_plaintext_concat(
                state.sid, sid, b_shares[idx], sk1_shares[idx]
            )
            ciphertext = encrypt(shared_key, plaintext)
            srcs.append(state.sid)
            dsts.append(sid)
            ciphertexts.append(ciphertext)

    log(INFO, "Client %d: stage 1 completes. uploading key shares...", state.sid)
    return {KEY_DESTINATION_LIST: dsts, KEY_CIPHERTEXT_LIST: ciphertexts}


# pylint: disable-next=too-many-locals
def _collect_masked_input(
    state: SecAggPlusState,
    configs: Dict[str, ConfigsRecordValues],
    fit: Callable[[], FitRes],
) -> Dict[str, ConfigsRecordValues]:
    log(INFO, "Client %d: starting stage 2...", state.sid)
    available_clients: List[int] = []
    ciphertexts = cast(List[bytes], configs[KEY_CIPHERTEXT_LIST])
    srcs = cast(List[int], configs[KEY_SOURCE_LIST])
    if len(ciphertexts) + 1 < state.threshold:
        raise ValueError("Not enough available neighbour clients.")

    # Decrypt ciphertexts, verify their sources, and store shares.
    for src, ciphertext in zip(srcs, ciphertexts):
        shared_key = state.ss2_dict[src]
        plaintext = decrypt(shared_key, ciphertext)
        actual_src, dst, rd_seed_share, sk1_share = share_keys_plaintext_separate(
            plaintext
        )
        available_clients.append(src)
        if src != actual_src:
            raise ValueError(
                f"Client {state.sid}: received ciphertext "
                f"from {actual_src} instead of {src}."
            )
        if dst != state.sid:
            raise ValueError(
                f"Client {state.sid}: received an encrypted message"
                f"for Client {dst} from Client {src}."
            )
        state.rd_seed_share_dict[src] = rd_seed_share
        state.sk1_share_dict[src] = sk1_share

    # Fit client
    fit_res = fit()
    parameters_factor = fit_res.num_examples
    parameters = parameters_to_ndarrays(fit_res.parameters)

    # Quantize parameter update (vector)
    quantized_parameters = quantize(
        parameters, state.clipping_range, state.target_range
    )

    quantized_parameters = parameters_multiply(quantized_parameters, parameters_factor)
    quantized_parameters = factor_combine(parameters_factor, quantized_parameters)

    dimensions_list: List[Tuple[int, ...]] = [a.shape for a in quantized_parameters]

    # Add private mask
    private_mask = pseudo_rand_gen(state.rd_seed, state.mod_range, dimensions_list)
    quantized_parameters = parameters_addition(quantized_parameters, private_mask)

    for client_id in available_clients:
        # Add pairwise masks
        shared_key = generate_shared_key(
            bytes_to_private_key(state.sk1),
            bytes_to_public_key(state.public_keys_dict[client_id][0]),
        )
        pairwise_mask = pseudo_rand_gen(shared_key, state.mod_range, dimensions_list)
        if state.sid > client_id:
            quantized_parameters = parameters_addition(
                quantized_parameters, pairwise_mask
            )
        else:
            quantized_parameters = parameters_subtraction(
                quantized_parameters, pairwise_mask
            )

    # Take mod of final weight update vector and return to server
    quantized_parameters = parameters_mod(quantized_parameters, state.mod_range)
    log(INFO, "Client %d: stage 2 completes. uploading masked parameters...", state.sid)
    return {
        KEY_MASKED_PARAMETERS: [ndarray_to_bytes(arr) for arr in quantized_parameters]
    }


def _unmask(
    state: SecAggPlusState, configs: Dict[str, ConfigsRecordValues]
) -> Dict[str, ConfigsRecordValues]:
    log(INFO, "Client %d: starting stage 3...", state.sid)

    active_sids = cast(List[int], configs[KEY_ACTIVE_SECURE_ID_LIST])
    dead_sids = cast(List[int], configs[KEY_DEAD_SECURE_ID_LIST])
    # Send private mask seed share for every avaliable client (including itclient)
    # Send first private key share for building pairwise mask for every dropped client
    if len(active_sids) < state.threshold:
        raise ValueError("Available neighbours number smaller than threshold")

    sids, shares = [], []
    sids += active_sids
    shares += [state.rd_seed_share_dict[sid] for sid in active_sids]
    sids += dead_sids
    shares += [state.sk1_share_dict[sid] for sid in dead_sids]

    log(INFO, "Client %d: stage 3 completes. uploading key shares...", state.sid)
    return {KEY_SECURE_ID_LIST: sids, KEY_SHARE_LIST: shares}
