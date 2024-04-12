# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
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
"""Client-side message handler."""


from typing import Optional, Tuple, cast

from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.client.typing import ClientFn
from flwr.common.configsrecord import ConfigsRecord
from flwr.common.constant import (
    TASK_TYPE_EVALUATE,
    TASK_TYPE_FIT,
    TASK_TYPE_GET_PARAMETERS,
    TASK_TYPE_GET_PROPERTIES,
)
from flwr.common.context import Context
from flwr.common.message import Message, Metadata
from flwr.common.recordset import RecordSet
from flwr.common.recordset_compat import (
    evaluateres_to_recordset,
    fitres_to_recordset,
    getparametersres_to_recordset,
    getpropertiesres_to_recordset,
    recordset_to_evaluateins,
    recordset_to_fitins,
    recordset_to_getparametersins,
    recordset_to_getpropertiesins,
)
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    Reason,
    ServerMessage,
)


class UnexpectedServerMessage(Exception):
    """Exception indicating that the received message is unexpected."""


class UnknownServerMessage(Exception):
    """Exception indicating that the received message is unknown."""


def handle_control_message(message: Message) -> Tuple[Optional[Message], int]:
    """Handle control part of the incoming message.

    Parameters
    ----------
    message : Message
        The Message coming from the server, to be processed by the client.

    Returns
    -------
    message : Optional[Message]
        Message to be sent back to the server. If None, the client should
        continue to process messages from the server.
    sleep_duration : int
        Number of seconds that the client should disconnect from the server.
    """
    if message.metadata.task_type == "reconnect":
        # Retrieve ReconnectIns from recordset
        recordset = message.message
        seconds = cast(int, recordset.get_configs("config")["seconds"])
        # Construct ReconnectIns and call _reconnect
        disconnect_msg, sleep_duration = _reconnect(
            ServerMessage.ReconnectIns(seconds=seconds)
        )
        # Store DisconnectRes in recordset
        reason = cast(int, disconnect_msg.disconnect_res.reason)
        recordset = RecordSet()
        recordset.set_configs("config", ConfigsRecord({"reason": reason}))
        out_message = Message(
            metadata=Metadata(
                run_id=0,
                task_id="",
                group_id="",
                ttl="",
                task_type="reconnect",
            ),
            message=recordset,
        )
        # Return TaskRes and sleep duration
        return out_message, sleep_duration

    # Any other message
    return None, 0


def handle_legacy_message_from_tasktype(
    client_fn: ClientFn, message: Message, context: Context
) -> Message:
    """Handle legacy message in the inner most middleware layer."""
    client = client_fn("-1")

    client.set_context(context)

    task_type = message.metadata.task_type

    # Handle GetPropertiesIns
    if task_type == TASK_TYPE_GET_PROPERTIES:
        get_properties_res = maybe_call_get_properties(
            client=client,
            get_properties_ins=recordset_to_getpropertiesins(message.message),
        )
        out_recordset = getpropertiesres_to_recordset(get_properties_res)
    # Handle GetParametersIns
    elif task_type == TASK_TYPE_GET_PARAMETERS:
        get_parameters_res = maybe_call_get_parameters(
            client=client,
            get_parameters_ins=recordset_to_getparametersins(message.message),
        )
        out_recordset = getparametersres_to_recordset(
            get_parameters_res, keep_input=False
        )
    # Handle FitIns
    elif task_type == TASK_TYPE_FIT:
        fit_res = maybe_call_fit(
            client=client,
            fit_ins=recordset_to_fitins(message.message, keep_input=True),
        )
        out_recordset = fitres_to_recordset(fit_res, keep_input=False)
    # Handle EvaluateIns
    elif task_type == TASK_TYPE_EVALUATE:
        evaluate_res = maybe_call_evaluate(
            client=client,
            evaluate_ins=recordset_to_evaluateins(message.message, keep_input=True),
        )
        out_recordset = evaluateres_to_recordset(evaluate_res)
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    # Return Message
    out_message = Message(
        metadata=Metadata(
            run_id=0,  # Non-user defined
            task_id="",  # Non-user defined
            group_id="",  # Non-user defined
            ttl="",
            task_type=task_type,
        ),
        message=out_recordset,
    )
    return out_message


def _reconnect(
    reconnect_msg: ServerMessage.ReconnectIns,
) -> Tuple[ClientMessage, int]:
    # Determine the reason for sending DisconnectRes message
    reason = Reason.ACK
    sleep_duration = None
    if reconnect_msg.seconds is not None:
        reason = Reason.RECONNECT
        sleep_duration = reconnect_msg.seconds
    # Build DisconnectRes message
    disconnect_res = ClientMessage.DisconnectRes(reason=reason)
    return ClientMessage(disconnect_res=disconnect_res), sleep_duration
