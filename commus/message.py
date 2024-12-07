import pickle
import base64
import numpy as np
from typing import Any
from . import gRPC_communication_manager_pb2 as gRPC_communication_manager_pb2
from datetime import datetime
from pympler import asizeof


"""
This module defines the Message class and associated functions for serializing, deserializing,
and managing messages in a federated learning system. It supports operations such as creating
messages of various types (e.g., for establishing connections, updating model parameters, and
transferring metrics), serializing and deserializing message contents, and calculating the size
of messages for network transmission.

The Message class encapsulates the details of a message, including its type, sender, receiver,
content, and other metadata. It provides methods for setting and getting these attributes,
transforming message content into a format suitable for transmission (including serialization
of complex objects like model parameters), and parsing received messages.

Serialization and deserialization leverage base64 encoding and the pickle module to handle
complex data types, such as model parameters represented as numpy arrays. The module also
integrates with gRPC for communication, using protobuf definitions for structured message
exchange.

Additionally, utility functions are provided for transforming message contents into lists or
dictionaries, building protobuf message objects from Python data structures, and parsing
protobuf messages back into Python objects. This facilitates the exchange of rich, structured
data over the network in a federated learning context.

Dependencies:
- pickle: For serializing and deserializing Python object structures.
- base64: For encoding binary data as ASCII strings.
- numpy: For handling numerical operations on arrays, used in model parameters.
- typing: For type hints in function signatures.
- gRPC_communication_manager_pb2: Protobuf definitions for structured message exchange.
- datetime: For timestamping messages.
- pympler: For estimating the size of Python objects in bytes.

Note:
This module is designed for use in a federated learning system and assumes a gRPC-based
communication mechanism. It should be integrated with a gRPC server and client setup for
full functionality. The message types are defined as follows:
- 100: Build federated learning connection.
- 101: End federated learning connection.
- 200: Update model parameters (content should contain a 'model' key with model parameters).
- 300: Transfer metrics.
"""


def b64serializer(x: Any) -> bytes:
    """
    Serialize the input data to bytes using base64 encoding.

    Args:
        x (Any): The input data to be serialized.

    Returns:
        bytes: The serialized data in bytes.

    Example:
        >>> b64serializer('Hello, World!')
        b'gASVEQAAAAAAAACMDUhlbGxvLCBXb3JsZCGULg=='
        >>> b64serializer({'data': [1, 2, 3]})
        b'gASVFQAAAAAAAAB9lIwEZGF0YZRdlChLAUsCSwNlcy4='
        >>> b64serializer(np.array([1, 2, 3]))
        b'gASVoAAAAAAAAACMFW51bXB5LmNvcmUubXVsdGlhcnJheZSMDF9yZWNvbnN0cnVjdJSTlIwFbnVtcHmUjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwOFlGgDjAVkdHlwZZSTlIwCaTiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYolDGAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAJR0lGIu'

    """
    return base64.b64encode(pickle.dumps(x))


def b64deserializer(x: bytes) -> Any:
    """
    Deserialize the input bytes to the original data using base64 decoding.

    Args:
        x (bytes): The input bytes to be deserialized.

    Returns:
        Any: The deserialized data.

    Example:
        >>> b64deserializer(b'gASVEQAAAAAAAACMDUhlbGxvLCBXb3JsZCGULg==')
        'Hello, World!'
        >>> b64deserializer(b'gASVFQAAAAAAAAB9lIwEZGF0YZRdlChLAUsCSwNlcy4=')
        {'data': [1, 2, 3]}
        >>> b64deserializer(b'gASVoAAAAAAAAACMFW51bXB5LmNvcmUubXVsdGlhcnJheZSMDF9yZWNvbnN0cnVjdJSTlIwFbnVtcHmUjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwOFlGgDjAVkdHlwZZSTlIwCaTiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYolDGAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAJR0lGIu')
        array([1, 2, 3])

    """
    return pickle.loads(base64.b64decode(x))


class Message(object):
    """
        Represents a message in a federated learning system.

        This class encapsulates details such as the message type, sender, receiver, content, and the communication
        round. This class provides methods for setting and getting these attributes, serializing complex objects
        like model parameters for transmission, and parsing received messages.

        The class also includes methods for transforming message content into a format suitable for
        transmission (including serialization of complex objects like model parameters using base64 encoding),
        building protobuf message objects from Python data structures, and parsing protobuf messages back
        into Python objects. This facilitates the exchange of rich, structured data over the network in a
        federated learning context.

        Attributes:
            message_type (int): The type of the message, indicating its purpose (e.g., connection establishment, model update).
            sender (str): Identifier of the sender.
            receiver (str | list): Identifier(s) of the receiver(s). Can be a single ID or a list of IDs.
            content (Any): The content of the message, which can be a simple message or a complex structure like model parameters.
            communication_round (int): The current round of communication in the federated learning process.
            timestamp (float): The timestamp of the message creation.

        Notes: The message types are defined as follows:
            - 100: Build federated learning connection.
            - 101: End federated learning connection.
            - 200: Update model parameters (content should contain a 'model' key with model parameters).
            - 300: Transfer metrics.


    """

    def __init__(
            self,
            message_type: int = -1,
            sender: str = "-1",
            receiver: str | list[str] = "-1",
            content: Any = "",
            communication_round: int = 0
    ):
        self._message_type = message_type
        self._sender = sender
        self._receiver = receiver
        self._content = content
        self._communication_round = communication_round
        self._timestamp = datetime.now().timestamp()
        self._param_serializer = b64serializer
        self._param_deserializer = b64deserializer

    @property
    def message_type(self):
        """
            Get the message type.
        Returns:
            int: The message type.
        """
        return self._message_type

    @message_type.setter
    def message_type(self, value):
        self._message_type = value

    @property
    def sender(self):
        """
            Get the sender of the message.
        Returns:
            str: The sender of the message
        """
        return self._sender

    @sender.setter
    def sender(self, value):
        self._sender = value

    @property
    def receiver(self):
        """
            Get the receiver(s) of the message.
        Returns:
            str | list: The receiver(s) of the message

        """
        return self._receiver

    @receiver.setter
    def receiver(self, value):
        self._receiver = value

    @property
    def content(self):
        """
            Get the content of the message.

        Returns:
            Any: The content of the message

        """
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    @property
    def communication_round(self):
        """
            Get the communication round of the message.

        Returns:
            int: The communication round of the message
        """
        return self._communication_round

    @communication_round.setter
    def communication_round(self, value):
        self._communication_round = value

    @property
    def timestamp(self):
        """
            Get the timestamp of the message.

        Returns:
            float: The timestamp of the message
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value

    def __lt__(self, other):
        """
            Compare two messages based on their timestamps and communication rounds.

        Args:
            other: Another message object to compare with.

        Returns:
            bool: True if this message is less than the other message, False otherwise.

        """
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        else:
            return self.communication_round < other.communication_round

    def _create_by_type(self, value: Any, nested: bool = False) -> gRPC_communication_manager_pb2.MsgValue:
        """
            Create a protobuf message object based on the type of the input value.

        Args:
            value: The input value to be converted to a protobuf message object.
            nested: A flag indicating whether the value is nested within another message object.

        Returns:
            gRPC_communication_manager_pb2.MsgValue: The protobuf message object representing the input value.

        """
        if isinstance(value, dict):
            if isinstance(list(value.keys())[0], str):
                m_dict = gRPC_communication_manager_pb2.mDict_keyIsString()
                key_type = 'string'
            else:
                m_dict = gRPC_communication_manager_pb2.mDict_keyIsInt()
                key_type = 'int'
            for key in value.keys():
                m_dict.dict_value[key].MergeFrom(
                    self._create_by_type(value[key], nested=True))
            if nested:
                msg_value = gRPC_communication_manager_pb2.MsgValue()
                if key_type == 'string':
                    msg_value.dict_msg_string_key.MergeFrom(m_dict)
                else:
                    msg_value.dict_msg_int_key.MergeFrom(m_dict)
                return msg_value
            else:
                return m_dict
        elif isinstance(value, list) or isinstance(value, tuple):
            m_list = gRPC_communication_manager_pb2.mList()
            for each in value:
                m_list.list_value.append(self._create_by_type(each, nested=True))
            if nested:
                msg_value = gRPC_communication_manager_pb2.MsgValue()
                msg_value.list_msg.MergeFrom(m_list)
                return msg_value
            else:
                return m_list
        else:
            m_single = gRPC_communication_manager_pb2.mSingle()
            if type(value) in [int, np.int32]:
                m_single.int_value = value
            elif type(value) in [str, bytes]:
                m_single.str_value = value
            elif type(value) in [float, np.float32]:
                m_single.float_value = value
            else:
                raise ValueError(
                    f'The data type {type(value)} has not been supported.')

            if nested:
                msg_value = gRPC_communication_manager_pb2.MsgValue()
                msg_value.single_msg.MergeFrom(m_single)
                return msg_value
            else:
                return m_single

    def _transform_to_list(self, x: Any) -> Any:
        """
            Transform the input data into a list format.

        Args:
            x: The input data to be transformed.

        Returns:
            Any: The transformed data in list format.

        Examples:
            >>> message = Message(content='model')
            >>> message._transform_to_list(message.content)
            'model'
            >>> message = Message(content=[(1, 2, 3), (2, 3, 4), (3, 4, 5)])
            >>> message._transform_to_list(message.content)
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
            >>> message = Message(content={'model': [1, 2, 3]})
            >>> message._transform_to_list(message.content)
            {'model': [1, 2,3]}
            >>> message = Message(content={'model': np.array([1, 2, 3])})
            >>> message._transform_to_list(message.content)
            {'model': b'gASVoAAAAAAAAACMFW51bXB5LmNvcmUubXVsdGlhcnJheZSMDF9yZWNvbnN0cnVjdJSTlIwFbnVtcHmUjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwOFlGgDjAVkdHlwZZSTlIwCaTiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYolDGAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAJR0lGIu'}

        """
        if isinstance(x, list) or isinstance(x, tuple):
            return [self._transform_to_list(each_x) for each_x in x]
        elif isinstance(x, dict):
            for key in x.keys():
                x[key] = self._transform_to_list(x[key])
            return x
        else:
            if hasattr(x, 'tolist'):
                return self._param_serializer(x)
            else:
                return x

    def _build_msg_value(self, value):
        """
            Build a protobuf message object based on the type of the input value.

        Args:
            value: The input value to be converted to a protobuf message object.

        Returns:
            gRPC_communication_manager_pb2.MsgValue: The protobuf message object representing the input value.

        """
        msg_value = gRPC_communication_manager_pb2.MsgValue()

        if isinstance(value, list) or isinstance(value, tuple):
            msg_value.list_msg.MergeFrom(self._create_by_type(value))
        elif isinstance(value, dict):
            if isinstance(list(value.keys())[0], str):
                msg_value.dict_msg_string_key.MergeFrom(
                    self._create_by_type(value))
            else:
                msg_value.dict_msg_int_key.MergeFrom(self._create_by_type(value))
        else:
            msg_value.single_msg.MergeFrom(self._create_by_type(value))

        return msg_value

    def transform(self, to_list: bool = False):
        """
            Transform the message into a protobuf message object for transmission.

        Args:
            to_list(bool): A flag indicating whether to transform the message's content into a list format. It is necessary if the content contains complex objects like numpy arrays, torch tensor and so on.

        Returns:
            gRPC_communication_manager_pb2.MessageRequest: The protobuf message object representing the message.

        Notes:
            Note that this operation might change the content of the message based on the to_list flag!
            If you want to transfer model parameters, you should put the model parameters into the value
            of the 'model' key in the content and set the to_list flag to True. The model parameters can
            be numpy arrays, torch tensors, torch state_dicts that containing tensors, etc.

        Examples:
            >>> message = Message(message_type=200, sender='0', receiver='1', content={'model': np.array([1, 2, 3])}, communication_round=0)
            >>> message.transform(to_list=True)
            msg {
              key: "timestamp"
              value {
                single_msg {
                  float_value: 1.7212119e+09
                }
              }
            }
            msg {
              key: "sender"
              value {
                single_msg {
                  str_value: "0"
                }
              }
            }
            msg {
              key: "receiver"
              value {
                single_msg {
                  str_value: "1"
                }
              }
            }
            msg {
              key: "message_type"
              value {
                single_msg {
                  int_value: 200
                }
              }
            }
            msg {
              key: "content"
              value {
                dict_msg_string_key {
                  dict_value {
                    key: "model"
                    value {
                      single_msg {
                        str_value: "gASVoAAAAAAAAACMFW51bXB5LmNvcmUubXVsdGlhcnJheZSMDF9yZWNvbnN0cnVjdJSTlIwFbnVtcHmUjAduZGFycmF5lJOUSwCFlEMBYpSHlFKUKEsBSwOFlGgDjAVkdHlwZZSTlIwCaTiUiYiHlFKUKEsDjAE8lE5OTkr/////Sv////9LAHSUYolDGAEAAAAAAAAAAgAAAAAAAAADAAAAAAAAAJR0lGIu"
                      }
                    }
                  }
                }
              }
            }
            msg {
              key: "communication_round"
              value {
                single_msg {
                  int_value: 0
                }
              }
            }
        """
        if to_list:
            self.content = self._transform_to_list(self.content)

        split_message = gRPC_communication_manager_pb2.MessageRequest()  # map/dict
        split_message.msg['message_type'].MergeFrom(
            self._build_msg_value(self.message_type))
        split_message.msg['sender'].MergeFrom(self._build_msg_value(self.sender))
        split_message.msg['receiver'].MergeFrom(
            self._build_msg_value(self.receiver))
        split_message.msg['content'].MergeFrom(self._build_msg_value(
            self.content))
        split_message.msg['communication_round'].MergeFrom(self._build_msg_value(self.communication_round))
        split_message.msg['timestamp'].MergeFrom(
            self._build_msg_value(self.timestamp))
        return split_message

    def _parse_msg(self, value):
        """
            Parse the input value based on its type.

        Args:
            value: The input value to be parsed.

        Returns:
            Any: The parsed value.

        """
        if isinstance(value, gRPC_communication_manager_pb2.MsgValue) or \
                isinstance(value, gRPC_communication_manager_pb2.mSingle):
            return self._parse_msg(getattr(value, value.WhichOneof("type")))
        elif isinstance(value, gRPC_communication_manager_pb2.mList):
            return [self._parse_msg(each) for each in value.list_value]
        elif isinstance(value, gRPC_communication_manager_pb2.mDict_keyIsString) or \
                isinstance(value, gRPC_communication_manager_pb2.mDict_keyIsInt):
            return {
                k: self._parse_msg(value.dict_value[k])
                for k in value.dict_value
            }
        else:
            return value

    def parse_model(self, value):
        """
            Parse the input value as a model.

        Args:
            value: The input value to be parsed.

        Returns:
            Any: The parsed model value.

        """
        if isinstance(value, dict):
            return {
                k: self.parse_model(value[k]) for k in value.keys()
            }
        elif isinstance(value, list):
            return [self.parse_model(each) for each in value]
        elif isinstance(value, str):
            return self._param_deserializer(value.encode())
        else:
            return value

    def parse(self, received_msg):
        """
            Parse the received message into its components.

        Args:
            received_msg: The received message to be parsed.

        Returns:
            None

        Notes:
            If the content of the message contains 'model' key, the value of 'model' key will be parsed as a model
            using base64 decoding.

        Examples:
            >>> message = Message(message_type=200, sender='0', receiver='1', content={'model': np.array([1, 2, 3])}, communication_round=0)
            >>> transform_message = message.transform(to_list=True)
            >>> received_message = Message()
            >>> received_message.parse(transform_message.msg)
            >>> received_message.message_type
            200
            >>> received_message.content['model']
            array([1, 2, 3])

        """
        self.message_type = self._parse_msg(received_msg['message_type'])
        self.sender = self._parse_msg(received_msg['sender'])
        self.receiver = self._parse_msg(received_msg['receiver'])
        self.communication_round = self._parse_msg(received_msg['communication_round'])
        self.content = self._parse_msg(received_msg['content'])
        if isinstance(self.content, dict) and "model" in self.content.keys():
            self.content["model"] = self.parse_model(self.content["model"])
        self.timestamp = self._parse_msg(received_msg['timestamp'])

    def count_bytes(self):
        """
            Calculate the message bytes to be sent/received.

        Returns:
            tuple: Tuple of bytes of the message to be sent and received.

        """
        download_bytes = asizeof.asizeof(self.content)
        upload_cnt = len(self.receiver) if isinstance(self.receiver,
                                                      list) else 1
        upload_bytes = download_bytes * upload_cnt
        return download_bytes, upload_bytes
