import time
import grpc
from . import gRPC_communication_manager_pb2_grpc as gRPC_communication_manager_pb2_grpc
from concurrent import futures
from .gRPC_server import gRPCComServeFunc
from .message import Message


"""
This module implements the gRPCCommunicationManager class, which serves as the core communication manager
in a federated learning system. It encapsulates the functionality required to set up and manage a gRPC server,
handle client connections, and facilitate message passing between clients and the server.

The gRPCCommunicationManager class provides methods to initialize the server with specific configurations,
manage server lifecycle (start and stop operations), and handle client communications. It supports adding
communicators (clients), sending messages to specific or all connected clients, and receiving messages from clients.

Key functionalities include:
- Server initialization with customizable IP, port, and gRPC server configurations.
- Dynamic management of client communicators, allowing for the addition and retrieval of client addresses.
- Message sending capability, supporting both direct and broadcast modes to communicate with one or multiple clients.
- Message receiving functionality, enabling the server to process incoming messages from clients.
- Utilization of gRPC for underlying communication, ensuring efficient and reliable message exchange.

Dependencies:
- grpc: For implementing the gRPC server and client communication.
- concurrent.futures: For managing a pool of threads to handle client connections.
- gRPC_communication_manager_pb2_grpc, gRPC_server, message: Custom modules for gRPC service definitions, server functionality, and message handling.

This module is designed to be used as part of a larger federated learning framework, where it facilitates
the communication layer between distributed clients and a central server.

Example usage:
    # Initialize the communication manager with default settings
    comm_manager = gRPCCommunicationManager()
    # Start the server to listen for client connections
    comm_manager.serve()
    # Add a communicator (client)
    comm_manager.add_communicator('server', {'ip': '127.0.0.1', 'port': '50052'})
    # Send a message to a specific client
    message = Message(message_type=100, content={'ip': '127.0.0.1', 'port': '50051'})
    comm_manager.send(message, 'server')
"""


class gRPCCommunicationManager(object):
    """
    Manages the gRPC server for federated learning system communications, handling client connections,
    and facilitating message exchange between clients and the server.

    This class is responsible for initializing the gRPC server with specific configurations, managing
    the server's lifecycle (including start and stop operations), and handling communications with clients.
    It supports adding communicators (clients), sending messages to specific or all connected clients,
    and receiving messages from clients.

    Attributes:
        ip (str): The IP address on which the gRPC server will listen.
        port (str): The port number on which the gRPC server will listen.
        max_connection_num (int): The maximum number of concurrent connections the server will accept, which should be the number of clients in a federated learning system.
        gRPC_config (dict): Configuration options for the gRPC server. Includes settings for message length (default 300M), HTTP proxy (default False), and compression (default no compression).
        compression_method (grpc.Compression): The compression method used by the gRPC server.
        communicators (dict): A dictionary of client communicators, mapping client IDs to their addresses.

    """
    def __init__(
            self,
            ip: str = "127.0.0.1",
            port: str = "50051",
            max_connection_num: int = 1,
            gRPC_config=None
    ):
        if gRPC_config is None:
            gRPC_config = {
                "grpc_max_send_message_length": 300 * 1024 * 1024,
                "grpc_max_receive_message_length": 300 * 1024 * 1024,
                "grpc_enable_http_proxy": False,
                "grpc_compression": "no_compression"
            }
        self._ip = ip
        self._port = port
        self._max_connection_num = max_connection_num
        self._gRPC_config = gRPC_config
        self.server_funcs = gRPCComServeFunc()
        options = [
            ("grpc.max_send_message_length", gRPC_config["grpc_max_send_message_length"]),
            ("grpc.max_receive_message_length", gRPC_config["grpc_max_receive_message_length"]),
            ("grpc.enable_http_proxy", gRPC_config["grpc_enable_http_proxy"]),
        ]
        if gRPC_config["grpc_compression"].lower() == 'deflate':
            self._compression_method = grpc.Compression.Deflate
        elif gRPC_config["grpc_compression"].lower() == 'gzip':
            self._compression_method = grpc.Compression.Gzip
        else:
            self._compression_method = grpc.Compression.NoCompression
        self._gRPC_server = self._serve(
            max_workers=max_connection_num,
            ip=ip,
            port=port,
            options=options
        )
        self._communicators = dict()

    @property
    def ip(self):
        """
        Get the IP address of the gRPC server.

        Returns:
            str: The IP address of the gRPC server.
        """
        return self._ip

    @property
    def port(self):
        """
        Get the port number of the gRPC server.

        Returns:
            str: The port number of the gRPC server.
        """
        return self._port

    @property
    def max_connection_num(self):
        """
        Get the maximum number of concurrent connections the server will accept.

        Returns:
            int: The maximum number of concurrent connections the server will accept.
        """
        return self._max_connection_num

    @property
    def gRPC_config(self):
        """
        Get the configuration options for the gRPC server.

        Returns:
            dict: The configuration options for the gRPC server.

        """
        return self._gRPC_config

    @property
    def compression_method(self):
        """
        Get the compression method used by the gRPC server.

        Returns:
            grpc.Compression: The compression method used by the gRPC server.
        """
        return self._compression_method

    @property
    def communicators(self):
        """
        Get the dictionary of client communicators.

        Returns:
            dict: A dictionary of client communicators, mapping client IDs to their addresses.

        """
        return self._communicators

    @communicators.setter
    def communicators(self, value):
        self._communicators = value

    def _serve(self, max_workers: int, ip: str, port: str, options: list | None):
        """
        Start the gRPC server with the specified configurations.

        Args:
            max_workers: maximum number of concurrent connections the server will accept
            ip: IP address on which the gRPC server will listen
            port: port number on which the gRPC server will listen
            options: configuration options for the gRPC server

        Returns:
            grpc.Server: The gRPC server instance that has been started.

        """
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            compression=self._compression_method,
            options=options
        )
        gRPC_communication_manager_pb2_grpc.add_gRPCComServeFuncServicer_to_server(
            self.server_funcs, server)
        server.add_insecure_port("{}:{}".format(ip, port))
        server.start()

        return server

    def terminate_server(self):
        """
        Stop the gRPC server and terminate the server process.

        Returns:
            None

        """
        self._gRPC_server.stop(grace=None)

    def add_communicator(self, communicator_id: str, communicator_address: dict | str):
        """
        Add a communicator (client) to the server's list of communicators.

        Args:
            communicator_id: The ID of the communicator (client)
            communicator_address: The address of the communicator (client), specified as a dictionary (containing key 'ip' and 'port') or string (formating as 'ip:port').

        Raises:
            TypeError: If the type of communicator_address is not supported.

        Returns:
            None

        Examples:
            >>> comm_manager = gRPCCommunicationManager(max_connection_num=2)
            >>> comm_manager.add_communicator('client1', {'ip':'127.0.0.1', 'port':'50052'})
            >>> comm_manager.communicators
            {'client1': '127.0.0.1:50052'}
            >>> comm_manager.add_communicator('client2', '127.0.0.1:50053')
            >>> comm_manager.communicators
            {'client1': '127.0.0.1:50052', 'client2': '127.0.0.1:50053'}

        """
        if isinstance(communicator_address, dict):
            self._communicators[communicator_id] = f"{communicator_address['ip']}:{communicator_address['port']}"
        elif isinstance(communicator_address, str):
            self._communicators[communicator_id] = communicator_address
        else:
            raise TypeError(f"The type of communicator_address ({type(communicator_address)}) is not supported")

    def get_communicators(self, communicator_id: str | list | None = None):
        """
        Get the address of a specific communicator (client) or all communicators.

        Args:
            communicator_id: The ID of the communicator (client) to retrieve the address for, or None to get all communicators.

        Returns:
            dict | str: The address of the specified communicator (client) or all communicators.

        """
        address = dict()
        if communicator_id:
            if isinstance(communicator_id, list):
                for each_communicator in communicator_id:
                    address[each_communicator] = self.get_communicators(each_communicator)
                return address
            else:
                return self._communicators[communicator_id]
        else:
            return self._communicators

    def _create_stub(self, receiver_address: str):
        """
        Create a gRPC stub for the specified receiver address.

        Args:
            receiver_address: The address of the receiver to create the stub for.

        Returns:
            gRPCComServeFuncStub: The gRPC stub for the specified receiver address.
            grpc.Channel: The gRPC channel associated with the stub.

        """
        channel = grpc.insecure_channel(receiver_address,
                                        compression=self.compression_method,
                                        # options=(('grpc.enable_http_proxy',
                                        #           0),)
                                        )
        stub = gRPC_communication_manager_pb2_grpc.gRPCComServeFuncStub(channel)
        return stub, channel

    def _send(self, receiver_address: str, message: Message, max_retry: int = 3):
        """
        Send a message to the specified receiver address with optional retry mechanism.

        Args:
            receiver_address: The address of the receiver to send the message to.
            message: The message to send.
            max_retry: The maximum number of retry attempts in case of failure.

        Returns:
            None

        """
        request = message.transform(to_list=True)
        attempts = 0
        retry_interval = 1
        success_flag = False
        while attempts < max_retry:
            stub, channel = self._create_stub(receiver_address)
            try:
                stub.sendMessage(request)
                channel.close()
                success_flag = True
            except grpc._channel._InactiveRpcError as error:
                attempts += 1
                time.sleep(retry_interval)
                retry_interval *= 2
            finally:
                channel.close()
            if success_flag:
                break
            else:
                raise ConnectionError(f"Failed to send message to {receiver_address}")

    def send(self, message: Message, receiver: str | list | None = None):
        """
        Send a message to the specified receiver(s).

        Args:
            message: The message to send.
            receiver: The ID of the receiver(s) to send the message to, or None to broadcast to all communicators.

        Returns:
            None

        """
        if receiver is not None:
            if not isinstance(receiver, list):
                receiver = [receiver]
            for each_receiver in receiver:
                if each_receiver in self._communicators.keys():
                    receiver_address = self._communicators[each_receiver]
                    self._send(receiver_address, message)
        else:
            for each_receiver in self._communicators.keys():
                receiver_address = self._communicators[each_receiver]
                self._send(receiver_address, message)

    def receive(self):
        """
        Receive a message from a client.

        Returns:
            Message: The received message from the client.

        """
        received_message = self.server_funcs.receive()
        message = Message()
        message.parse(received_message.msg)
        return message
