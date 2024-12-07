
from collections import deque
from . import gRPC_communication_manager_pb2 as gRPC_communication_manager_pb2
from . import gRPC_communication_manager_pb2_grpc as gRPC_communication_manager_pb2_grpc


"""
This module implements a gRPC server for a federated learning system, facilitating communication
between clients and the server. It defines a gRPCComServeFunc class that inherits from the gRPC
generated servicer class, enabling the server to handle incoming messages from clients,queue them,
and process them as needed.

The gRPCComServeFunc class provides methods for receiving messages from clients and acknowledging
their receipt. It uses a deque to maintain a queue of messages, ensuring that messages are processed
in the order they are received. The class exposes two main RPC methods: sendMessage, which clients call
to send messages to the server, and receive, which the server uses internally to dequeue and process messages.

This server component is designed to be integrated into a larger federated learning system, where it
can manage communications between multiple clients participating in the learning process. It is
responsible for handling message serialization and deserialization, managing the message queue, and
ensuring reliable communication between the server and clients.

Dependencies:
- collections.deque: For efficiently managing the message queue.
- gRPC_communication_manager_pb2, gRPC_communication_manager_pb2_grpc: For gRPC communication, including message structures and service definitions.

Note:
This module requires the gRPC framework and the protobuf definitions for the federated learning system
to be properly set up and compiled. It is part of the communication layer of the federated learning framework,
enabling asynchronous, reliable communication between distributed clients and the server.
"""


class gRPCComServeFunc(gRPC_communication_manager_pb2_grpc.gRPCComServeFuncServicer):
    """
        A gRPC server class that inherits from the generated gRPCComServeFuncServicer class.

        This class is responsible for handling incoming messages from clients, queuing them, and
        processing them as needed within a federated learning system.

        The class uses a deque to maintain a queue of messages, ensuring FIFO (First In, First Out)
        processing. It provides methods to receive messages from clients and to process or dequeue
        these messages for further handling.

        Attributes:
            message_queue (collections.deque): A thread-safe queue that holds incoming messages from clients.
        """
    def __init__(self):
        self.message_queue = deque()

    def sendMessage(self, request, context):
        """
        Receives an incoming message from a client and adds it to the message queue.
        Args:
            request: The incoming message from the client.
            context: The context of the gRPC request.

        Returns:
            A MessageResponse object with an acknowledgment message.
        """
        self.message_queue.append(request)

        return gRPC_communication_manager_pb2.MessageResponse(msg='ACK')

    def receive(self):
        """
        Retrieves the next message from the message queue and returns it.
        Returns:
            The next message in the queue.
        """
        while len(self.message_queue) == 0:
            continue
        message = self.message_queue.popleft()
        return message
