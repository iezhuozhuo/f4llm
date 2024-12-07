# """
# This module defines a Registry class that serves as a central repository for various components used in a federated learning framework. It allows for the dynamic registration and retrieval of components such as models, data managers, loss functions, metrics, trainers, and evaluators. This flexibility facilitates the easy extension and customization of the federated learning framework by enabling the addition of new components without modifying the core codebase.
#
# The Registry class uses class methods to register and retrieve components, ensuring that the registration process is centralized and accessible globally throughout the application. It supports the registration of components under specific categories, such as models, data managers, and trainers, among others. Additionally, it provides methods to unregister components and to retrieve a list of all registered keys, aiding in dynamic configuration and introspection of the federated learning system.
#
# Key Features:
# - Centralized management of federated learning components.
# - Support for dynamic registration and retrieval of models, data managers, loss functions, metrics, trainers, and evaluators.
# - Facilitation of framework extension and customization.
# - Global accessibility through class methods.
#
# Usage:
# The Registry class is typically used to register components at the beginning of the application's lifecycle. Components are then retrieved and instantiated as needed during the execution of the federated learning tasks.
#
# Example:
#     # Register a model
#     Registry.register_model('example_model', ExampleModelClass)
#
#     # Retrieve the registered model class
#     model_class = Registry.get_model_class('example_model')
#     model_instance = model_class(...)
# """


class Registry:
    """
    The Registry class serves as a central repository for various components used in a federated learning framework
    such as models, data managers, loss functions, metrics, trainers, and evaluators.

    It allows for the dynamic registration and retrieval of components, facilitating the extension and customization
    of the federated learning framework.
    """
    mapping = {
        "state": {},
        "fedtrainer_name_mapping": {},
        "loctrainer_name_mapping": {},
        "loss_name_mapping": {},
        "data_name_mapping": {},
        "model_name_mapping": {},
        "metric_name_mapping": {},
        "eval_name_mapping": {}
    }

    @classmethod
    def register(cls, name, obj):
        """
        Register a component with the given name and object.

        Args:
            name (str): The name of the component.
            obj (object): The object to be registered.

        Returns:
            None

        """
        cls.mapping["state"][name] = obj

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        """
        Get the component registered with the given name.

        Args:
            name (str): The name of the component.
            default (object): The default value to return if the component is not found.
            no_warning (bool): Whether to suppress the warning message if the component is not found.

        Returns:
            object: The component registered with the given name, or the default value if not found.

        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
                "writer" in cls.mapping["state"]
                and value == default
                and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def register_loss(cls, name):
        """
        Register a loss function with the given name.

        Args:
            name (str): The name of the loss function.

        Returns:
            wrap: The wrapper function to register the loss

        """
        def wrap(func):
            # from utils.loss import BaseLoss
            #
            # assert issubclass(
            #     func, BaseLoss
            # ), "All loss must inherit utils.loss.BaseLoss class"
            # cls.mapping["loss_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_data(cls, name):
        """
        Register a data manager with the given name.

        Args:
            name (str): The name of the data manager.

        Returns:
            wrap: The wrapper function to register the data manager.

        """
        def wrap(func):
            from datas.base_data import FedBaseDataManger

            assert issubclass(
                func, FedBaseDataManger
            ), "All dataset must inherit data.base_data_loader.BaseDataLoader class"
            cls.mapping["data_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        """
        Register a model with the given name.

        Args:
            name (str): The name of the model.

        Returns:
            wrap: The wrapper function to register the model

        """
        def wrap(func):
            from models.base_model import BaseModels

            assert issubclass(
                func, BaseModels
            ), "All model must inherit models.base_models.BaseModels class"
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_fedtrainer(cls, name):
        """
        Register a federated trainer with the given name.

        Args:
            name (str): The name of the federated trainer.

        Returns:
            wrap: The wrapper function to register the federated trainer

        """
        def wrap(func):
            from trainers.FedBaseTrainer import BaseTrainer
            assert issubclass(
                func, BaseTrainer
            ), "All federated algorithm must inherit trainers.FedBaseTrainer.BaseTrainer class"
            cls.mapping["fedtrainer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_loctrainer(cls, name):
        """
        Register a local trainer with the given name.

        Args:
            name (str): The name of the local trainer.

        Returns:
            wrap: The wrapper function to register the local trainer

        """
        def wrap(func):
            # from trainers.LocBaseTrainer import LocalBaseTrainer
            # assert issubclass(
            #     func, LocalBaseTrainer
            # ), "All federated algorithm must inherit trainers.LocBaseTrainer.LocalBaseTrainer class"
            cls.mapping["loctrainer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_metric(cls, name):
        """
        Register a metric with the given name.

        Args:
            name (str): The name of the metric.

        Returns:
            wrap: The wrapper function to register the metric

        """
        def wrap(func):
            from metrics.base_metric import BaseMetric

            assert issubclass(
                func, BaseMetric
            ), "All metric must inherit utils.metrics.BaseMetric class"
            cls.mapping["metric_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_eval(cls, name):
        """
        Register an evaluation function with the given name.

        Args:
            name (str): The name of the evaluation function.

        Returns:
            wrap: The wrapper function to register the evaluation

        """
        def wrap(func):
            from evals.BaseEvaluator import BaseEvaluator

            assert issubclass(
                func, BaseEvaluator
            ), "All evaluation must inherit utils.BaseEvaluator.BaseEvaluator class"
            cls.mapping["eval_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def get_loss_class(cls, name):
        """
        Get the loss class registered with the given name.

        Args:
            name (str): The name of the loss function.

        Returns:
            object: The loss class registered with the given name, or None if not found

        """
        return cls.mapping["loss_name_mapping"].get(name, None)

    @classmethod
    def get_data_class(cls, name):
        """
        Get the data manager class registered with the given name.

        Args:
            name (str): The name of the data manager.

        Returns:
            object: The data manager class registered with the given name, or None if not found

        """
        return cls.mapping["data_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        """
        Get the model class registered with the given name.

        Args:
            name (str): The name of the model.

        Returns:
            object: The model class registered with the given name, or None if not found

        """
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_fedtrainer(cls, name):
        """
        Get the federated trainer registered with the given name.

        Args:
            name (str): The name of the federated trainer.

        Returns:
            object: The federated trainer registered with the given name, or None if not found

        """
        return cls.mapping["fedtrainer_name_mapping"].get(name, None)

    @classmethod
    def get_loctrainer(cls, name):
        """
        Get the local trainer registered with the given name.

        Args:
            name (str): The name of the local trainer.

        Returns:
            object: The local trainer registered with the given name, or None if not found

        """
        return cls.mapping["loctrainer_name_mapping"].get(name, None)

    @classmethod
    def get_metric_class(cls, name):
        """
        Get the metric class registered with the given name.

        Args:
            name (str): The name of the metric.

        Returns:
            object: The metric class registered with the given name, or None if not found

        """
        return cls.mapping["metric_name_mapping"].get(name, None)

    @classmethod
    def get_eval_class(cls, name):
        """
        Get the evaluation class registered with the given name.

        Args:
            name (str): The name of the evaluation function.

        Returns:
            object: The evaluation class registered with the given name, or None if not found

        """
        return cls.mapping["eval_name_mapping"].get(name, None)

    @classmethod
    def unregister(cls, name):
        """
        Unregister a component with the given name.

        Args:
            name (str): The name of the component to unregister.

        Returns:
            None

        """
        return cls.mapping["state"].pop(name, None)

    @classmethod
    def get_keys(cls):
        """
        Get a list of all registered keys.

        Returns:
            list: A list of all registered keys.

        """
        keys = list(cls.mapping["state"].keys())
        return keys


registry = Registry()
