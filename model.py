from abc import ABC
from abc import abstractmethod
from enum import auto
from enum import Enum
from itertools import combinations
from itertools import combinations_with_replacement
from random import gauss
from typing import Iterator

import numpy as np


class Model(ABC):
    @staticmethod
    @abstractmethod
    def random_collaboration(num_nodes: int, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def random_curiosity(num_nodes: int, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def objective_function(num_nodes: int, curiosity_matrix, collaboration_matrix):
        pass

    @staticmethod
    @abstractmethod
    def optimal_choice(num_nodes: int, curiosity_matrix, collaboration_matrix):
        pass

    @staticmethod
    def random_value(mu: float, std: float) -> float:
        return min(1.0, max(0.0, gauss(mu=mu, sigma=std)))

    @staticmethod
    def generate_path_combinations(
        paths: list[int], num_paths: int, with_replacement: bool = True
    ) -> Iterator[tuple[int, ...]]:
        if len(paths) == 0:
            return
        if num_paths == 0:
            return

        if with_replacement:
            for _paths in combinations_with_replacement(paths, num_paths):
                yield _paths
        else:
            for _paths in combinations(paths, num_paths):
                yield _paths


class ProbabilisticModel(Model):
    class ObjectFunction(Enum):
        NO_NODE_BREAK_SECRET = auto()
        SOME_NODE_BREAK_SECRET = auto()

    @staticmethod
    def parse_gauss_params(**kwargs) -> tuple[float, float]:
        mu = kwargs["mu"] if "mu" in kwargs else 0.5
        std = kwargs["std"] if "std" in kwargs else 0.25
        return mu, std

    @staticmethod
    def random_curiosity(num_nodes: int, **kwargs) -> np.ndarray:
        """
        Generates a random curiosity matrix for a probabilistic model.

        The curiosity matrix is represented as a 1D numpy array of size num_nodes.
        The first and last elements of the array represent the source and destination nodes respectively and
        are set to 1.0. The remaining elements are set to a random value between 0 and 1 using a gaussian distribution
        with mean `mu` and standard deviation `std`.

        Parameters
        ----------
        num_nodes: int
            The number of nodes in the probabilistic model
        mu: float
            The mean of the gaussian distribution
        std: float
            The standard deviation of the gaussian distribution

        Returns
        -------
        curiosity_matrix: np.ndarray
            A 1D numpy array of size num_nodes representing the curiosity matrix
        """
        mu, std = ProbabilisticModel.parse_gauss_params(**kwargs)
        curiosity_matrix = np.ones(num_nodes)
        for i in range(1, num_nodes - 1):
            curiosity_matrix[i] = ProbabilisticModel.random_value(mu, std)
        return curiosity_matrix

    @staticmethod
    def random_collaboration(num_nodes: int, **kwargs) -> np.ndarray:
        """
        Generates a random collaboration matrix for a probabilistic model.

        The collaboration matrix is represented as a 2D numpy array of size (num_nodes, num_nodes).
        The diagonal elements of the array represent a node's willingness to collaborate with itself and
        are set to 1.0. The remaining elements are set to a random value between 0 and 1 using a gaussian distribution
        with mean `mu` and standard deviation `std`.

        Parameters
        ----------
        num_nodes: int
            The number of nodes in the probabilistic model
        mu: float
            The mean of the gaussian distribution
        std: float
            The standard deviation of the gaussian distribution

        Returns
        -------
        collaboration_matrix: np.ndarray
            A 2D numpy array of size (num_nodes, num_nodes) representing the collaboration matrix
        """
        mu, std = ProbabilisticModel.parse_gauss_params(**kwargs)
        collaboration_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            collaboration_matrix[i, i] = 1.0
        for i in range(1, num_nodes - 1):
            for j in range(1, i):
                collaboration_matrix[i, j] = collaboration_matrix[j, i] = ProbabilisticModel.random_value(mu, std)
        return collaboration_matrix

    @staticmethod
    def gathering_probability(
        num_nodes: int, paths: list[tuple[int, ...]], collaboration_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the probability of a node gathering all the shares from given paths.

        The probability of a node gathering a share from a path is calculated as the product of the probabilities
        of the nodes in the path collaborating with each other. The probability of a node gathering all the shares
        from all the paths is then calculated as the product of the probabilities of the node gathering each of the
        shares from the paths.

        Parameters
        ----------
        num_nodes: int
            The number of nodes in the probabilistic model
        paths: list[tuple[int, ...]]
            The paths of the shares
        collaboration_matrix: np.ndarray
            The collaboration matrix of the nodes

        Returns
        -------
        gathering_probability: np.ndarray
            A 1D numpy array of size num_nodes representing the probability of each node gathering all the shares
        """
        return np.array(
            [
                np.prod([1 - np.prod([1 - collaboration_matrix[i, j] for j in path]) for path in paths])
                for i in range(num_nodes)
            ]
        )

    @staticmethod
    def decoding_probability(num_nodes: int, curiosity_matrix: np.ndarray, num_shares: int) -> np.ndarray:
        """
        Calculates the probability of a node decoding the secret given the number of shares.

        Parameters
        ----------
        num_nodes: int
            The number of nodes in the probabilistic model
        curiosity_matrix: np.ndarray
            The curiosity matrix of the nodes
        num_shares: int
            The number of shares

        Returns
        -------
        decoding_probability: np.ndarray
            A 1D numpy array of size num_nodes representing the probability of each node decoding the secret
        """
        return np.array([curiosity_matrix[i] ** num_shares for i in range(num_nodes)])

    @staticmethod
    def breaking_probability(gathering_probability: np.ndarray, decoding_probability: np.ndarray) -> np.ndarray:
        """
        Calculates the probability of a node breaking the secret given the gathering and decoding probabilities.

        Parameters
        ----------
        gathering_probability: np.ndarray
            A 1D numpy array of size num_nodes representing the probability of each node gathering all the shares
        decoding_probability: np.ndarray
            A 1D numpy array of size num_nodes representing the probability of each node decoding the secret given the number of shares

        Returns
        -------
        breaking_probability: np.ndarray
            A 1D numpy array of size num_nodes representing the probability of each node breaking the secret
        """
        return gathering_probability * decoding_probability

    @staticmethod
    def no_node_break_secret(breaking_probability: np.ndarray) -> np.int64:
        """
        Calculates the probability that no node breaks the secret given the breaking probabilities of the nodes.

        Parameters
        ----------
        breaking_probability: np.ndarray
            A 1D numpy array of size num_nodes representing the probability of each node breaking the secret

        Returns
        -------
        no_node_break_secret: np.int64
            The probability that no node breaks the secret
        """
        return np.prod(1 - breaking_probability[1:-1])

    @staticmethod
    def some_node_break_secret(breaking_probability: np.ndarray) -> np.int64:
        """
        Calculates the probability that at least one node breaks the secret given the breaking probabilities of the nodes.

        Parameters
        ----------
        breaking_probability: np.ndarray
            A 1D numpy array of size num_nodes representing the probability of each node breaking the secret

        Returns
        -------
        some_node_break_secret: np.int64
            The probability that at least one node breaks the secret
        """
        return 1 - max(breaking_probability[1:-1])

    @staticmethod
    def objective_value(
        breaking_probability: np.ndarray, obj_fn: ObjectFunction = ObjectFunction.NO_NODE_BREAK_SECRET
    ) -> np.int64:
        if obj_fn == ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET:
            return ProbabilisticModel.no_node_break_secret(breaking_probability)
        elif obj_fn == ProbabilisticModel.ObjectFunction.SOME_NODE_BREAK_SECRET:
            return ProbabilisticModel.some_node_break_secret(breaking_probability)

    @staticmethod
    def objective_function(num_nodes: int, curiosity_matrix: np.ndarray, collaboration_matrix: np.ndarray):
        def fn(
            paths: list[tuple[int, ...]],
            obj_fn: ProbabilisticModel.ObjectFunction = ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET,
            return_probabilities: bool = False,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int64] | np.int64:
            """
            Calculates the objective value of a set of paths in a probabilistic routing model.

            Parameters
            ----------
            paths: list[tuple[int, ...]]
                The paths of the shares
            obj_fn: ProbabilisticModel.ObjectFunction, optional
                The objective function to use, by default ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET
            return_probabilities: bool, optional
                Whether to return the probabilities of each node gathering all the shares, decoding the secret, and breaking the secret, by default False

            Returns
            -------
            If return_probabilities is False, the objective value as an integer. Otherwise, a tuple of four numpy arrays:
                - gathering_probability: the probability of each node gathering all the shares
                - decoding_probability: the probability of each node decoding the secret given the number of shares
                - breaking_probability: the probability of each node breaking the secret
                - objective_value: the objective value of the paths as an integer
            """
            num_shares = len(paths)
            gathering_probability = ProbabilisticModel.gathering_probability(num_nodes, paths, collaboration_matrix)
            decoding_probability = ProbabilisticModel.decoding_probability(num_nodes, curiosity_matrix, num_shares)
            breaking_probability = ProbabilisticModel.breaking_probability(gathering_probability, decoding_probability)

            objective_value = ProbabilisticModel.objective_value(breaking_probability, obj_fn)
            if return_probabilities:
                return gathering_probability, decoding_probability, breaking_probability, objective_value
            return objective_value

        return fn

    @staticmethod
    def optimal_choice(num_nodes: int, curiosity_matrix: np.ndarray, collaboration_matrix: np.ndarray):
        objective_fn = ProbabilisticModel.objective_function(num_nodes, curiosity_matrix, collaboration_matrix)

        def fn(
            paths: list[tuple[int, ...]],
            num_paths: int,
            obj_fn: ProbabilisticModel.ObjectFunction = ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET,
        ) -> tuple[int, ...]:
            """
            Finds the optimal combination of paths from given set of paths that results in
            maximum objective value.

            Parameters
            ----------
            paths: list[tuple[int, ...]]
                The set of paths to choose from
            num_paths: int
                The number of paths to choose
            obj_fn: ProbabilisticModel.ObjectFunction, optional
                The objective function to use, by default ProbabilisticModel.ObjectFunction.NO_NODE_BREAK_SECRET

            Returns
            -------
            tuple[int, ...]
                The optimal combination of paths
            """
            max_value = -1
            optimal_paths = None
            for _paths in ProbabilisticModel.generate_path_combinations(paths, num_paths):
                objective_value = objective_fn(_paths, obj_fn)
                assert objective_value >= 0
                if objective_value > max_value:
                    max_value = objective_value
                    optimal_paths = _paths
            return optimal_paths

        return fn
