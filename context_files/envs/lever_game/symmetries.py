import numpy as np
from lever_game import TwoPlayerLeverGame

class RandomPermutation:
    """
    Implements a symmetry transformation for an environment, to be used with the Other-Play (OP) algorithm.

    This class represents a single, randomly sampled symmetry transformation (a function $\phi \in \Phi$).
    It provides methods to apply this transformation (and its inverse) to actions and observations,
    effectively creating a permuted view of the environment for an agent.
    """

    def __init__(self, seed=None):
        self.env = TwoPlayerLeverGame()
        self.rng = np.random.default_rng(seed)

    def sample_permutation(self):
        """
        Samples a single symmetry transformation from the set of all possible symmetries ($\Phi$) for the environment.
        """
        rewards = self.env.rewards
        unique_rewards = np.unique(rewards)
        self.action_permutation = np.arange(self.env.num_levers)
        for reward in unique_rewards:
            indices = np.where(rewards == reward)[0]
            if len(indices) > 1:
                self.action_permutation[indices] = self.rng.permutation(indices)
        return self.action_permutation

    def permute_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Applies the sampled symmetry transformation to an observation.

        This method takes an observation from the base environment and maps it to the agent's
        permuted view of the environment.

        Args:
            observation: The observation from the original, unpermuted environment.

        Returns:
            The transformed observation for the agent's perspective.
        """
        return observation

    def permute_action(self, action: int) -> int:
        """
        Applies the sampled symmetry transformation to an action.

        This method takes an action chosen by the agent (in its permuted action space) and maps
        it to the corresponding action in the base environment's action space.

        Args:
            action: The action chosen by the agent in its permuted frame of reference.

        Returns:
            The corresponding action in the base environment.
        """
        return self.action_permutation[action]

    def invert_observation_permutation(self, observation: np.ndarray) -> np.ndarray:
        """
        Applies the inverse of the symmetry transformation to an observation.

        This maps an observation from an agent's permuted view back to the
        base environment's observation space.

        Args:
            observation: The observation from the agent's permuted perspective.

        Returns:
            The corresponding observation in the original, unpermuted environment.
        """
        return observation

    def invert_action_permutation(self, action: int) -> int:
        """
        Applies the inverse of the symmetry transformation to an action.

        This maps an action from the base environment to the agent's permuted
        action space. This is useful for interpreting an external action (e.g., from a
        human partner) within the agent's transformed perspective.

        Args:
            action: An action in the base environment's action space.

        Returns:
            The corresponding action in the agent's permuted action space.
        """
        return np.argsort(self.action_permutation)[action]

if __name__ == "__main__":
    random_permutation = RandomPermutation()
    print(random_permutation.sample_permutation())
    print(random_permutation.invert_observation_permutation(np.array([0.0])))
    print(x:=random_permutation.permute_action(4))
    print(random_permutation.invert_action_permutation(x))