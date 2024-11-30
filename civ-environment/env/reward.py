class RewardCalculator:
    """
    A class for computing rewards and state transitions in a PPO framework.
    """

    @staticmethod
    def compute_reward(s, s_next, theta):
        """
        Compute the reward function based on state transitions and policy parameters.

        Args:
            s: Current state.
            s_next: Next state.
            theta: Policy parameters.

        Returns:
            Computed reward.
        """
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, epsilon = theta
        return (
            k1 * RewardCalculator.P_progress(s, s_next) +
            k2 * RewardCalculator.P_completion(s, s_next) +
            k3 * RewardCalculator.C_tiles(s, s_next) +
            k4 * RewardCalculator.C_cities(s, s_next) -
            k5 * RewardCalculator.L_cities(s, s_next) +
            k6 * RewardCalculator.C_units(s, s_next) -
            k7 * RewardCalculator.L_units(s, s_next) +
            k8 * RewardCalculator.Delta_GDP(s, s_next) +
            k9 * RewardCalculator.Delta_energy(s, s_next) +
            k10 * RewardCalculator.C_resources(s, s_next) -
            epsilon * RewardCalculator.E_impact(s, s_next)
        )

    @staticmethod
    def P_progress(s, s_next):
        """
        Calculate the change in ongoing projects between states.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Change in the number of ongoing projects.
        """
        return s_next["ongoing_projects"] - s["ongoing_projects"]

    @staticmethod
    def P_completion(s, s_next):
        """
        Calculate the number of completed projects.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Number of completed projects.
        """
        return s_next["completed_projects"] - s["completed_projects"]

    @staticmethod
    def C_tiles(s, s_next):
        """
        Calculate the number of explored tiles.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Number of explored tiles.
        """
        return s_next["explored_tiles"] - s["explored_tiles"]

    @staticmethod
    def C_cities(s, s_next):
        """
        Calculate the number of captured cities.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Number of captured cities.
        """
        return s_next["captured_cities"] - s["captured_cities"]

    @staticmethod
    def L_cities(s, s_next):
        """
        Calculate the number of lost cities.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Number of lost cities.
        """
        return s["lost_cities"] - s_next["lost_cities"]

    @staticmethod
    def C_units(s, s_next):
        """
        Calculate the number of enemy units eliminated.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Number of enemy units eliminated.
        """
        return s_next["enemy_units_eliminated"] - s["enemy_units_eliminated"]

    @staticmethod
    def L_units(s, s_next):
        """
        Calculate the number of units lost.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Number of units lost.
        """
        return s["units_lost"] - s_next["units_lost"]

    @staticmethod
    def Delta_GDP(s, s_next):
        """
        Calculate the change in GDP.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Change in GDP.
        """
        return s_next["GDP"] - s["GDP"]

    @staticmethod
    def Delta_energy(s, s_next):
        """
        Calculate the change in energy output.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Change in energy output.
        """
        return s_next["energy_output"] - s["energy_output"]

    @staticmethod
    def C_resources(s, s_next):
        """
        Calculate the number of resources gained control over.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Number of resources gained control over.
        """
        return s_next["resources_controlled"] - s["resources_controlled"]

    @staticmethod
    def E_impact(s, s_next):
        """
        Calculate the environmental impact.

        Args:
            s: Current state.
            s_next: Next state.

        Returns:
            Environmental impact.
        """
        return s_next["environmental_impact"] - s["environmental_impact"]
