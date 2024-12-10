import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, max_units_per_agent, max_cities, max_projects):
        super(ActorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc_action_type = nn.Linear(hidden_size, 7)
        self.fc_unit_id = nn.Linear(hidden_size, max_units_per_agent)
        self.fc_direction = nn.Linear(hidden_size, 4)
        self.fc_city_id = nn.Linear(hidden_size, max_cities)
        self.fc_project_id = nn.Linear(hidden_size, max_projects)

    @staticmethod
    def process_observation(obs):
        # obs is a dict or array. Flatten all components.
        if isinstance(obs, dict):
            processed_obs = []
            for key in obs:
                value = obs[key]
                if isinstance(value, torch.Tensor):
                    # Already tensor
                    tensor_value = value.flatten()
                else:
                    # Convert numpy arrays to torch tensors
                    tensor_value = torch.tensor(value, dtype=torch.float32).flatten()
                processed_obs.append(tensor_value)
            obs_tensor = torch.cat(processed_obs)
        elif isinstance(obs, (np.ndarray, torch.Tensor)):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).flatten()
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")
        # obs_tensor shape: (input_size,)
        return obs_tensor

    def forward(self, observations, hidden_states):
        """
        Forward pass:
        observations: (batch_size, seq_len, input_size)
        hidden_states: (1, batch_size, hidden_size)

        Returns:
            action_probs: dict of (batch_size, seq_len, num_actions)
            hidden_states: (1, batch_size, hidden_size)
        """
        # Pass through GRU
        output, hidden_states = self.rnn(observations, hidden_states)
        # output: (batch_size, seq_len, hidden_size)

        # Compute probabilities for each action component at each timestep
        action_type_probs = F.softmax(self.fc_action_type(output), dim=-1)
        unit_id_probs = F.softmax(self.fc_unit_id(output), dim=-1)
        direction_probs = F.softmax(self.fc_direction(output), dim=-1)
        city_id_probs = F.softmax(self.fc_city_id(output), dim=-1)
        project_id_probs = F.softmax(self.fc_project_id(output), dim=-1)

        # Return probabilities and hidden states
        # Each probability distribution is (batch_size, seq_len, ...)
        action_probs = {
            'action_type': action_type_probs,
            'unit_id': unit_id_probs,
            'direction': direction_probs,
            'city_id': city_id_probs,
            'project_id': project_id_probs,
        }
        return action_probs, hidden_states


class CriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CriticRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, observations, hidden_states):
        """
        Forward pass:
        observations: (batch_size, seq_len, input_size)
        hidden_states: (1, batch_size, hidden_size)

        Returns:
            values: (batch_size, seq_len, 1)
            hidden_states: (1, batch_size, hidden_size)
        """
        output, hidden_states = self.rnn(observations, hidden_states)
        values = self.fc(output)  # (batch_size, seq_len, 1)
        return values, hidden_states
