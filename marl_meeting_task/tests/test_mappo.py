import pytest
import numpy as np
from marl_meeting_task.src.algos.mappo_agent import MAPPOAgent
from marl_meeting_task.src.config import device


def test_mappo_agent_select_action_happy_path():
    n_agents = 2
    obs_dim = 4
    action_dim = 5
    agent = MAPPOAgent(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32, share_actor=True, use_centralized_critic=True)

    # Create a fake observation dict as returned by MeetingGridworldEnv.reset()
    obs = {i: np.array([0, 0, 2, 2], dtype=np.float32) for i in range(n_agents)}
    actions, log_probs, values = agent.select_action(obs, deterministic=False)

    assert isinstance(actions, dict)
    assert len(actions) == n_agents
    assert all(isinstance(a, int) for a in actions.values())
    assert all(np.isfinite(list(log_probs.values())))
    assert all(np.isfinite(list(values.values())))


def test_mappo_agent_empty_obs_raises():
    n_agents = 2
    obs_dim = 4
    action_dim = 5
    agent = MAPPOAgent(n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim)

    with pytest.raises(AssertionError):
        agent.select_action({})

