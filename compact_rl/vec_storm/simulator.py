from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import chex

from .sparse_array import SparseArray


@chex.dataclass
class States:
    vertices: chex.Array
    steps: chex.Array

    def slice(self, start: int, end: int) -> "States":
        """Return a sliced version of the States object."""
        return States(
            vertices=self.vertices[start:end],
            steps=self.steps[start:end]
        )


@chex.dataclass
class ResetInfo:
    states: States
    observations: chex.Array
    allowed_actions: chex.Array
    metalabels: chex.Array
    integer_observations: chex.Array = None


@chex.dataclass
class StepInfo:
    states: States
    observations: chex.Array
    rewards: chex.Array
    done: chex.Array
    truncated: chex.Array
    allowed_actions: chex.Array
    metalabels: chex.Array
    integer_observations: chex.Array = None

    @staticmethod
    def combine(step_infos: List["StepInfo"]) -> "StepInfo":
        """Combine multiple StepInfo objects into one."""
        combined_states = States(
            vertices=jnp.concatenate([si.states.vertices for si in step_infos]),
            steps=jnp.concatenate([si.states.steps for si in step_infos])
        )
        combined_observations = jnp.concatenate([si.observations for si in step_infos])
        combined_rewards = jnp.concatenate([si.rewards for si in step_infos])
        combined_done = jnp.concatenate([si.done for si in step_infos])
        combined_truncated = jnp.concatenate([si.truncated for si in step_infos])
        combined_allowed_actions = jnp.concatenate([si.allowed_actions for si in step_infos])
        combined_metalabels = jnp.concatenate([si.metalabels for si in step_infos])
        integer_observations = jnp.concatenate([si.integer_observations for si in step_infos]) if step_infos[0].integer_observations is not None else None

        return StepInfo(
            states=combined_states,
            observations=combined_observations,
            rewards=combined_rewards,
            done=combined_done,
            truncated=combined_truncated,
            allowed_actions=combined_allowed_actions,
            metalabels=combined_metalabels,
            integer_observations=integer_observations
        )


@chex.dataclass
class Simulator:
    id: int

    initial_state: int
    max_outcomes: int
    max_steps: int
    random_init: bool

    transitions: SparseArray
    rewards: SparseArray
    observations: chex.Array
    sinks: chex.Array
    allowed_actions: chex.Array
    metalabels: chex.Array
    labels: chex.Array

    action_labels: List[str]
    observation_labels: List[str]

    state_values: chex.Array
    state_labels: chex.Array
    state_observation_ids: chex.Array

    observation_by_ids: chex.Array

    FREE_ID = 0

    def __hash__(self):
        return self.id

    @staticmethod
    def get_free_id():
        Simulator.FREE_ID += 1
        return Simulator.FREE_ID

    def sample_next_vertex(self: "Simulator", vertex, action, rng_key):
        l, r = self.transitions.get_row_range(vertex, action)
        entry_indices = jnp.arange(0, self.max_outcomes, 1) + l
        mask = entry_indices < r
        probs = jnp.where(mask, self.transitions.data[entry_indices], 0)
        idx = jax.random.choice(key=rng_key, a=entry_indices, p=probs)
        return self.transitions.indices[idx], idx

    def get_observation(self: "Simulator", vertex):
        return self.observations[vertex]

    def get_reward(self: "Simulator", entry_idx):
        return self.rewards.data[entry_idx]

    def is_done(self: "Simulator", vertex):
        return self.sinks[vertex]

    def get_init_states(self: "Simulator", states, rng_key):
        if self.random_init == False:
            vertices = states.vertices.at[:].set(self.initial_state)
        else:
            vertices = jax.random.randint(rng_key, states.vertices.shape, 0, len(self.sinks))
            vertices = jnp.where(self.sinks[vertices], self.initial_state, vertices)
        return States(
            vertices = vertices,
            steps = jnp.zeros_like(states.steps),
        )

    @partial(jax.jit, static_argnums=0)
    def reset(self: "Simulator", states: States, rng_key) -> ResetInfo:
        new_states = self.get_init_states(states, rng_key)
        observations = jax.vmap(lambda s: self.get_observation(s))(new_states.vertices)
        integer_observations = self.state_observation_ids[new_states.vertices].reshape(-1, 1) 
        return ResetInfo(
            states = new_states,
            observations = observations,
            allowed_actions = self.allowed_actions[new_states.vertices],
            metalabels = self.metalabels[new_states.vertices],
            integer_observations = integer_observations
        )

    @partial(jax.jit, static_argnums=0)
    def step(self: "Simulator", states, actions, rng_key) -> StepInfo:
        key1, key2 = jax.random.split(rng_key)
        prev_trunc = states.steps >= self.max_steps # added
        prev_done = self.sinks[states.vertices]
        new_vertices, new_vertex_idxs = jax.vmap(lambda s, a, k: self.sample_next_vertex(s, a, k))(states.vertices,
                                                                                                   actions,
                                                                                                   jax.random.split(
                                                                                                       key1,
                                                                                                       len(actions)))
        # Compute rewards of the transitions s -> a -> s'
        rewards = jax.vmap(lambda new_s: self.get_reward(new_s))(new_vertex_idxs)
        steps = states.steps + 1
        # Reset done state s' to initial state i
        if not self.random_init:
            key2 = None
        vertices_after_reset = jnp.where(prev_done | prev_trunc, self.get_init_states(states, rng_key=key2).vertices, new_vertices)
        # done = jnp.where(prev_done, False, done)

        done = self.sinks[vertices_after_reset]
        steps_after_reset = jnp.where(prev_done | prev_trunc, 0, steps)
        trunc = steps_after_reset >= self.max_steps
        rewards = jnp.where(prev_done | prev_trunc, 0, rewards)

        # Compute observation of states after reset (s' or i)
        observations = jax.vmap(lambda s: self.get_observation(s))(vertices_after_reset)
        metalabels = self.metalabels[vertices_after_reset]
        allowed_actions = self.allowed_actions[vertices_after_reset]
        integer_observations = self.state_observation_ids[vertices_after_reset].reshape(-1, 1)
        return StepInfo(
            states=States(vertices=vertices_after_reset, steps=steps_after_reset),
            observations=observations,
            rewards=rewards,
            done=done | trunc | metalabels.reshape(-1),  # If the state is a sink or truncated or metalabels, we consider it done
            truncated=trunc,
            allowed_actions=allowed_actions,
            metalabels=metalabels,
            integer_observations=integer_observations
        )
    
    def no_step(self : "Simulator", states):
        """Simulate a step without any action taken."""
        # Compute observation of states after reset (s' or i)
        observations = jax.vmap(lambda s: self.get_observation(s))(states.vertices)
        metalabels = self.metalabels[states.vertices]
        trunc = states.steps >= self.max_steps
        done = self.sinks[states.vertices]
        allowed_actions = self.allowed_actions[states.vertices]
        allowed_actions = jnp.where(jnp.tile(jnp.reshape(done, (-1, 1)), (1, allowed_actions.shape[1])), 
                                    jnp.ones_like(allowed_actions), allowed_actions)
        rewards = jax.vmap(lambda s: self.get_reward(s))(states.steps)

        return StepInfo(
            states = states,
            observations = observations,
            rewards = rewards,
            done = done,
            truncated = trunc,
            allowed_actions = allowed_actions,
            metalabels = metalabels,
        )

    
    def set_max_steps(self, max_steps):
        self.max_steps = max_steps
