import sumo_rl
import time
import numpy as np
import tensorflow as tf
import random

max_num_timesteps = 7200
intersection = '2x2' # or '3x3'
num_agents = 4 #4 for 2x2, 9 for 3x3
num_actions = 4
intersection_folder = f'intersections/{intersection}'
weights_folder = f'{intersection}_agents_weights'

env = sumo_rl.parallel_env(net_file=f'{intersection_folder}.net.xml',
                           route_file=f'{intersection_folder}.rou.xml',
                           use_gui=True,
                           num_seconds=max_num_timesteps)


def get_action(q_values, epsilon=0.0):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(num_actions))

q_networks = []

for i in range(num_agents):
    model_path = f'{weights_folder}/{intersection}_agent_{i}_model.h5'
    q_networks.append(tf.keras.models.load_model(model_path))

observations = env.reset()
state = observations[0]

for t in range(max_num_timesteps):
    actions_dict = {}
    agent_idx = 0
    for agent in env.agents:
        current_state = state[agent]
        state_qn = np.expand_dims(current_state, axis=0)
        q_values = q_networks[agent_idx](state_qn)
        action = get_action(q_values, epsilon=0)
        actions_dict[agent] = action
        agent_idx += 1

    next_state, rewards, terminations, truncations, infos = env.step(actions_dict)
    done_vals = {key: terminations[key] or truncations[key] for key in env.agents}

    if all(done_vals.values()):
        break
    state = next_state.copy()

env.close()