import time
from collections import deque, namedtuple
import sumo_rl
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

MEMORY_SIZE = 100_000
GAMMA = 0.995
ALPHA = 1e-4
NUM_STEPS_FOR_UPDATE = 4
SEED = 0
MINIBATCH_SIZE = 64
TAU = 1e-3
E_DECAY = 0.95
E_MIN = 0.01

intersection = '2x2' # or '3x3'
intersection_folder = f'intersections/{intersection}'
weights_folder = f'{intersection}_agents_weights'
state_size = (21,)
num_actions = 4
num_agents = 4 # 4 for 2x2, 9 for 3x3

num_episodes = 50
max_num_timesteps = 3600
total_point_history = []
num_p_av = 5
epsilon = 1.0
memory_buffers = [deque(maxlen=MEMORY_SIZE) for _ in range(num_agents)]

env = sumo_rl.parallel_env(net_file=f'{intersection_folder}.net.xml',
                           route_file=f'{intersection_folder}.rou.xml',
                           use_gui=True,
                           num_seconds=max_num_timesteps)

def create_q_network(state_size, num_actions):
    model = Sequential([
        Input(shape=state_size),
        Dense(units=64, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=num_actions, activation='linear')
    ])
    return model

q_networks = [create_q_network(state_size, num_actions) for _ in range(num_agents)]
target_q_networks = [create_q_network(state_size, num_actions) for _ in range(num_agents)]
optimizers = [Adam(learning_rate=ALPHA) for _ in range(num_agents)]
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def compute_loss(experiences, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, done_vals = experiences
    
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
         
    loss = MSE(y_targets, q_values) 

    return loss

def agent_learn(q_network, target_q_network, experiences, gamma, optimizer):    
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    update_target_network(q_network, target_q_network)

def update_target_network(q_network, target_q_network):
    for target_weights, q_net_weights in zip(
        target_q_network.weights, q_network.weights
    ):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

def check_update_conditions(t, num_steps_upd, memory_buffer):
    if (t + 1) % num_steps_upd == 0 and len(memory_buffer) > MINIBATCH_SIZE:
        return True
    else:
        return False

def get_experiences(memory_buffer):
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences if e is not None]), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences if e is not None]), dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences if e is not None]).astype(np.uint8), dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)

def get_action(q_values, epsilon=0.0):
    if random.random() > epsilon:
        return np.argmax(q_values.numpy()[0])
    else:
        return random.choice(np.arange(num_actions))


for idx in range(num_agents):
    target_q_networks[idx].set_weights(q_networks[idx].get_weights())

start = time.time()

for i in range(num_episodes):
    start_state = env.reset()
    state = start_state[0]
    total_points = 0

    for t in range(max_num_timesteps):
        actions_dict = {}
        agent_idx = 0
        for agent in env.agents:
            current_state = state[agent]
            state_qn = np.expand_dims(current_state, axis=0)
            q_values = q_networks[agent_idx](state_qn)
            action = get_action(q_values, epsilon)
            actions_dict[agent] = action
            agent_idx +=1

        next_state, rewards, terminations, truncations, infos = env.step(actions_dict)
        done_vals = {key: terminations[key] or truncations[key] for key in env.agents}

        if all(done_vals.values()):
            break

        update_conditions=[]
        agent_idx = 0
        for agent in env.agents:
            current_state = state[agent]
            action = actions_dict[agent]
            reward = rewards[agent]
            next_state_qn = next_state[agent]
            done = done_vals[agent]
            memory_buffers[agent_idx].append(Experience(current_state, action, reward, next_state_qn, done))
            update = check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffers[agent_idx])
            if update:
                experiences = get_experiences(memory_buffers[agent_idx])
                agent_learn(q_networks[agent_idx], target_q_networks[agent_idx], experiences, GAMMA, optimizers[agent_idx])
            agent_idx +=1
        state = next_state.copy()
        total_points += sum(rewards.values())

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    epsilon = max(E_MIN, E_DECAY * epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")


for i in range(num_agents):
    model_path = f'{weights_folder}/{intersection}_agent_{i}_model.h5'
    q_networks[i].save(model_path)

tot_time = time.time() - start
print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")