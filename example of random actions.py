import sumo_rl

intersection = '2x2' # or '3x3'
intersection_folder = f'intersections/{intersection}'

env = sumo_rl.parallel_env(net_file=f'{intersection_folder}.net.xml',
                           route_file=f'{intersection_folder}.rou.xml',
                           use_gui=True,
                           num_seconds=max_num_timesteps)

observations = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
