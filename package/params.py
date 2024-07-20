# https://gymnasium.farama.org/environments/box2d/lunar_lander/
import gymnasium as gy

env = gy.make("LunarLander-v2")
state_shape = env.observation_space.shape  # return vector
state_size = env.observation_space.shape[0]
number_action = env.action_space.n

print(state_shape, state_size, number_action)

learning_rate = 5e-4  # 5 X 10 raise to -4
minbatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_param = 1e-3

print(
    learning_rate,
    minbatch_size,
    discount_factor,
    replay_buffer_size,
    interpolation_param,
)
