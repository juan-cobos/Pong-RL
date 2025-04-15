import numpy as np
from pong import *
import pickle
import matplotlib.pyplot as plt

# hyperparams
H = 200
batch_size = 10 # every how many episodes to do a param update?
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
learning_rate = 0.01

# model init
D = 64 * 64 # Pong dims
model = {}
model["W1"] = np.random.uniform(-1, 1, size = (H, D)) / np.sqrt(D)
model["W2"] = np.random.uniform(-1, 1, size = H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(reward_array, T=50):
    """ take 1D float array of rewards and compute truncated (last T) discounted reward """
    # TODO: This can be written in one line
    # TODO: consider not truncating or other methods
    r = reward_array[-1][:T] # Reverse and truncate reward array
    t = np.arange(T) # Get time steps array
    discount_r = r * gamma ** t # Compute discounted reward based on time step
    return discount_r[-1]

def forward(x):
    h = x.ravel() @ model["W1"].T
    h[h < 0] = 0 # relu
    logp = h @ model["W2"].T # self connections
    out = sigmoid(logp)
    return out, h

def backward(eph, epdlogp, trunc_length=50):
    dW2 = np.dot(eph.T, epdlogp).flatten()
    dh = np.outer(epdlogp, model["W2"]) # Propagate gradients to previous layers
    dh[eph < 0] = 0 # relu backprop
    dW1 = np.dot(dh.T, epx)
    return {"W1": dW1, "W2": dW2}

prev_x = None
obs = np.zeros((64, 64)) # First obs is empty
xs,hs,dlogps,drs = [],[],[],[]

wins = 0
reward_sum = 0
episode_count = 0

n_episodes = 1000
env = Pong(render_mode=None)
while n_episodes > episode_count:

    # Using frame difference as input img
    x = (obs - prev_x if prev_x is not None else obs).flatten()
    prev_x = obs

    # forward the policy network and sample an action from the returned probability
    aprob, h = forward(x)
    action = 1 if aprob < 0.5 else 2 # Max prob policy

    # record various intermediates (needed later for backprop)
    xs.append(x)
    hs.append(h)
    y = 1 if action == 2 else 0 # "Fake label"
    dlogps.append(y - aprob)

    obs, reward, done = env.step(action)

    drs.append(reward)  # record reward (has to be done after we call step() to get reward (t+1) for previous action)
    reward_sum += reward # Track number of rewards after each episode

    if done:
        if reward == 1: wins += 1
        episode_count += 1
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = backward(eph, epdlogp)

        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_count % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
            print(f"Current winrate {wins/episode_count * 100}")
    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print(f"ep {episode_count}: game finished, reward: {reward}{'' if reward == -1 else ' !!!!!!!!'} ")

print(f"Total winrate: {wins/n_episodes * 100} %")
env.reset()

# Save weights after training
pickle.dump(model, open("pong-model.p", "wb"))
print("Weights written correctly. All done!")


