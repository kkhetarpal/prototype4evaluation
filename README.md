# Installation

Assumes a clean install of Ubuntu 18.04 LTS

- Install deps

```bash
sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv  # system-wide install
```

- Create virtualenv

```bash
virtualenv --system-site-packages -p python3 ./p4e_env
```

- Source environment and install deps

```bash
pip3 install --upgrade tensorflow(-gpu)
pip3 install keras-rl
pip3 install numpy
pip3 install 'gym[box2d]'
pip3 install 'gym[classic_control]'
```

# Framework agnostic example scripts


Example 1
Alice implements 3A (Alice Amazing Algorithm) and tests her hyperparameters on grid worlds of size (1, 10)
```
# file: alice_in_wondeRLand.py
Import tensorflow as tf
# some tensorflow code
class AliceAmazingAlgorithm():
	def __init__(self, hyperparameters):
		"""
		The 3A algorithm, the best RL algorithm ever (for gridworlds)
		:param hyperparameters: the hyperparameters for the algorithm
		"""
		self.state = tf.placeholder()
		Linear_layer = tf.linear()
		self.policy_result = Linear_layer(state)

	def learn(self, environment, iterations):
		Action = sess.run(results, feed_dict={state}
		environment.step(action)
		# policy learning here

	def Act(self, state):
		Return sess.run(results, feed_dict={state}
```

Example 2
Bob implements 2BP (BobBuildsPolicies) and tests his hyperparameters on grid worlds of size (10, 20)
```
Import torch 
# some pytorch code
class BobBuildsPolicies():
	def __init__(self, hyperparameters):
		"""
		The 2BO algorithm, the best RL algorithm ever (for gridworlds)
		:param hyperparameters: the hyperparameters for the algorithm
		"""
		Self.policy = torch.nn.Linear(bla, bla)

	def learn(self, environment, iterations):
		Loss = do_rollout(environment, self.policy)
		optimizer.zero_grad()
		los.backward()
		optimizer.step()
		
	def Act(self, state):
		Return self.policy(state)
```

## Benchmarking
How would one benchmark these two algorithms on a set of gridworlds?
Both Alice and Bob claim that their algorithms are the best grid world solvers for a specific task in a specific grid world. Without having access to each other's evaluation and source code scripts, how would they compare their algorithms one-to-one? In addition to this, it is likely that each use their own crazy ways to plot data. What if there was an agreed upon benchmarking script that they could use made available by the makers of a grid world benchmark? 

```
Grid world benchmarking! By GridWorld Technologies.

1. User edits the configuration file
# file name: config.json
{
Algo_path: "path/to/your/algorithm/file” # user algorithm file path
Algorithm: "main(train_ddpg, ‘Algorithm’) # bind user algo with benchmark algo
Hyperparameters = load_txt(“path/to/hyperparameters”)
}

2. User launches the benchmarking script
# minimal example of the benchmark script (user cannot edit this)
For env_seed in LIST_OF_AGREED_UPON_SEEDS_FOR_ENV: # technically should be a multiprocessing loop but ok
	Env = env(env_seed, env_characteristics) # make a new environment
Alg = Algorithm(Hyperparameters)	
Alg.learn(Env, 10000)
Reward_curve, other_metrics = do_test_rollout(Alg.act, Env)
np.save([reward_curve, other_metrics], ‘evaluation_on_env{n}.npy’.format(env_seed))
```

Now the authors only have to make available the evaluation_on_env.npy files to allow one to one comparison. Therefore, Bob, with access to this script and the evaluated curves can recreate plots to compare with Alice!

