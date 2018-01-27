import matplotlib
matplotlib.use('Agg')

import random
import gym
import numpy as np 

from collections import deque

import keras
from keras.models import Sequential
from keras.layers import TimeDistributed, Flatten, Dense, Conv2D, LSTM
from keras import backend as K
from keras.optimizers import Adam

from skimage import color
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt

frame_size = 80
num_frames = 4


class Agent:

	def __init__(self, num_actions, batch_size):
		self.num_actions = num_actions
		self.batch_size = batch_size
		self.memory = deque(maxlen=200000)		
		self.gamma = 0.99
		
		self.epsilon = 1.0
		self.epsilon_min = 0.10
		self.exploration_steps = 350000.
		self.epsilon_decay_step = (self.epsilon - self.epsilon_min) / self.exploration_steps
		self.observe = 50000
		
		self.model = self.create_model()
		self.target_model = self.create_model()		

		self.update_target = 5000
		self.save_weights_interval = 20000


	def replay(self):
		if len(self.memory) < self.observe:
			return

		minibatch = random.sample(self.memory, self.batch_size)

		state = np.zeros((self.batch_size, num_frames, frame_size, frame_size, 1))
		next_state = np.zeros((self.batch_size, num_frames, frame_size, frame_size, 1))
		action, reward, done = [], [], []

		for i in range(self.batch_size):
			state[i] = np.float32(minibatch[i][0] / 255.0)
			next_state[i] = np.float32(minibatch[i][3] / 255.0)
			action.append(minibatch[i][1])
			reward.append(minibatch[i][2])
			done.append(minibatch[i][4])

		target = self.model.predict(state, batch_size=self.batch_size)
		target_next = self.target_model.predict(next_state, batch_size=self.batch_size)
		
		for i in range(self.batch_size):
			index = np.argmax(target_next[i])
			target[i][action[i]] = reward[i] + self.gamma * (target_next[i][index])
				
		self.model.train_on_batch(state, target)

		if self.epsilon > self.epsilon_min:
			self.epsilon -= self.epsilon_decay_step	


	def create_model(self):
		input_shape = (num_frames, frame_size, frame_size, 1)

		model = Sequential()
		model.add(TimeDistributed(Conv2D(32, (8, 8), strides=(4,4), activation='relu'), input_shape=(input_shape)))
		model.add(TimeDistributed(Conv2D(64, (4, 4), strides=(2,2), activation='relu')))
		model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1,1), activation='relu')))
		model.add(TimeDistributed(Flatten()))

		# Use last trace for training
		model.add(LSTM(512,  activation='tanh'))
		model.add(Dense(units=self.num_actions, activation='linear'))

		model.compile(loss='mse', optimizer=Adam(lr=0.0001))

		return model


	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())


	def load_model(self, name):
		self.model.load_weights(name)


	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))


	def preprocess(self, img): 
		return np.uint8(resize(rgb2gray(img), (frame_size, frame_size), mode='constant') * 255)


	def choose_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.num_actions)

		state = np.expand_dims(state, axis=0)
		state = np.float32(state / 255.0)
		return np.argmax(self.model.predict(state)[0])




if __name__ == "__main__":

	env = gym.make('BreakoutDeterministic-v4')

	batch_size = 32
	done = False

	num_actions = env.action_space.n
	agent = Agent(num_actions, batch_size)

	global_step = 0
	num_episodes = 0
	max_steps = 3000000

	all_scores = []
	score_at_step = []

	while global_step < max_steps:
		score = 0
		state = env.reset()
		state = agent.preprocess(state)

		history = np.stack((state, state, state, state), axis=2)
		history = np.reshape([history], (num_frames, frame_size, frame_size, 1))

		while True:
			#env.render()
			global_step += 1

			action = agent.choose_action(history)

			next_state, reward, done, _ = env.step(action)

			score += reward	
			reward = np.clip(reward, -1., 1.)


			next_state = agent.preprocess(next_state)
			next_state = np.reshape([next_state], (1, frame_size, frame_size, 1))
				
			next_history = np.append(history[1:, :, :, :], next_state, axis=0)

			agent.remember(history, action, reward, next_history, done)

			if global_step % 4 == 0:
				agent.replay()		

			history = next_history

			if global_step % agent.save_weights_interval == 0:
				agent.model.save_weights("my_weights_agent_rnn.h5", overwrite=True)

			if global_step % agent.update_target == 0:
				agent.update_target_model()

			if done:
				print("episode: {}, score: {}, e: {:.4}, steps: {}".format(num_episodes, score, agent.epsilon, global_step))
				break

		all_scores.append(score)
		score_at_step.append(global_step)
		num_episodes+=1

		if num_episodes % 100 == 0:
			plt.plot(score_at_step[:], all_scores[:], linewidth=0.5)
			plt.ylabel('Score')
			plt.xlabel('Steps')
			plt.savefig('./score.jpg')

