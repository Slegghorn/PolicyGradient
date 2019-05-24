import tensorflow as tf
import numpy as np
import gym


class Buffer:
    def __init__(self, max_size):
        #all memory initialisation
        self.state_buffer = np.zeros((max_size, 84, 84, 4), dtype = np.uint8)
        self.action_buffer = np.zeros((max_size), dtype = np.int8)
        self.reward_buffer = np.zeros((max_size), dtype = np.float32)
        self.advantage_buffer = np.zeros((max_size), dtype = np.float32)

        #variables necessary to make a slice and the buffer maximum size
        self.step, self.epi_start, self.max_size = 0, 0, max_size

    def store(self, state, action, reward):
        assert self.step < self.max_size

        self.state_buffer[self.step] = state
        self.action_buffer[self.step] = action
        self.reward_buffer[self.step] = reward

        self.step += 1

    def epi_end(self):
        #at the end of the episode cut the rewards corresponding to the episode in the buffer
        epi_slice = slice(self.epi_start, self.step)

        #discount the rewards
        self.advantage_buffer[epi_slice] = self.discount_reward(self.reward_buffer[epi_slice])

        self.epi_start = self.step

    def get_buffer(self):
        #the buffer need to be full before get
        assert self.step == self.max_size

        self.epi_start, self.step = 0, 0

        #normalize the advantage buffer
        mean = np.mean(self.advantage_buffer)
        std = np.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - mean)/std
        return self.state_buffer, self.action_buffer, self.advantage_buffer, self.reward_buffer

    def discount_reward(self, episode_rewards):
        #NB : dtype for the discount was forgotten
        discounted_episode_rewards = np.zeros_like(episode_rewards, dtype = np.float32)
        x = 0.0
        for i in reversed(range(len(episode_rewards))):
            x = x * 0.95 + episode_rewards[i]
            discounted_episode_rewards[i] = x
        return discounted_episode_rewards


class PolicyAgent:
    def __init__(self, action_space, learning_rate, max_size):
        self.action_space = action_space
        self.sess = None
        self.lr = learning_rate
        self.buffer = Buffer(max_size)

        #initialize the agent neural net
        self.model()

        #stateprocessing
        self.input_state = tf.placeholder(shape = [210, 160, 3], dtype = tf.uint8)
        self.output = tf.image.rgb_to_grayscale(self.input_state)
        self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
        self.output = tf.image.resize_images(self.output, [84, 84], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.output = tf.squeeze(self.output)

    def model(self):
        self.st = tf.placeholder(shape = [None, 84, 84, 4], dtype = tf.uint8)
        self.act = tf.placeholder(shape = [None], dtype = tf.uint8)
        self.adv = tf.placeholder(shape = [None], dtype = tf.float32)

        x = tf.cast(self.st, tf.float32) / 255.0
        act = tf.one_hot(self.act, depth=self.action_space)

        conv1 = tf.layers.conv2d(x, 32, 8, 4, activation = tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation = tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activ
        ation = tf.nn.relu)

        flatten = tf.layers.flatten(conv3)
        fc1 = tf.layers.dense(flatten, 512, activation = tf.nn.relu)
        self.predictions = tf.layers.dense(fc1, self.action_space, activation = None)

        self.action_distribution = tf.nn.softmax(self.predictions)


        self.log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.predictions, labels = act)
        self.loss = tf.reduce_mean(self.log_prob * self.adv)

        self.train = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99, momentum=0.0, epsilon=1e-6).minimize(self.loss)

        self.entropy = tf.reduce_mean(self.action_distribution)

    def process(self, state):
        return self.sess.run(self.output, {self.input_state : state})

    def session(self, sess):
        #define the session for tensorflow
        self.sess = sess

    def step(self, state):
        state = np.expand_dims(state, 0)
        predictions = self.sess.run(self.action_distribution, {self.st : state})
        action_step = np.random.choice(np.arange(self.action_space), p = predictions.ravel())
        return action_step

    def train_agent(self):
        #get from the buffer
        state, action, advantage, reward = self.buffer.get_buffer()

        _, entropy,  loss = self.sess.run([self.train, self.entropy, self.loss], {self.st : state, self.act : action, self.adv : advantage})
        reward_mean = np.mean(reward)
        print(reward)

    def store(self, state, action, reward):
        self.buffer.store(state, action, reward)

    def epi_end(self):
        self.buffer.epi_end()

def stack_frame(state_stack, state):
    state_stack = np.append(state_stack[:,:,1:], np.expand_dims(state, 2), axis = 2)
    return state_stack
#MAIN program - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#hyperparameters
valid_actions = [0, 1, 2, 3]
learning_rate = 0.0001
action_space = len(valid_actions)
episode = 10000
buffer_size = 5000

env = gym.make('Breakout-v0')
env = gym.wrappers.Monitor(env, './video', video_callable=lambda episode_id: episode_id%10==0,force=True)

agent = PolicyAgent(action_space, learning_rate, buffer_size)

#initialise session
sess = tf.Session()
#define the session
agent.session(sess)
#initialise variables
sess.run(tf.global_variables_initializer())


#track the size of the buffer
memory_size = 0

for i_episode in range(episode):
    state = env.reset()
    state = agent.process(state)
    state = np.stack([state] * 4, axis = 2)
    done = False
    while not done:
        action = agent.step(state)
        next_state, reward, done, _ = env.step(valid_actions[action])
        agent.store(state, action, reward)
        memory_size += 1

        next_state = agent.process(next_state)
        next_state = stack_frame(state, next_state)
        state = next_state

        if memory_size == buffer_size:
            if not done:
                agent.epi_end()

            agent.train_agent()
            memory_size = 0

        if done:
            agent.epi_end()
