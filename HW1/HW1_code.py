import numpy as np
import random

class Environment:
    def __init__(self, a=0.1, b=-0.04, p=0.4, miu=0.02, sigma=0.1, r=0.01, w=1.0, util_a=2, num_actions=11, model='bbbbbnnnnn'):
        self.a = a # this a is one possible return Yt for t<5
        self.b = b
        self.p = p
        self.miu = miu
        self.sigma = sigma
        self.r = r # risk free rate
        self.util_a = util_a # this a is in utility function (1 - a*exp(-a*w)) / a
        self.init_wealth = w
        self.state = (0, self.init_wealth) # (current_time, current wealth)
        self.num_actions = num_actions # How many discretised actions, for example 11 means proportion [0,0.1,0.2,0.3,.....,0.9,1] of wealth allocated in risky asset
        self.model = model # 'b' means binomial and 'n' means normal distribution. 'bbbbbnnnnn' means first 5 time steps follow binomial distribution and the following 5 follow normal distribution

    def reset(self):
        self.state = (0, self.init_wealth)
        return self.state

    def step(self, action):
        current_time, wealth = self.state
        ratio = action / (self.num_actions - 1)
        money_risky = ratio * wealth
        money_riskless = (1 - ratio) * wealth
        if self.model[current_time] == 'b':
            if random.random() < self.p:
                Yt = self.a
            else:
                Yt = self.b
        else:
            Yt = random.normalvariate(self.miu, self.sigma)
        wealth = money_risky * (1 + Yt) + money_riskless * (1 + self.r)
        current_time = current_time + 1
        self.state = (current_time, wealth)
        done = current_time >= len(self.model)
        reward = (1 - np.exp(-self.util_a*wealth)) / self.util_a if done else 0
        return self.state, reward, done, Yt # Here return Yt is used to visualized checking

class Agent:
    def __init__(self, time_steps=10, degree=2, num_actions=6):
        # degree = 2 means linear and 3 means quadratic and 4 means cubic
        self.coefficients = np.zeros([num_actions, time_steps, degree])

    def get_q_values(self, state):
        time_step, wealth = state # state[0] is time step and state[1] is wealth
        q_values = []
        for action in range(self.coefficients.shape[0]):
            q_value = sum([self.coefficients[action, time_step, i] * wealth**i for i in range(self.coefficients.shape[2])])
            q_values.append(q_value)
        return q_values

    def update(self, state, action, target, lr=0.01):
        time_step, wealth = state
        current_q = self.get_q_values(state)[action]
        error = target - current_q
        for i in range(self.coefficients.shape[2]):
            self.coefficients[action, time_step, i] += lr * error * wealth**i

def train(env, agent, episodes=1000, lr=0.01, gamma=0.99, explore=1.0, explore_decrease=0.99, num_actions=6):
    for episode in range(episodes):
        explore *= explore_decrease
        if episode % 100000==0:
            print(f'episode {episode+1}/{episodes}')
        state = env.reset()
        done = False
        while not done:
            if random.random() < explore:
                action = random.randint(0, num_actions-1)
            else:
                q_values = agent.get_q_values(state)
                action = np.argmax(q_values)
            next_state, reward, done, _ = env.step(action)
            target = gamma * (reward if done else max(agent.get_q_values(next_state)))
            # print('target=',target)
            agent.update(state, action, target, lr)
            state = next_state

print('===== test 1 - single step normal distribution =====')

def q_normal(Wt, xt, miu, sigma, r, util_a, gamma, T, t):
    d = np.exp(-((miu-r)**2*(T-t-1))/(2*sigma**2))
    e = -util_a*(1+r)**(T-t)*Wt - util_a*(miu-r)*(1+r)**(T-t-1)*xt + (util_a*sigma*(1+r)**(T-t-1))**2/2*xt**2
    return gamma ** (T - t) * (1-d * np.exp(e)) / util_a

def xt_star_normal(miu, sigma, r, util_a, T, t):
    return (miu-r) / (sigma **2 * util_a * (1+r)**(T-t-1))

num_actions = 11 # [0, 0.1, 0.2 ... 1.0]
miu = 0.02
sigma = 0.1
util_a = 2.0
# best option should be (0.02-0.01)/0.01/2 = 0.5 (i.e. locate $0.5 (50% of $1 at initial))
env = Environment(num_actions=num_actions, miu=miu, sigma=sigma, model='n', util_a=util_a)
agent = Agent(time_steps=1, degree=3, num_actions=num_actions)
train(env, agent, lr=0.001, episodes=300000, explore_decrease=1, num_actions=num_actions) # explore_decrease=1 since no need to exploit since no following time steps in this case

# Test on trained trading strategy
state = env.reset()
done = False
while not done:
    q_values = agent.get_q_values(state)
    correct_q_values = [q_normal(state[1], action / (num_actions - 1), miu, sigma, 0.01, util_a, 0.99, 1, 0) for action in range(num_actions)]
    print('modeled q_values:', q_values)
    print('theoretical q_values:', correct_q_values)
    action = np.argmax(q_values)
    state, reward, done, Yt = env.step(action)
    print(f'time={state[0]}, wealth={state[1]}, allocated {action/(num_actions-1)*100:.0f}% at risky asset in previous time with return {Yt}. reward={reward}')

print('===== test 2 - multiple step normal distribution =====')
env = Environment(num_actions=num_actions, miu=miu, sigma=sigma, model='n'*5, util_a=util_a)
agent = Agent(time_steps=5, degree=3, num_actions=num_actions)
train(env, agent, lr=0.01, episodes=1000000, explore_decrease=0.999998, num_actions=num_actions)

# Test on trained trading strategy
state = env.reset()
done = False
while not done:
    q_values = agent.get_q_values(state)
    correct_q_values = [q_normal(state[1], action / (num_actions - 1), miu, sigma, 0.01, util_a, 0.99, 5, state[0]) for action in range(num_actions)]
    print('modeled q_values:', q_values)
    print('theoretical q_values:', correct_q_values)
    correct_x_star = xt_star_normal(miu, sigma, 0.01, util_a, 5, state[0])
    print(f'theoretical x*: ${correct_x_star:.3f} ({correct_x_star/state[1]*100:.1f}%)')
    action = np.argmax(q_values)
    state, reward, done, Yt = env.step(action)
    print(f'time={state[0]}, wealth={state[1]}, allocated {action/(num_actions-1)*100:.0f}% at risky asset in previous time with return {Yt}. reward={reward}')

print('===== test 3 - single step binomial distribution =====')
# ay, by is two possible Yt with probability p and 1-p
# util_a is a in utility function
def q_binomial(Wt, xt, ay, by, p, r, util_a, gamma, T, t):
    up = -p * (ay - r) / ((1-p) * (by - r))
    d_star = p * up**((ay-r)/(ay-by)) + (1-p) * up**((by-r)/(ay-by))
    ctp1 = util_a * (1+r)**(T-t-1) #(c_(t+1))
    ee = np.exp(-ctp1*Wt*(1+r)) * (p * np.exp(-ctp1*(ay-r)*xt) + (1-p) * np.exp(-ctp1*(by-r)*xt))
    return gamma ** (T-t) / util_a * (1- d_star**(T-t-1) * ee)

def xt_star_binomial(ay, by, p, r, util_a, T, t):
    ctp1 = util_a * (1+r)**(T-t-1) #(c_(t+1))
    return np.log (-p * (ay - r) / ((1-p) * (by - r))) / (ctp1 * (ay - by))

num_actions = 11 # [0, 0.1, 0.2 ... 1.0]
ay = 0.1
by = -0.04
p = 0.4
util_a = 2.0
r = 0.01
# best option should be xt_star_binomial(ay, by, p, r, util_a, 1, 0) = 0.65 (i.e. locate $0.65 (65% of $1 at initial))
env = Environment(num_actions=num_actions, a=ay, b=by, p=p, model='b', r=r, util_a=util_a)
agent = Agent(time_steps=1, degree=3, num_actions=num_actions)
train(env, agent, lr=0.001, episodes=300000, explore_decrease=1, num_actions=num_actions) # explore_decrease=1 since no need to exploit since no following time steps in this case

# Test on trained trading strategy
state = env.reset()
done = False
while not done:
    q_values = agent.get_q_values(state)
    correct_q_values = [q_binomial(state[1], action / (num_actions - 1), ay, by, p, r, util_a, 0.99, 1, 0) for action in range(num_actions)]
    print('modeled q_values:', q_values)
    print('theoretical q_values:', correct_q_values)
    action = np.argmax(q_values)
    state, reward, done, Yt = env.step(action)
    print(f'time={state[0]}, wealth={state[1]}, allocated {action/(num_actions-1)*100:.0f}% at risky asset in previous time with return {Yt}. reward={reward}')

print('===== test 4 - multiple step normal distribution =====')
env = Environment(num_actions=num_actions, a=ay, b=by, p=p, model='b'*5, util_a=util_a)
agent = Agent(time_steps=5, degree=3, num_actions=num_actions)
train(env, agent, lr=0.01, episodes=1000000, explore_decrease=0.999998, num_actions=num_actions)

# Test on trained trading strategy
state = env.reset()
done = False
while not done:
    q_values = agent.get_q_values(state)
    correct_q_values = [q_binomial(state[1], action / (num_actions - 1), ay, by, p, r, util_a, 0.99, 5, state[0]) for action in range(num_actions)]
    print('modeled q_values:', q_values)
    print('theoretical q_values:', correct_q_values)
    correct_x_star = xt_star_binomial(ay, by, p, 0.01, util_a, 5, state[0])
    print(f'theoretical x*: ${correct_x_star:.3f} ({correct_x_star/state[1]*100:.1f}%)')
    action = np.argmax(q_values)
    state, reward, done, Yt = env.step(action)
    print(f'time={state[0]}, wealth={state[1]}, allocated {action/(num_actions-1)*100:.0f}% at risky asset in previous time with return {Yt}. reward={reward}')

print('===== assignment - 5 step binormal and 5 step normal =====')
env = Environment(num_actions=num_actions) # default parameters are used for assignment 1
agent = Agent(time_steps=10, degree=3, num_actions=num_actions)
train(env, agent, lr=0.01, episodes=1000000, explore_decrease=0.999998, num_actions=num_actions)

# Test on trained trading strategy
state = env.reset()
done = False
while not done:
    q_values = agent.get_q_values(state)
    correct_q_values = [q_binomial(state[1], action / (num_actions - 1), ay, by, p, r, util_a, 0.99, 10, state[0]) for action in range(num_actions)]
    print('modeled q_values:', q_values)
    action = np.argmax(q_values)
    state, reward, done, Yt = env.step(action)
    print(f'time={state[0]}, wealth={state[1]}, allocated {action/(num_actions-1)*100:.0f}% at risky asset in previous time with return {Yt}. reward={reward}')