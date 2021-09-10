

from os import path
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import logging
import time
import datetime
from agent import Agent
from bristle import Bristle
from replay_memory import ReplayMemory
import random
import torch
import matplotlib
from tensorboardX import SummaryWriter


ACTOR_LR = 2e-3  # Actor net learning rate
CRITIC_LR = 2e-3  # Critic net learning rate
GAMMA = 0.99  # reward Attenuation factor
TAU = 0.3  # Coefficient of soft update
MEMORY_SIZE = int(5e3)  # Experience pool size
MEMORY_WARMUP_SIZE = MEMORY_SIZE/2  # Pre-save some experience before you start training
BATCH_SIZE = 152
REWARD_SCALE = 52  # reward Scaling factor
TRAIN_EPISODE = 50  # Total number of episodes trained
Pendulum = 800  # Number of steps per episode


# train the first episode
def run_episode(agent, env, rpm, e_greedy = 0.9):
    obs = env.reset()
    total_reward = 0

    steps = 0
    while True:
        steps += 1
        #print(steps)
        #print(obs.shape)
        batch_obs = np.expand_dims(obs, axis=0)
        #print(batch_obs)
        #print(batch_obs)
        #print(batch_obs.shape)
        #time_start=time.time()
        action = agent.predict(batch_obs.astype('float32'))
        #[1,0,0,0]

        k = random.random()
        if(k > e_greedy): #Random Actions
            #print(e_greedy)
            action = np.zeros(4)
            action[random.randint(0,3)] = 1

        #time_end=time.time()

        # Add exploration perturbation, output limited to [-1.0, 1.0] range
        #action = np.clip(np.random.normal(action, NOISE), -2.0, 2.0)
        #action = np.random.normal(action, NOISE)#Add Gaussian noise to explore
        #print(type(action))
        next_obs, reward, done, info = env.step(action)
        #print(next_obs.shape)#(16,)
        #next_obs = next_obs.squeeze(1)
        #action = [action]  #
        if(done):
            rpm.append_god((obs, action, REWARD_SCALE * reward, next_obs, done))
        else:
            rpm.append_normal((obs, action, REWARD_SCALE * reward, next_obs, done))

            u = np.argmax(action)
            action1 = np.zeros(4)
            if(u == 0):
                action1[1] = 1
            elif(u == 1):
                action1[0] = 1
            elif(u == 2):
                action1[3] = 1
            elif(u == 3):
                action1[2] = 1
            rpm.append_normal((next_obs, action1, -REWARD_SCALE * reward, obs, done))



        if len(rpm) > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            #batch_action = batch_action.squeeze(1)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)

        obs = next_obs
        total_reward += reward
        #print(total_reward)

        if done or steps >= Pendulum:
            print('Pendulum='+str(steps))
            #print(env.lastdis,total_reward)
            break
    return total_reward


# Evaluate agent, run 5 episodes, average total rewards
def evaluate(env, agent):
    eval_reward = []
    for i in range(4):
        #print(i,end='')
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            #print(batch_obs.shape)
            action = agent.predict(batch_obs.astype('float32'))
            #action = np.clip(action, -2.0, 2.0)

            steps += 1
            next_obs, reward, done, info = env.step(action)
            #next_obs = next_obs.squeeze(1)
            #print(reward)

            obs = next_obs
            total_reward += reward

            if done or steps >= Pendulum:#Pendulum
                #total_reward /= (env.first_dis1 * 100)
                break
        eval_reward.append(total_reward)
    #print()
    return np.mean(eval_reward)

# Evaluate agent, run 5 episodes, average total rewards
def test(env, agent):
    eval_reward = []
    for i in range(4):
        #print(i,end='')
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))

            steps += 1
            next_obs, reward, done, info = env.step(action)
            #next_obs = next_obs.squeeze(1)
            obs = next_obs
            total_reward += reward

            if done or steps >= Pendulum:#Pendulum
                #total_reward /= (env.first_dis1 * 100)
                break
        eval_reward.append(total_reward)
    #print()
    return np.mean(eval_reward)

def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: Folder path to log files
    mode: 'a', append; 'w', Overwrite the original file to write.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # Logging output cannot be used here
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging

def main():

    logging = init_logger(log_dir='log')
    logging.info('Code Start!')
    writer = SummaryWriter(log_dir='log')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    env = Bristle()



    #env.reset()

    testflag = 1

    params = {
        'gamma': 0.99,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 32,

        'obs_dim':2,
        'act_dim':4,

    }
    agent = Agent(**params)

    if(testflag):
        name = './bristle_model/' + str(65)
        if os.path.exists(name + 'actor.pkl'):
            agent.load_weights(name)
            env.train_flag = 0
            test_reward = test(env, agent)
            logging.info('total_reward:{}'.format(test_reward))
            env.close()
            return 0


    # Create an experience pool
    logging.info('MEMORY Start!')
    rpm = ReplayMemory(MEMORY_SIZE)
    # Pre-stocking data into the experience pool
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(agent, env, rpm, e_greedy = 0.1)
    #print("MEMORY Done!")
    #env.render()
    logging.info('MEMORY Done!')

    Episode = []
    Total_reward =[]
    episode = 0
    best_eval_reward = 0
    while episode < TRAIN_EPISODE:
        #E_GREEDY = min(0.9,max(0.2 + episode/100, 0.2))
        #print(f"E_GREEDY:{E_GREEDY}")
        for i in range(5):
            episode += 1
            e_greedy = min(0.9, episode/10)
            total_reward = run_episode(agent, env, rpm, e_greedy)
            writer.add_scalar('train/total_reward', total_reward, episode)

            #print(f"episode:{episode},total_reward:{total_reward/(env.first_dis1 * 100)}")
            print(f"episode:{episode},total_reward:{total_reward}")
            Episode.append(episode)
            Total_reward.append(total_reward)
        #total_reward = run_episode(agent, env, rpm)
        env.train_flag = 0
        eval_reward = evaluate(env, agent)
        env.train_flag = 1        #print("1")
        logging.info('episode:{}    Test reward:{}'.format(episode, eval_reward))

        writer.add_scalar('eval/eval_reward', eval_reward, episode)

        if(episode != 0 and eval_reward>0 and eval_reward > best_eval_reward):
            best_eval_reward = eval_reward
            savename = './bristle_model/' + str(episode)
            agent.save_model(savename)

    agent.save_model('./bristle_model')
    writer.close()
    env.close()

    matplotlib.plot(Episode,Total_reward)

if __name__ == '__main__':
    main()
