import numpy as np
import os
import pickle

class MAZE:
    def __init__(self,size,max_length,sparse):
        self.size = size
        self.maze = np.zeros((size,size))
        self.state_dim = 2
        self.act_dim = 5
        self.max_length = max_length
        self.sparse = sparse
        self.target_pos_x = np.random.randint(0, self.size)
        self.target_pos_y = np.random.randint(0, self.size)

    def reset(self):
        # self.maze = np.zeros(self.size,self.size)
        self.agent_pos_x = self.size//2+1
        self.agent_pos_y = self.size//2+1
        self.res_length = self.max_length
        self.reach_time = 0
        self.cur_distance = self.size*2
        return np.array([self.agent_pos_x,self.agent_pos_y]),np.array([self.target_pos_x,self.target_pos_y])

    def step(self, action):
        if action == 0:
            self.agent_pos_x += 0
            self.agent_pos_y += 0
        elif action == 1:
            self.agent_pos_x +=0
            if self.agent_pos_y+1<self.size:
                self.agent_pos_y += 1
        elif action == 2:
            self.agent_pos_x += 0
            if self.agent_pos_y - 1 >= 0:
                self.agent_pos_y -= 1
        elif action == 3:
            if self.agent_pos_x - 1 >= 0:
                self.agent_pos_x -= 1
            self.agent_pos_y +=0
        elif action == 4:
            if self.agent_pos_x + 1 < self.size:
                self.agent_pos_x += 1
            self.agent_pos_y +=0
        else:
            print("invalid action ！！！！！！！！！！！！！！")
            exit()
        distance = abs(self.target_pos_x - self.agent_pos_x) + abs(self.target_pos_y - self.agent_pos_y)
        if self.agent_pos_x==self.target_pos_x and self.agent_pos_y==self.target_pos_y:
            self.reach_time+=1
        if self.sparse:
            if self.reach_time==1:
                reward = 1
            else: reward = 0
        else:
            if distance<self.cur_distance:
                reward = 1
                self.cur_distance = distance
            else: reward = 0
        self.res_length -= 1
        if self.res_length ==0 or self.reach_time>0:
            done = True
        else: done = False

        return np.array([self.agent_pos_x,self.agent_pos_y]), reward, done


class ScriptAgent:
    def __init__(self, gamma=0.5):
        self.gamma = gamma

    def reset(self):
        self.gamma = np.random.uniform(0, 1)

    def get_action(self,obs,target_pos):
        x,y = obs[0],obs[1]
        tx,ty = target_pos[0],target_pos[1]
        disance_x = abs(x-tx)
        disance_y = abs(y-ty)
        tmp = np.random.uniform(0, 1)
        if tmp<self.gamma:
            action = np.random.randint(0,5)
        else:
            if disance_x==0 and disance_y==0:
                action = 0
            elif disance_x<=disance_y:
                if y<ty:
                    action = 1#
                else: action = 2#
            else:
                if x<tx:
                    action = 4#
                else: action = 3#
        return action

    def set_gamma(self,gamma):
        self.gamma = gamma


if __name__ == '__main__':
    data_path = os.getcwd()
    env_nums = 10
    maze_size = 9
    max_length = 20
    sparse = False
    envs = [MAZE(maze_size,max_length,sparse) for i in range(env_nums)]
    agent = ScriptAgent()
    max_steps = 1e6 // env_nums
    for env_id in range(env_nums):
        name = f'Darkroom-{env_id}'
        env = envs[env_id]
        env_data = []
        env_step = 0
        while env_step<max_steps:
            traj = dict()
            observations, actions, rewards, terminals = [], [], [], []
            done = False
            agent.reset()
            obs, target = env.reset()
            while not done:
                observations.append(obs)
                act = agent.get_action(obs,target)
                actions.append(act)
                obs,r,done = env.step(act)
                rewards.append(r)
                terminals.append(done)
                env_step += 1
            traj['observations'] = np.array(observations)
            traj['rewards'] = np.array(rewards)
            traj['actions'] = np.array(actions)
            traj['terminals'] = np.array(terminals)
            env_data.append(traj)
            print("="*10)
            print('env_id:',str(env_id),'   env_step:',str(env_step))
            print("="*10)

        with open(data_path + '/' + f'{name}.pkl', 'wb') as f:
            pickle.dump(env_data, f)
