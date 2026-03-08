from callback import PerformanceBasedTermination
import os
from customwrappers.RandomVecEnv import RandomSubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import gymnasium
import torch
import minari
from sbx import SAC, PPO
from utils.env_utils import get_parameter_from_env
import shutil
class DataCollector:
    def __init__(self, 
                 policy= None,
                 base_path = "./data/", 
                 task_num = 40,  
                 device = 'cuda:0' if torch.cuda.is_available() else 'cpu,',
                 episode_per_task = 50
                 ):

        
        self.base_path = base_path
        self.episode_per_task = episode_per_task
        self.device = device
        self.task_num = task_num

        
        self.policy = policy if policy is not None else "SAC"

    def collect_data(self,train = True, reward_target = None,callback = None,model = None, saved_model_dir = None):
        par_env = model.env
        #if par_env is not parallel env, make it parallel
        if not isinstance(par_env, RandomSubprocVecEnv):
            par_env = make_vec_env(lambda: par_env, n_envs=4) 
            #raise warning
            raise Warning("The provided model does not have a parallel environment. Creating a new parallel environment with 4 processes.")

        if model is None:
            if self.policy == "SAC":
                model = SAC("MlpPolicy", par_env, verbose=1, device=self.device, tensorboard_log=self.base_path+"tb_logs/")
                raise Warning("No model provided. Training a new SAC model from scratch.")
            else:
                raise NotImplementedError("Only SAC is implemented for data collection.")
        if callback is None:
            callback = PerformanceBasedTermination(eval_env=par_env, log_dir=self.base_path+"tb_logs/", patience=10)
            raise Warning("No callback provided. Using default PerformanceBasedTermination callback.")
        if not isinstance(saved_model_dir,(list,np.ndarray)):
            saved_model_dir = [saved_model_dir]
 

        param = get_parameter_from_env(par_env)
        
        trained_task = []
        trained_model_path = []
        if saved_model_dir is not None:
            for path in saved_model_dir:
                if path is None:
                    continue
                
                # else:
                #     trained_model = np.load(path+"trained_tasks.npy", allow_pickle=True)
                #     print(f"Loading model from {path} for tasks {trained_model}")
                #     for model in trained_model:
                #         trained_task.append(model)
                #         trained_model_path.append(path+f"{np.round(model,2)}/")
                
                # Ensure path ends with /
                if not path.endswith('/'):
                    path += '/'
                
                # Helper to track added tasks
                added_tasks_hashes = set()
                def get_task_hash(t):
                    return tuple(np.round(t, 4).tolist())

                # # 1. Try loading from trained_tasks.npy
                # # if os.path.exists(path + "trained_tasks.npy"):
                # #     try:
                # #         trained_model = np.load(path+"trained_tasks.npy", allow_pickle=True)
                # #         print(f"Loading model from {path} for tasks {trained_model}")
                # #         for model in trained_model:
                # #             model_path_candidate = path+f"{np.round(model,2)}/"
                # #             if os.path.exists(model_path_candidate):
                # #                 trained_task.append(model)
                # #                 trained_model_path.append(model_path_candidate)
                # #                 added_tasks_hashes.add(get_task_hash(model))
                # #     except Exception as e:
                # #         print(f"Warning: Error loading trained_tasks.npy: {e}")

                # 2. Scan directory for task folders
                if os.path.exists(path):
                    items = sorted(os.listdir(path))
                    # Prioritize specific task
                    priority_task = "[3.53 3.93 2.71 2.94 3.93 2.71 2.94 0.4  0.45 0.6  0.2  0.9  1.9 ]"
                    if priority_task in items:
                        items.remove(priority_task)
                        items.insert(0, priority_task)
                        
                    for item in items:
                        item_path = os.path.join(path, item)
                        clean_item = item.strip()
                        if os.path.isdir(item_path) and clean_item.startswith('[') and clean_item.endswith(']'):
                            content = clean_item[1:-1].replace(',', ' ')
                            task_values = [float(x) for x in content.split()]
                            task = np.array(task_values)
                            
                            task_hash = get_task_hash(task)
                            if task_hash not in added_tasks_hashes:
                                trained_task.append(task)
                                trained_model_path.append(item_path + "/")
                                added_tasks_hashes.add(task_hash)
                                print(f"Found task folder: {item}")


        print(f"Number of already trained model paths: {len(trained_model_path)}")
        episodes = []
        ref_max_scores = []
        ref_min_scores = []
        for i in range(self.task_num):
            ref_reward = None
            if i < len(trained_task):
                train = False
                task = trained_task[i]
                par_env.set_task(np.tile(task, (param['num_envs'],1)))
                model_path = trained_model_path[i]
                print(f"Skipping already trained task {task}")
                #run an episode to get ref reward
                model = SAC.load(model_path+"best_model.zip", env=par_env, device=self.device)

            else:
                train = True
                if i == 0:
                    task = param['default_task']
                else:
                    task = par_env.env_method('get_random_task')[0]
                model_path = self.base_path + f"{np.round(task,4)}/"

                par_env.set_task(np.tile(task, (param['num_envs'],1)))
                if callback is not None:
                    callback.model_save_path = model_path
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                
                print(f"Training on task: {task}")
                if trained_task != []:
                    # find the nearest task in trained_task
                    nearest_task_index = np.argmin(np.linalg.norm(
                                                (np.array(trained_task)-task[None,:])/param['max_task'][None,:]
                                                , axis=1))
                    
                    nearest_task = trained_task[nearest_task_index]
                    print(f"Loading model from nearest task {nearest_task} for training")
                    model = SAC.load(self.base_path + f"{np.round(nearest_task,4)}/best_model.zip", env=par_env, device=self.device)
                model.learn(
                                total_timesteps=param['max_train_step'],
                                log_interval=10,
                                callback=callback,
                                tb_log_name=f"task_{i}",
                                )
                model.save(model_path+f"last_model.zip")


                if callback.best_mean_reward < reward_target and reward_target is not None and train:
                    print(f"Discard the model as mean reward {callback.best_mean_reward} is less than target {reward_target}")
                    shutil.rmtree(model_path,ignore_errors= True)  # This line is causing the error
                    continue

                callback.reset()
            



            # clear the replay buffer
            model = SAC.load(model_path+f"best_model.zip", env=par_env, device=f"cuda:1" if torch.cuda.is_available() else "cpu")
            ob = par_env.reset()
            ref_reward = 0.0
            episode_length = 0
            cum_done = 0
            while not np.all(cum_done):
                action = model.predict(ob, deterministic=True)[0]
                ob, reward, done, info = par_env.step(action)
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                if episode_length >= 1000:
                    break
                episode_length += 1
                
                ref_reward += reward.mean()
            print(f"Reference reward for task {task} is {ref_reward}")
            print(f"Episode length: {episode_length}")
            if ref_reward < reward_target and reward_target is not None:
                print(f"Skipping task {task} as reference reward {ref_reward} is less than target {reward_target}")
                continue
            # if episode_length < 500:
            #     print(f"Skipping task {task} as episode length {episode_length} is less than 500")
            #     continue

            
            print(f"Collecting data on task: {task}")
            ref_max_score = -np.inf
            ref_min_score = None
            episode_collected = 0
            
            while True:
                # par_env.set_task(np.tile(task, (param['num_envs'],1))*np.random.uniform(0.9,1.1,size=(param['num_envs'],task.shape[0])))
                par_env.set_task(np.tile(task, (param['num_envs'],1)))
                obss = np.empty((param['num_envs'],0,param['observation_space'].shape[0]))
                actions = np.empty((param['num_envs'],0, param['action_space'].shape[0]))
                rewards = np.empty((param['num_envs'],0))
                terminations = np.empty((param['num_envs'],0))
                done = np.zeros(param['num_envs'], dtype=bool)
                obs = par_env.reset()
                cum_done = 0
                for _ in range(1000):
                    action = model.predict(obs, deterministic=True)[0]
                    obss = np.append(obss, obs[:, np.newaxis, :], axis=1)
                    actions = np.append(actions, action[:, np.newaxis, :], axis=1)
                    obs, reward, done, info = par_env.step(action)
                    cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                    reward = reward * (1 - cum_done)
                    rewards = np.append(rewards, reward[:, np.newaxis], axis=1)
                    terminations = np.append(terminations,cum_done[:, np.newaxis], axis=1)
                accu_rewards = rewards.sum(axis = 1)
                print(accu_rewards)
                print(accu_rewards.mean())
                
                print(accu_rewards.std())

                for j in range(par_env.num_envs):

                    if np.sum(rewards[j]) < accu_rewards.mean() - 3*accu_rewards.std() or np.sum(rewards[j]) < reward_target:
                        print(f"Episode {j} discarded due to low reward: {np.sum(rewards[j])}")
                        continue
                    if np.sum(terminations[j]) > 1:
                        print(f"Episode {j} discarded due to early stop")
                        continue

                    ref_max_score = max(ref_max_score, np.sum(rewards[j]))
                    ref_min_score = min(ref_min_score, np.sum(rewards[j])) if ref_min_score is not None else np.sum(rewards[j])
                    truncations = np.zeros(obss.shape[1])

                    truncations[-1] = 1
                    infos = {}
                    infos['task_index'] = i * np.ones(obss.shape[1])
                    infos['individual_task'] = par_env.get_task()[j]
                    infos['task'] = task
                    infos['ref_score'] = np.sum(rewards[j])
                    infos['no_randomization_score'] = ref_reward if ref_reward is not None else callback.best_mean_reward
                    episodes.append(minari.data_collector.EpisodeBuffer(len(episodes),
                                                    observations=obss[j],
                                                    actions=actions[j],
                                                    rewards=rewards[j],
                                                    terminations = terminations[j],
                                                    truncations = truncations,
                                                     infos = infos))
                    episode_collected += 1
                    
                    if episode_collected >= self.episode_per_task:
                        break
                print(f"Episode {len(episodes)} collected with {obss[j].shape[0]} timesteps")
                if episode_collected >= self.episode_per_task:
                    ref_max_scores.append(ref_max_score)
                    ref_min_scores.append(ref_min_score)

                    break
            if train:
                trained_task.append(task)
                np.save(self.base_path+"trained_tasks.npy", np.array(trained_task))
        spec_id_base = param['spec_id'][:-3]
        dataset_id = f'{spec_id_base}/{self.task_num}dynamics-v7'
        # dataset_id = f'{param['spec_id'][:-3]}/{self.task_num}dynamics-v1'


        dataset = minari.create_dataset_from_buffers(dataset_id,
                                                        buffer = episodes,
                                                        env = gymnasium.make(param['spec_id']),ref_max_score = ref_max_scores,
                                                        ref_min_score = ref_min_scores,
                                                        )


