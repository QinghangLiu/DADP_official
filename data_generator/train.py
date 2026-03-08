# from OpenGL.GL import glGetError
# print(glGetError())
# import gym
import os
  # Set your desired GPU here
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import gymnasium
import dr_envs
import argparse
import numpy as np
from customwrappers.RandomVecEnv import RandomSubprocVecEnv
from cleandiffuser.dataset.d4rl_mujoco_dataset import  D4RLMuJoCoTDDataset, RandomMuJoCoSeqDataset
# from utils import seed_everything
from utils.env_utils import *
from utils.gym_utils import *
import wandb
import torch
import imageio
import glob

from sbx import SAC,PPO
import time
import minari
from dataCollector import DataCollector
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from callback import PerformanceBasedTermination
def seed_everything(seed,**kwarg):
    '''Set seed for reproducibility.
    You can seed env here by passing env=env in kwarg.
    '''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if kwarg .get("env") is not None:
       kwarg["env"].seed(seed)

if __name__ == "__main__":
    wandb.init(project="your_project_name",mode = "disabled")
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',           default="sac", type=str, help='Algorithm to use')
    parser.add_argument('--env',            default='RandomAnt-v0', type=str, help='Train gym env')
    parser.add_argument('--lr',             default=None, type=float, help='Learning rate')
    parser.add_argument('--gamma',          default=0.99, type=float, help='gamma discount factor')
    parser.add_argument('--now',            default=20, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--gradient_steps', default=20, type=int, help='Number of gradient steps per update')
    parser.add_argument('--base_path',     default="./best_model/", help='Train from scratch')
    parser.add_argument('--mode',           default="train_and_data", help='Evaluate the policy')
    # parser.add_argument('--task',           default=np.array([8,7,7,7,7,7,7,0.9,0.78,0.2,0.9,2.4,0.8]), help='Use wandb for logging')
    parser.add_argument('--video_save_path',default="./video", help='Path to save video')
    parser.add_argument('--data_collect_episode',            default=100, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--gpu',           default=1, type=int, help='Which gpu to use')
    parser.add_argument('--seed',          default=42, type=int, help='Random seed')
    #callback for early stopping and evaluation
    parser.add_argument('--eval_freq',     default=5000, type=int, help='Evaluation frequency')
    parser.add_argument('--num_eval_episodes', default=1, type=int, help='Number of episodes for each evaluation')
    parser.add_argument('--patience',      default=5, type=int, help='Patience for early stopping')
    parser.add_argument('--min_improvement', default=0.05, type=float, help='Minimum improvement to reset patience')
    parser.add_argument('--std_threshold', default=0.05, type=float, help="Standard deviation threshold for early stopping")
    parser.add_argument('--save_freq',    default=10000, type=int, help='Model save frequency')
                        
    args = parser.parse_args()

    base_path = args.base_path
    base_path = base_path + f"{args.env}/"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_path = base_path + f"{timestamp}/"
    set_random_seed(args.seed,True)

    env = gymnasium.make(args.env)
    env.unwrapped.set_task(*env.unwrapped.get_random_task())
    default_task = env.unwrapped.get_task()
    bound = np.zeros((env.unwrapped.dyn_ind_to_name.__len__(),2))
    for i in range(env.unwrapped.dyn_ind_to_name.__len__()):
        bound[i,:] = np.array(env.unwrapped.get_search_bounds_mean(i))
    series_task = np.linspace(bound[:,0],bound[:,1], 5)
    # series_task = default_task.copy()



    task_num = env.unwrapped.task_seq.__len__() + 1


    print('='*50)
    print(f"Training on environment: {args.env} with {task_num} tasks")
    print('='*50)


    # for _ in range(task_num):
    #     a = env.unwrapped.get_random_task()
    # task_list = [default_task]
    # for i in range(1,task_num):
    #     task_list.append(default_task + np.eye(default_task.shape[0])[0]*(i-3)*0.4)


    if os.path.exists(base_path):
        print(f"Warning: {base_path} already exists!")
    else:
        os.makedirs(base_path)

    #save arguments as .json
    import json
    with open(base_path+'args.json', 'w') as f:
        json.dump(vars(args), f, indent=4) 


    
    par_env = make_vec_env(args.env, n_envs=args.now,  vec_env_cls=RandomSubprocVecEnv)
    model = SAC("MlpPolicy",
                 par_env, 
                 learning_rate=3e-4 if args.lr is None else args.lr, 
                 gamma=args.gamma, 
                 gradient_steps=args.gradient_steps, 
                 verbose=1, 
                 device='cuda:0' if torch.cuda.is_available() else "cpu",
                 tensorboard_log=base_path+"logs/",)



    termination_callback = PerformanceBasedTermination(par_env, 
                                        eval_freq=args.eval_freq, 
                                        n_eval_episodes=args.num_eval_episodes, 
                                        patience=args.patience, 
                                        least_train_step = env.unwrapped.least_train_step if env.unwrapped.least_train_step is not None else 1000000,

                                        min_improvement=args.min_improvement,
                                        std_threshold=args.std_threshold, 
                                        target_reward=None)

    datacollector = DataCollector(task_num=task_num,
                                  base_path=base_path,
                                  episode_per_task=args.data_collect_episode,
                                  device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
                                    )
    datacollector.collect_data(train = args.mode != "data",
                              reward_target = 0,
                              model = model,
                              callback = termination_callback,
                              saved_model_dir=
                                                # None
                              [
                               './best_model/'+args.env+'/20251211-190427/',
                               './best_model/'+args.env+'/20251211-190539/',
                               './best_model/'+args.env+'/20251211-190609/',]


                              )
    # print(env.get_task())
    # env.set_task(*task)
    # par_env = make_vec_env(args.env, n_envs=args.now,  vec_env_cls=RandomSubprocVecEnv)

    # # par_env.set_task(np.tile(task, (args.now,1)))
    # eff_lr = get_learning_rate(args, par_env)
    
    # seed_everything(args.seed, env=par_env)
    # # print(f"Initial GPU: {torch.cuda.current_device()}")
    # # # torch.cuda.set_device(args.gpu)
    # # print(f"Set GPU to: {torch.cuda.current_device()}")
    # device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    # model = SAC("MlpPolicy",
    #              par_env, 
    #              learning_rate=eff_lr, 
    #              gamma=args.gamma, 
    #              gradient_steps=args.gradient_steps, 
    #              verbose=1, 
    #              device=device,
    #              tensorboard_log=base_path+"logs/",)
    # # print(f"Set GPU to: {torch.cuda.current_device()}")

    # if args.mode == "eval":
        
    #     model = SAC.load(f"./best_model/best_model_{args.env}_{np.zeros((len(default_task),))}.zip", env=par_env, device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    #     accu_reward = 0
    #     frames = []
    #     print(f"=============== Parameter of Dynamics ====================================")
    #     print(env.get_task())
    #     obs = env.reset()

    #     done = False
    #     random_action = 0
    #     for _ in range(16):
    #         obs,_,_,_ = env.step(np.random.uniform(low = -1,high = 1,size = (par_env.action_space.shape[0])))
    #         frame = env.render(mode = "rgb_array")
    #     while not done:

    #         action = model.predict(obs, deterministic=True)[0]

    #         obs, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished due to:", info.get('terminated_due_to_fall', 'unknown reason'))
    #             print(obs)
    #         frame = env.render(mode = "rgb_array")  # This returns an RGB array
    #         frames.append(frame)
    #         accu_reward += reward
    #     # Save video
    #     print(f"Total reward: {accu_reward}")
    #     video_path = f"{args.video_save_path}/{args.env}"
    #     if not os.path.exists(video_path):
    #         os.makedirs(video_path)
    #     task_str = "_".join(map(str, np.array(args.task).flatten()))
    #     imageio.mimsave(f"{video_path}/evaluation{timestamp}_task{task_str}.gif", frames, fps=30)

    # elif args.mode == "train":
    #     accu_reward = 0.01
    #     cnt_accureward = 0
    #     while not (0 < (cnt_accureward - accu_reward)/accu_reward < 1e-2):
    #         accu_reward = cnt_accureward
    #         print(par_env.get_task())
    #         model.learn(
    #                     total_timesteps=400000,
    #                     log_interval=10,
    #                     )
    #         # model.save("./best_model/best_model_heavy.zip")
    #         cnt_accureward = 0
    #         frames = []
    #         env = gym.make(args.env)
    #         obs = env.reset()
    #         done = False
    #         while not done:
    #             action = model.predict(obs, deterministic=True)[0]
    #             obs, reward, done, info = env.step(action)
    #             cnt_accureward += reward
    #     frame = env.render(mode = "rgb_array")  # This returns an RGB array
    #     frames.append(frame)
    #     # Save video
    #     print(f"Total reward: {accu_reward}")
    #     video_path = f"{args.video_save_path}/{args.env}"
    #     if not os.path.exists(video_path):
    #         os.makedirs(video_path)
    #     task_str = "_".join(map(str, np.array(args.task).flatten()))
    #     imageio.mimsave(f"{video_path}/evaluation{timestamp}_task{task_str}.gif", frames, fps=30)
    # elif args.mode == "data":

    #     dataset = minari.load_dataset("RandomWalker2d/40dynamics-v1")

    #     episodes = []
    #     dynamics_models_list = glob.glob("./best_model/RandomWalker2d-v0/20250925-195614/*/")
    #     dynamics_models_list = sorted(dynamics_models_list,key=lambda x:os.stat(x).st_ctime)
    #     task_index = 0
    #     planner_dataset = RandomMuJoCoSeqDataset(
    #     dataset, horizon=20, discount=0.997, 
    #     stride=1, center_mapping=True,
    #     terminal_penalty=0,max_path_length=1016,
    #     full_traj_bonus=0,padding=16
    # )
    #     task_list = planner_dataset.task_list
        
    #     for dynamics_model_path in dynamics_models_list:
    #         model = SAC.load(
    #             dynamics_model_path+"best_model.zip",
    #             env=par_env,
    #             device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
    #             custom_objects={
    #                 "observation_space": par_env.observation_space,
    #                 "action_space": par_env.action_space
    #             }
    #         )
            
    #         task = task_list[task_index]
    #         print(f"Collecting data on task: {task} from model {dynamics_model_path}")
    #         par_env.set_task(np.tile(task, (args.now,1)))
    #         episode_collected = 0
    #         while True:
    #             obss = np.empty((args.now,0, par_env.observation_space.shape[0]))
    #             actions = np.empty((args.now,0, par_env.action_space.shape[0]))
    #             rewards = np.empty((args.now,0))
    #             terminations = np.empty((args.now,0))
    #             done = np.zeros(args.now, dtype=bool)
    #             obs = par_env.reset()
    #             cum_done = 0
    #             for _ in range(1000):

    #                 action = model.predict(obs, deterministic=True)[0]
    #                 obss = np.append(obss, obs[:, np.newaxis, :], axis=1)
    #                 actions = np.append(actions, action[:, np.newaxis, :], axis=1)
    #                 obs, reward, done, info = par_env.step(action)
    #                 cum_done = done if cum_done is None else np.logical_or(cum_done, done)

    #                 reward = reward * (1 - cum_done)
    #                 rewards = np.append(rewards, reward[:, np.newaxis], axis=1)

    #                 terminations = np.append(terminations,cum_done[:, np.newaxis], axis=1)

    #             accu_rewards = rewards.sum(axis = 1)
    #             print(accu_rewards)
    #             print(accu_rewards.mean())

    #             print(accu_rewards.std())
    #             # if accu_rewards.std() > accu_rewards.mean() * 0.1:
    #             #     print("High variance in rewards, skipping this batch")
    #             #     break
    #             # if accu_rewards.mean() < 3000:
    #             #     break
    #             for j in range(args.now):

    #                 if np.sum(rewards[j]) < accu_rewards.mean() - 3 * accu_rewards.std():
    #                     print(f"Episode {j} discarded due to low reward: {np.sum(rewards[j])}")
    #                     continue
    #                 if np.sum(terminations[j]) > 1:
    #                     print(f"Episode {j} discarded due to early stop")
    #                     continue
    #                 truncations = np.zeros(obss.shape[1])

    #                 truncations[-1] = 1
    #                 infos = {}
    #                 infos['task_index'] = task_index * np.ones(obss.shape[1])
    #                 infos['task'] = task
    #                 infos['ref_score'] = np.sum(rewards[j])
    #                 episodes.append(minari.data_collector.EpisodeBuffer(len(episodes),
    #                                                 observations=obss[j],
    #                                                 actions=actions[j],
    #                                                 rewards=rewards[j],
    #                                                 terminations = terminations[j],
    #                                                 truncations = truncations,
    #                                                 infos = infos))
    #                 episode_collected += 1
                    
    #                 if episode_collected >= args.data_collect_episode:
    #                     break
    #             print(f"Episode {len(episodes)} collected with {obss[j].shape[0]} timesteps")
    #             if episode_collected >= args.data_collect_episode:
    #                 task_index += 1
    #                 break


    #     dataset_id = f'{args.env[:-3]}/{task_num}dynamics-v2'
    #     dataset = minari.create_dataset_from_buffers(dataset_id,
    #                                                     buffer = episodes,
    #                                                     env = gymnasium.make(args.env),
    #                                                     )

    # elif args.mode == "train_and_data":
    #     episodes = []


    #     trained_task = []
    #     ref_max_scores = []
    #     ref_min_scores = []
    #     for i in range(task_num):
    #         task = task_list[i]
    #         # if i == 0:
    #         #     task = default_task
    #         # else:

    #         #     # task = env.unwrapped.get_random_task()
    #         #     task[0]-= 4

            
    #         model_path = base_path + f"{np.round(task,2)}/"
    #         if not os.path.exists(model_path):
    #             os.makedirs(model_path)
    #         trained_task.append(task)
    #         termination_callback = PerformanceBasedTermination(par_env, 
    #                                             eval_freq=args.eval_freq, 
    #                                             n_eval_episodes=args.num_eval_episodes, 
    #                                             patience=args.patience, 
    #                                             least_train_step = env.unwrapped.least_train_step if env.unwrapped.least_train_step is not None else 1000000,

    #                                             min_improvement=args.min_improvement,
    #                                             std_threshold=args.std_threshold, 
    #                                             model_save_path=model_path,target_reward=None)
    #         par_env.set_task(np.tile(task, (args.now,1)))
    #         print(f"Training on task: {task}")
    #         print(f"Current GPU: {torch.cuda.current_device()}")
    #         print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    #         print(f"GPU memory: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3:.1f} GB")
    #         model.learn(
    #                         total_timesteps=env.unwrapped.max_train_step if env.unwrapped.max_train_step is not None else 2000000,
    #                         log_interval=10,
    #                         callback=termination_callback,
    #                         tb_log_name=f"task_{i}",
    #                         )
    #         model.save(model_path+f"last_model.zip")
    #         #save arguments as .json
    #         import json
    #         with open(model_path+'args.json', 'w') as f:
    #             json.dump(vars(args), f, indent=4) 
            
    #         # clear the replay buffer
    #         model = SAC.load(model_path+f"best_model.zip", env=par_env, device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    #         print(f"Collecting data on task: {task}")
    #         ref_max_score = -np.inf
    #         ref_min_score = None
    #         episode_collected = 0
    #         while True:
    #             obss = np.empty((args.now,0, par_env.observation_space.shape[0]))
    #             actions = np.empty((args.now,0, par_env.action_space.shape[0]))
    #             rewards = np.empty((args.now,0))
    #             terminations = np.empty((args.now,0))
    #             done = np.zeros(args.now, dtype=bool)
    #             obs = par_env.reset()
    #             random_action = 0
    #             cum_done = 0
    #             for _ in range(1000):
                    
    #                 # if np.random.rand() < 0.02 and random_action == 0:
    #                 #     random_action = 4
    #                 # if random_action > 0:
    #                 #     action = np.random.uniform(low = -1,high = 1,size = (args.now,par_env.action_space.shape[0]))
    #                 #     random_action -= 1
    #                 # else:
    #                 action = model.predict(obs, deterministic=True)[0]
    #                 # action = model.predict(obs, deterministic=True)[0]
    #                 obss = np.append(obss, obs[:, np.newaxis, :], axis=1)
    #                 actions = np.append(actions, action[:, np.newaxis, :], axis=1)
    #                 obs, reward, done, info = par_env.step(action)
    #                 cum_done = done if cum_done is None else np.logical_or(cum_done, done)
    #                 reward = reward * (1 - cum_done)
    #                 rewards = np.append(rewards, reward[:, np.newaxis], axis=1)
    #                 terminations = np.append(terminations,cum_done[:, np.newaxis], axis=1)
    #             accu_rewards = rewards.sum(axis = 1)
    #             print(accu_rewards)
    #             print(accu_rewards.mean())

    #             print(accu_rewards.std())
    #             # if accu_rewards.std() > accu_rewards.mean() * 0.1:
    #             #     print("High variance in rewards, skipping this batch")
    #             #     break
    #             # if accu_rewards.mean() < 3000:
    #             #     break
    #             for j in range(args.now):

    #                 if np.sum(rewards[j]) < accu_rewards.mean() - 3 * accu_rewards.std():
    #                     print(f"Episode {j} discarded due to low reward: {np.sum(rewards[j])}")
    #                     continue
    #                 if np.sum(terminations[j]) > 1:
    #                     print(f"Episode {j} discarded due to early stop")
    #                     continue

    #                 ref_max_score = max(ref_max_score, np.sum(rewards[j]))
    #                 ref_min_score = min(ref_min_score, np.sum(rewards[j])) if ref_min_score is not None else np.sum(rewards[j])
    #                 truncations = np.zeros(obss.shape[1])

    #                 truncations[-1] = 1
    #                 infos = {}
    #                 infos['task_index'] = i * np.ones(obss.shape[1])
    #                 infos['task'] = task
    #                 infos['ref_score'] = np.sum(rewards[j])
                    
    #                 episodes.append(minari.data_collector.EpisodeBuffer(len(episodes),
    #                                                 observations=obss[j],
    #                                                 actions=actions[j],
    #                                                 rewards=rewards[j],
    #                                                 terminations = terminations[j],
    #                                                 truncations = truncations,
    #                                                  infos = infos))
    #                 episode_collected += 1
                    
    #                 if episode_collected >= args.data_collect_episode:
    #                     break
    #             print(f"Episode {len(episodes)} collected with {obss[j].shape[0]} timesteps")
    #             if episode_collected >= args.data_collect_episode:
    #                 ref_max_scores.append(ref_max_score)
    #                 ref_min_scores.append(ref_min_score)
    #                 break

    #     dataset_id = f'{args.env[:-3]}/{task_num}dynamics-v1'

    #     # delete the test dataset if it already exists
    #     # local_datasets = minari.list_local_datasets()
    #     # if dataset_id in local_datasets:
    #     #     minari.delete_dataset(dataset_id)
    #     dataset = minari.create_dataset_from_buffers(dataset_id,
    #                                                     buffer = episodes,
    #                                                     env = gymnasium.make(args.env),ref_max_score = ref_max_scores,
    #                                                     ref_min_score = ref_min_scores,
    #                                                     )


    #     #save the trained tasks
    #     np.save(base_path+"trained_tasks.npy", np.array(trained_task))
    #     #save the trained tasks
        




