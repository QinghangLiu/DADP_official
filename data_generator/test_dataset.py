import minari
from torch.utils.data import DataLoader
import torch
import numpy as np
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.dataset.dataset_utils import GaussianNormalizer
from cleandiffuser.utils import loop_dataloader
from cleandiffuser.nn_condition import MLPCondition
from cleandiffuser.dataset.base_dataset import BaseDataset
import imageio
import gym
import dr_envs
import os
import time
import argparse
def collate_fn(batch,segment_size = 8):
    i = np.random.randint(0, batch[0].observations.shape[0] - segment_size + 1)

    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations) for x in batch],
            batch_first=True
        )[:,i:i+segment_size,:],
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch],
            batch_first=True
        )[:,i:i+segment_size,:],
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch],
            batch_first=True
        )[:,i:i+segment_size],
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch],
            batch_first=True
        )[:,i:i+segment_size],
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch],
            batch_first=True
        )[:,i:i+segment_size]
    }


class GaussianNormalizer:
    """
        normalizes data to have zero mean and unit variance
    """

    def __init__(self, X):

        self.means, self.stds = X.mean(0), X.std(0)
        self.stds[self.stds == 0] = 1.

    def normalize(self, x):
        if self.means.device != x.device:
            self.means = self.means.to(x.device)
            self.stds = self.stds.to(x.device)
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        if self.means.device != x.device:
            self.means = self.means.to(x.device)
            self.stds = self.stds.to(x.device)
        return x * self.stds + self.means

# class SeqDataset(torch.utils.data.Dataset):
parser = argparse.ArgumentParser()
parser.add_argument('--algo',           default="sac", type=str, help='Algorithm to use')
parser.add_argument('--env',            default="RandomHalfCheetah-v0", type=str, help='Train gym env')
parser.add_argument('--lr',             default=None, type=float, help='Learning rate')
parser.add_argument('--gamma',          default=0.99, type=float, help='gamma discount factor')
parser.add_argument('--now',            default=16, type=int, help='Number of cpus for parallelization')
parser.add_argument('--gradient_steps', default=4, type=int, help='Number of gradient steps per update')
parser.add_argument('--model_path',     default="./best_model/best_model_heavy.zip", help='Train from scratch')
parser.add_argument('--mode',           default="data", help='Evaluate the policy')
parser.add_argument('--task',           default=np.array([2,7,7,7,7,7,7,1.8]), help='Use wandb for logging')
parser.add_argument('--video_save_path',default="./video", help='Path to save video')


dataset = minari.load_dataset('standard_RandomHalfCheetah-v0')

loader = DataLoader(dataset, batch_size=len(dataset), 
                    collate_fn=lambda x:collate_fn(x,segment_size = 500), 
                    shuffle= True,num_workers=8)
all_data = next(iter(loader))
print(all_data["observations"].shape, all_data["actions"].shape)

action_normalizer = GaussianNormalizer(
    all_data["actions"].reshape(-1, all_data["actions"].shape[-1])
)
observation_normalizer = GaussianNormalizer(
    all_data["observations"].reshape(-1, all_data["observations"].shape[-1])
)

act_dim = dataset.action_space.shape[0]
obs_dim = dataset.observation_space.shape[0]
print(f"Action dimension: {act_dim}, Observation dimension: {obs_dim}")
dataloader = DataLoader(dataset, batch_size=256, shuffle=True,collate_fn = collate_fn,)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


nn_diffusion = DiT1d(in_dim=act_dim+obs_dim, emb_dim=128,  timestep_emb_type="untrainable_fourier",)
nn_condition = MLPCondition(in_dim = 17,dropout=0.0,out_dim = 128,hidden_dims=256)
actor = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition,
        ema_rate=0.9999, device=device)

# n_gradient_steps = 0
# avg_loss = 0.
# actor.train()

# for batch in loop_dataloader(dataloader):
    
#     obs, act = batch["observations"].to(device), batch["actions"].to(device)
#     obs = observation_normalizer.normalize(obs)
#     act = action_normalizer.normalize(act)
#     # print(obs.mean((0,1)), obs.std((0,1)), act.mean((0,1)), act.std((0,1)))
#     # print(obs.shape, act.shape)
#     avg_loss += actor.update(x0=torch.concatenate((obs,act),dim = 2).float(),condition = obs[:,4,:].float())["loss"]
    
#     n_gradient_steps += 1

#     if n_gradient_steps % 100 == 0:
#         print(f'Step: {n_gradient_steps} | Loss: {avg_loss / 1000}')
#         avg_loss = 0.
    
#     if n_gradient_steps % 100_00 == 0:
#         actor.save("diffusion.pt")
    
#     if n_gradient_steps == 300_00:
#         break

actor.load("diffusion.pt")
actor.eval()
#inference
args = parser.parse_args()
env = gym.make(args.env)
obs = env.reset()
solver = "ddim"
sampling_step = 20
num_episodes = 3
frames = []
done = False
prior = torch.zeros((8, act_dim+obs_dim), device=device)
trajectory = torch.zeros((508,act_dim+obs_dim),device = device,dtype=torch.float32)
for i in range(500):  # 50 steps
    obs = torch.tensor(obs,device=device)
    obs = observation_normalizer.normalize(obs)
    # Expand obs to [batch_size, 1, obs_dim]
    prior = trajectory[max(i-4,0):max(i-4,0) + 8,:]
    prior[4,obs_dim:] = 0
    prior[5:,:] = 0
    seg, log = actor.sample(
        prior[None,], solver=solver, n_samples=1, sample_steps=sampling_step,
        sample_step_schedule="quad_continuous",
        w_cfg=1.0, condition_cfg=obs.float())
    seg = seg.squeeze(0)  # Remove batch dimension
    act = seg[min(i,4),obs_dim:]
    
    act = action_normalizer.unnormalize(act)
    act = act.cpu().numpy()
    act = act.squeeze()  # Ensure shape is (act_dim,)
    frame = env.render(mode='rgb_array')  # Get image frame
    frames.append(frame)

    trajectory[i,:] = seg[min(i,4),:]
    obs, reward, done, info = env.step(act)
    
video_path = f"{args.video_save_path}/{args.env}"
timestamp = time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(video_path):
    os.makedirs(video_path)
task_str = "_".join(map(str, np.array(args.task).flatten()))
# Save as GIF
imageio.mimsave(f"{video_path}/evaluation{timestamp}_task{task_str}.gif", frames, fps=20)