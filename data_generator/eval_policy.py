import torch
import argparse
import os
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from PIL import Image
import imageio

from dadp.dadp import DADP
from dadp.train_utils import (
    get_dataset, 
    seed_everything,
)

from cleandiffuser.dataset.d4rl_mujoco_dataset import add_gaussian_noise

NOISE = 0.0


def parse_args():
    """Parse command line arguments for policy evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate Trained Policy Performance")
    
    # Required arguments
    parser.add_argument("--checkpoint_path", type=str, 
                        
                        # default = "/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/cleanup/40dynamics_v2/transformer/256_cross_01_01/best_model.zip",
                        default="/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/cleanup/RandomContinuousCartPoleHard_10dynamics-v1/transformer/20251017_131440/best_model.zip",

                        # default="/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/transformer/detach_policy/best_model.zip",
                        # default="/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/transformer/policy/best_model.zip",
                        # default="/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/transformer/20251008_174524/best_model.zip",
                        
                
                        
                        
                        
                        help="Path to the checkpoint zip file")
    
    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate per task")
    parser.add_argument("--max_episode_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--task_ids", type=int, nargs="+", default=[0], help="List of task IDs to evaluate on (if None, use all tasks)")
    parser.add_argument("--history_length", type=int, default=16, help="Length of history to use for embedding")
    
    # Visualization parameters
    parser.add_argument("--save_episode_frames", action="store_true", default=False, help="Save episode frames as PNG")
    parser.add_argument("--create_episode_gif", action="store_true", default=True, help="Create 2x5 grid GIF from 10 episodes")
    parser.add_argument("--frame_skip", type=int, default=4, help="Save every N-th frame (to reduce storage)")
    parser.add_argument("--gif_fps", type=int, default=30, help="FPS for the output GIF")
    
    # Other parameters
    parser.add_argument("--device", type=str, default='cuda:0', help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--render", action="store_true", help="Render environment during evaluation")
    parser.add_argument("--save_results", action="store_true", default=True, help="Save evaluation results")
    
    return parser.parse_args()

def load_observation_function_from_metadata(metadata):
    """
    Load observation function from checkpoint metadata
    
    Returns:
        tuple: (observation_function, observation_kwargs) or (None, None)
    """
    # Import observation functions
    from cleandiffuser.dataset.d4rl_mujoco_dataset import add_gaussian_noise, mask_dimensions
    
    observation_type = metadata.get('observation_type', None)
    if observation_type == 'gaussian_noise':
        observation_function = add_gaussian_noise
        noise_std = metadata.get('noise_std', 0.1)
        observation_kwargs = {"noise_std": noise_std}
        print(f"Recovered Gaussian noise observation with std={noise_std}")
        return observation_function, observation_kwargs
    elif observation_type == 'mask_dimensions':
        observation_function = mask_dimensions
        mask_dims = metadata.get('mask_dims', None)
        if mask_dims is not None:
            observation_kwargs = {"mask_dims": mask_dims}
            print(f"Recovered mask dimensions observation with mask_dims={mask_dims}")
            return observation_function, observation_kwargs
        else:
            print("Warning: mask_dimensions observation type found but no mask_dims specified")
            return None, None
    else:
        print("No observation function found in metadata")
        return None, None

class PolicyEvaluator:
    def __init__(self, model, policy_dataset, metadata, device="cpu", history_length=16):
        self.model = model
        self.policy_dataset = policy_dataset
        self.metadata = metadata
        self.device = device
        self.history_length = history_length
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Determine which normalizer and observation mode to use
        # use_observation = metadata.get('use_observation', False)
        use_observation = True
        
        if use_observation:
            # Check if observation normalizer is available
            observation_normalizer = policy_dataset.get_observation_normalizer()
            if (observation_normalizer is not None):
                self.normalizer = observation_normalizer
                self.use_observation_mode = True
                print("Using observation normalizer for transformed observations")
            else:
                print("Warning: use_observation=True but no observation normalizer found, using state normalizer")
                self.normalizer = policy_dataset.get_normalizer()
                self.use_observation_mode = False
        else:
            # Use state normalizer
            self.normalizer = policy_dataset.get_normalizer()
            self.use_observation_mode = False
            print("Using state normalizer for original states")
        
        # Persistent history for cross-episode state and action tracking
        self.persistent_history = None
        self.enable_persistent_history = True  # Configurable
        
        print(f"PolicyEvaluator initialized with history_length={history_length}")
        print(f"Normalizer type: {type(self.normalizer)}")
        print(f"Use observation mode: {self.use_observation_mode}")
    
    def normalize_observation(self, obs):
        """
        Normalize observation using the appropriate normalizer
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            normalized_obs: Normalized observation
        """
        return self.normalizer.normalize(obs.reshape(1, -1)).flatten()
    
    def get_action(self, state_history, action_history, current_obs):
        """
        Get action from policy given state and action history
        
        Args:
            state_history: numpy array of shape (history_length, state_dim)
            action_history: numpy array of shape (history_length, action_dim)
            
        Returns:
            action: numpy array of shape (action_dim,)
        """
        with torch.no_grad():
            # Convert to torch tensors and add batch dimension
            states_tensor = torch.FloatTensor(state_history).unsqueeze(0).to(self.device)  # (1, L, S)
            actions_tensor = torch.FloatTensor(action_history).unsqueeze(0).to(self.device)  # (1, L, A)
            
            # Get embedding
            embedding = self.model.dynamics.encode_history(states_tensor, actions_tensor)  # (1, embedding_size)
            
            # Predict action using policy head (with detached embedding)
            current_obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0).to(self.device)  # (1, S)
            embedding_detached = embedding.detach()
            pred_action = self.model.dynamics.predict_policy_action(embedding_detached, current_obs_tensor)  # (1, A)
            
            # Convert back to numpy
            action = pred_action.cpu().numpy().squeeze()  # (A,)

            # Online Factor Checking
            pred_factor = self.model.dynamics.pred_factor(embedding_detached).cpu().numpy().reshape(-1)  # (factor_dim,)
            print(f"Predicted factors: {pred_factor}")
            env_factor = self.policy_dataset.task_list[0] # (factor_dim,)
            print(f"Environment factors: {env_factor}")
            factor_diff = pred_factor - env_factor
            relative_diff = factor_diff / (env_factor + 1e-6) * 100
            print(f"Factor relative difference: {relative_diff}")

            # 修正action shape，确保是1D向量
            action = np.asarray(action)
            if action.ndim > 1:
                action = action.flatten()
            # 如果是标量，转为1D
            if action.shape == ():
                action = np.array([action], dtype=np.float32)
            return action

    def evaluate_task(self, task_id, num_episodes=10, max_episode_steps=1000, render=False,
                     save_frames=False, frames_dir=None, frame_skip=4):
        """
        Evaluate policy on a specific task
        
        Args:
            task_id: Task ID to evaluate
            num_episodes: Number of episodes to run
            max_episode_steps: Maximum steps per episode
            render: Whether to render the environment
            save_frames: Whether to save episode frames
            frames_dir: Directory to save frames
            frame_skip: Save every N-th frame
            
        Returns:
            results: Dictionary containing evaluation results
        """
        print(f"\nEvaluating Task {task_id}")
        
        # Recover environment for this task
        try:
            env = self.policy_dataset.recover_environment(task_id)
            print(f"Environment recovered for task {task_id}")
        except Exception as e:
            print(f"Failed to recover environment for task {task_id}: {e}")
            return None
        
        # Get reference scores if available
        try:
            ref_min_score, ref_max_score = self.policy_dataset.get_ref_score(task_id)
            print(f"Reference scores: min={ref_min_score:.2f}, max={ref_max_score:.2f}")
        except Exception as e:
            print(f"Could not get reference scores: {e}")
            ref_min_score, ref_max_score = None, None
        
        # Initialize results storage
        episode_returns = []
        episode_lengths = []
        success_episodes = 0
        episode_frames_list = []  # Store frames for each episode
        
        # Reset persistent history at the start of the task
        if self.enable_persistent_history:
            self.persistent_history = {
                'states': np.zeros((self.history_length, env.observation_space.shape[0])),
                'actions': np.zeros((self.history_length, env.action_space.shape[0]))
            }
        
        for episode in tqdm(range(num_episodes), desc=f"Task {task_id}"):
            # Reset environment
            obs = env.reset()
            
            # Normalize observation using appropriate normalizer
            obs = add_gaussian_noise(obs, noise_std=NOISE)  # Add noise if needed
            obs_normalized = self.normalize_observation(obs)
            
            # Use persistent history to initialize state and action history
            if self.enable_persistent_history and self.persistent_history is not None:
                state_history = self.persistent_history['states'].copy()
                action_history = self.persistent_history['actions'].copy()
            else:
                # Original zero initialization
                state_history = np.zeros((self.history_length, len(obs_normalized)))
                action_history = np.zeros((self.history_length, env.action_space.shape[0]))
            
            episode_return = 0.0
            episode_length = 0
            episode_frames = []  # Frames for this episode
            
            for step in range(max_episode_steps):
                # Get action from policy
                action = self.get_action(state_history, action_history, obs_normalized)
                # 修正action shape，确保与env.action_space一致
                action = np.asarray(action, dtype=np.float32)
                if hasattr(env.action_space, "shape") and env.action_space.shape != () and action.shape != env.action_space.shape:
                    action = action.reshape(env.action_space.shape)
                elif hasattr(env.action_space, "shape") and env.action_space.shape == () and action.shape != ():
                    action = np.array(action.item(), dtype=np.float32)
                
                # Capture frame if requested
                if save_frames and (step % frame_skip == 0):
                    try:
                        frame = env.unwrapped.render(mode='rgb_array')
                        if frame is not None:
                            episode_frames.append(frame)
                    except Exception as e:
                        print(f"Warning: Failed to capture frame at step {step}: {e}")
                
                # Execute action in environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                if render:
                    try:
                        env.render()
                    except:
                        pass  # Ignore render errors
                
                # Update episode statistics
                episode_return += reward
                episode_length += 1
                
                # Normalize next observation using appropriate normalizer
                next_obs = add_gaussian_noise(next_obs, noise_std=NOISE)  # Add noise if needed
                next_obs_normalized = self.normalize_observation(next_obs)
                
                obs_normalized = next_obs_normalized
                
                # Update history (shift and append)
                state_history[:-1] = state_history[1:]
                state_history[-1] = obs_normalized
                action_history[:-1] = action_history[1:]
                action_history[-1] = action
                
                # Check if episode is done
                if terminated or truncated:
                    if episode_length == max_episode_steps:
                        success_episodes += 1
                    break
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            # Store frames for this episode
            if save_frames and episode_frames:
                episode_frames_list.append(episode_frames)
                
                # Save individual episode frames if frames_dir provided
                if frames_dir is not None:
                    episode_dir = os.path.join(frames_dir, f"task_{task_id}_episode_{episode:02d}")
                    os.makedirs(episode_dir, exist_ok=True)
                    
                    for frame_idx, frame in enumerate(episode_frames):
                        frame_path = os.path.join(episode_dir, f"frame_{frame_idx:04d}.png")
                        Image.fromarray(frame).save(frame_path)
                    
                    print(f"Saved {len(episode_frames)} frames for episode {episode}")
            
            # Save the last history for the next episode
            if self.enable_persistent_history:
                self.persistent_history['states'] = state_history.copy()
                self.persistent_history['actions'] = action_history.copy()
        
        # Calculate statistics
        results = {
            'task_id': task_id,
            'num_episodes': num_episodes,
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'min_return': np.min(episode_returns),
            'max_return': np.max(episode_returns),
            'mean_length': np.mean(episode_lengths),
            'success_rate': success_episodes / num_episodes,
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths,
            'ref_min_score': ref_min_score,
            'ref_max_score': ref_max_score,
            'episode_frames': episode_frames_list if save_frames else None
        }
        
        # Calculate normalized score if reference scores are available
        if ref_min_score is not None and ref_max_score is not None:
            if ref_max_score != ref_min_score:
                normalized_score = (results['mean_return'] - ref_min_score) / (ref_max_score - ref_min_score)
                results['normalized_score'] = normalized_score
            else:
                results['normalized_score'] = 1.0 if results['mean_return'] >= ref_max_score else 0.0
        
        # Print results
        print(f"Task {task_id} Results:")
        print(f"  Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
        print(f"  Range: [{results['min_return']:.2f}, {results['max_return']:.2f}]")
        print(f"  Mean Length: {results['mean_length']:.1f}")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        if 'normalized_score' in results:
            print(f"  Normalized Score: {results['normalized_score']:.3f}")
        
        return results

def create_episode_grid_gif(episode_frames_list, output_path, fps=30, grid_size=(2, 5)):
    """
    Create a grid GIF from multiple episodes running simultaneously
    
    Args:
        episode_frames_list: List of frame lists for each episode (up to 10 episodes)
        output_path: Path to save the output GIF
        fps: Frames per second for the GIF
        grid_size: Grid layout (rows, cols)
    """
    if not episode_frames_list:
        print("No frames to create GIF")
        return
    
    num_episodes = min(len(episode_frames_list), grid_size[0] * grid_size[1])
    if num_episodes < grid_size[0] * grid_size[1]:
        print(f"Warning: Only {num_episodes} episodes available, filling remaining grid positions with blank")
    
    # Find the MAXIMUM number of frames across all episodes (changed from minimum)
    max_frames = max(len(frames) for frames in episode_frames_list[:num_episodes])
    
    if max_frames == 0:
        print("No frames available to create GIF")
        return
    
    print(f"Creating grid GIF with {num_episodes} episodes, {max_frames} frames (using max length)")
    
    # Print episode lengths for debugging
    for i, frames in enumerate(episode_frames_list[:num_episodes]):
        print(f"  Episode {i+1}: {len(frames)} frames" + 
              (f" (will pad with {max_frames - len(frames)} frames)" if len(frames) < max_frames else ""))
    
    # Get frame dimensions from first episode
    frame_height, frame_width = episode_frames_list[0][0].shape[:2]
    
    # Create grid frames
    grid_frames = []
    
    for frame_idx in tqdm(range(max_frames), desc="Creating grid frames"):
        # Create blank grid image
        grid_height = frame_height * grid_size[0]
        grid_width = frame_width * grid_size[1]
        grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Fill grid with episode frames
        for episode_idx in range(num_episodes):
            row = episode_idx // grid_size[1]
            col = episode_idx % grid_size[1]
            
            # Get frame from this episode
            # If episode has ended, use the last frame to pad
            episode_frame_list = episode_frames_list[episode_idx]
            if frame_idx < len(episode_frame_list):
                # Use current frame
                episode_frame = episode_frame_list[frame_idx]
            else:
                # Episode has ended, use last frame for padding
                episode_frame = episode_frame_list[-1]
            
            # Place frame in grid
            y_start = row * frame_height
            y_end = y_start + frame_height
            x_start = col * frame_width
            x_end = x_start + frame_width
            
            grid_frame[y_start:y_end, x_start:x_end] = episode_frame
            
            # Add episode number text and status indicator
            from PIL import ImageDraw, ImageFont
            pil_frame = Image.fromarray(grid_frame)
            draw = ImageDraw.Draw(pil_frame)
            
            # Add episode label
            text = f"Ep {episode_idx + 1}"
            
            # Add padding indicator if episode has ended
            if frame_idx >= len(episode_frame_list):
                text += " (ended)"
            
            text_position = (x_start + 5, y_start + 5)
            
            try:
                # Try to use a nice font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            # Draw text with background
            bbox = draw.textbbox(text_position, text, font=font)
            
            # Change background color if episode has ended (semi-transparent red)
            if frame_idx >= len(episode_frame_list):
                bg_color = (200, 100, 100, 180)  # Reddish to indicate padding
            else:
                bg_color = (0, 0, 0, 128)  # Black for active episodes
            
            draw.rectangle(bbox, fill=bg_color)
            draw.text(text_position, text, fill=(255, 255, 255), font=font)
            
            grid_frame = np.array(pil_frame)
        
        grid_frames.append(grid_frame)
    
    # Save as GIF
    imageio.mimsave(output_path, grid_frames, fps=fps)
    print(f"Grid GIF saved to: {output_path}")
    print(f"GIF info: {len(grid_frames)} frames, {grid_size[0]}x{grid_size[1]} grid, {fps} FPS")
    print(f"Note: Shorter episodes are padded with their last frame")

def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Set device
    device_str = args.device
    if device_str.startswith("cuda:"):
        device_id = int(device_str.split(":")[1])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device_id >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device_id} is not available. Available devices: 0-{torch.cuda.device_count()-1}")
    print(f"Using device: {device_str}")
    
    # Set random seed
    seed_everything(args.seed)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    
    # === 新的policy loading逻辑 ===
    # 只加载模型和metadata，然后用get_dataset重新构造policy_dataset
    model, metadata = DADP.load_checkpoint(args.checkpoint_path, device_str)
    print("Checkpoint loaded.")
    
    # 恢复dataset相关参数
    dataset_name = metadata.get(
        'dataset_name', 
        # 'RandomWalker2d/40dynamics-v2',
        "RandomContinuousCartPoleHard/10dynamics-v1"
        )
    history = metadata.get('history', 16)
    window_size = metadata.get('window_size', 2)
    delta_t = metadata.get('delta_t', 1)
    BIAS = 2
    horizon = history + window_size + BIAS + delta_t - 1 - 1
    observation_type = metadata.get('observation_function', None)
    use_observation = metadata.get('use_observation', False)
    observation_kwargs = metadata.get('observation_kwargs', None)
    # 恢复observation_function
    if observation_type == 'gaussian_noise':
        observation_function = add_gaussian_noise
    elif observation_type == 'mask_dimensions':
        from cleandiffuser.dataset.d4rl_mujoco_dataset import mask_dimensions
        observation_function = mask_dimensions
    else:
        observation_function = None

    print(f"\nObservation Configuration (recovered from checkpoint):")
    print(f"  Observation function: {observation_type}")
    print(f"  Use observation: {use_observation}")
    print(f"  Observation kwargs: {observation_kwargs}")

    # 重新构造policy_dataset
    state_dim, action_dim, policy_dataset = get_dataset(
        dataset_name=dataset_name,
        horizon=horizon,
        observation_function=observation_function,
        observation_kwargs=observation_kwargs,
        history=history
    )

    print(f"\nModel Configuration:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Encoder type: {model.dynamics_config.embedding_config.encoder_type}")
    print(f"  Embedding size: {model.dynamics_config.embedding_config.embedding_size}")

    # Create policy evaluator with metadata for observation handling
    evaluator = PolicyEvaluator(
        model=model,
        policy_dataset=policy_dataset,
        metadata=metadata,
        device=device_str,
        history_length=history
    )

    # Determine which tasks to evaluate
    if args.task_ids is not None:
        task_ids_to_eval = args.task_ids
        print(f"Evaluating specified tasks: {task_ids_to_eval}")
    else:
        if hasattr(policy_dataset, 'task_list'):
            task_ids_to_eval = list(range(len(policy_dataset.task_list)))
            print(f"Evaluating all {len(task_ids_to_eval)} tasks")
        else:
            task_ids_to_eval = [0]
            print("No task information found, evaluating task 0 only")

    frames_base_dir = None
    if args.save_episode_frames or args.create_episode_gif:
        results_dir = os.path.dirname(args.checkpoint_path)
        frames_base_dir = os.path.join(results_dir, "episode_visualizations")
        os.makedirs(frames_base_dir, exist_ok=True)
        print(f"Episode frames will be saved to: {frames_base_dir}")

    all_results = []
    for task_id in task_ids_to_eval:
        try:
            task_frames_dir = None
            if frames_base_dir is not None:
                task_frames_dir = os.path.join(frames_base_dir, f"task_{task_id}")
                os.makedirs(task_frames_dir, exist_ok=True)
            
            results = evaluator.evaluate_task(
                task_id=task_id,
                num_episodes=args.num_episodes,
                max_episode_steps=args.max_episode_steps,
                render=args.render,
                save_frames=args.save_episode_frames or args.create_episode_gif,
                frames_dir=task_frames_dir if args.save_episode_frames else None,
                frame_skip=args.frame_skip
            )
            
            if results is not None:
                all_results.append(results)
                # 总是保存GIF（只要有帧）
                if results.get('episode_frames'):
                    gif_path = os.path.join(task_frames_dir, f"task_{task_id}_episodes_grid.gif")
                    create_episode_grid_gif(
                        episode_frames_list=results['episode_frames'],
                        output_path=gif_path,
                        fps=args.gif_fps,
                        grid_size=(2, 5)
                    )
        except Exception as e:
            print(f"Error evaluating task {task_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print overall summary
    if all_results:
        print("\n" + "="*60)
        print("OVERALL EVALUATION SUMMARY")
        print("="*60)
        
        all_mean_returns = [r['mean_return'] for r in all_results]
        all_success_rates = [r['success_rate'] for r in all_results]
        all_normalized_scores = [r['normalized_score'] for r in all_results if 'normalized_score' in r]
        
        print(f"Evaluated {len(all_results)} tasks with {args.num_episodes} episodes each")
        print(f"Overall Mean Return: {np.mean(all_mean_returns):.2f} ± {np.std(all_mean_returns):.2f}")
        print(f"Overall Success Rate: {np.mean(all_success_rates):.2%} ± {np.std(all_success_rates):.2%}")
        
        if all_normalized_scores:
            print(f"Overall Normalized Score: {np.mean(all_normalized_scores):.3f} ± {np.std(all_normalized_scores):.3f}")
        
        print(f"Return Range: [{np.min(all_mean_returns):.2f}, {np.max(all_mean_returns):.2f}]")
        
        # Save results if requested
        if args.save_results:
            results_dir = os.path.dirname(args.checkpoint_path)
            results_path = os.path.join(results_dir, "policy_evaluation_results.json")
            
            import json
            
            # Remove frames from results before saving (too large for JSON)
            save_results = []
            for r in all_results:
                r_copy = r.copy()
                r_copy.pop('episode_frames', None)
                save_results.append(r_copy)
            
            save_data = {
                'checkpoint_path': args.checkpoint_path,
                'evaluation_args': vars(args),
                'task_results': save_results,
                'summary': {
                    'num_tasks': len(all_results),
                    'overall_mean_return': float(np.mean(all_mean_returns)),
                    'overall_std_return': float(np.std(all_mean_returns)),
                    'overall_success_rate': float(np.mean(all_success_rates)),
                    'overall_normalized_score': float(np.mean(all_normalized_scores)) if all_normalized_scores else None,
                }
            }
            
            with open(results_path, 'w') as f:
                json.dump(save_data, f, indent=4)
            
            print(f"\nResults saved to: {results_path}")
    else:
        print("No successful evaluations completed.")

if __name__ == "__main__":
    main()