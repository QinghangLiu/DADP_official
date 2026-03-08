import torch
import argparse
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dadp.dadp import DADP
from dadp.train_utils import (
    get_dataset, 
    create_data_loaders_from_dataset,
    seed_everything,

)
from dadp.eval_utils import (
    extract_and_save_embeddings,
    load_embeddings_data,
    load_dataset_from_checkpoint_metadata,

)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Trained DADP Model")
    # Core arguments
    parser.add_argument("--checkpoint_path", type=str, 
                        # default="./dadp/embedding/logs/cleanup/40dynamics_v2/transformer/256_cross_01_01/checkpoints/epoch_0001_final.zip",
                        default="/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/exp/RandomHopper_82dynamics-v7/transformer/20251215_145213/best_model.zip",
                        
                        help="Path to checkpoint file")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default='cuda:1', help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Dataset override
    parser.add_argument("--dataset_name", type=str, 
                        default="RandomHopper/28dynamics-v0",
                        # default="RandomWalker2d/40dynamics-v2",
                        # default="RandomContinuousCartPoleHard/10dynamics-v1",
                        help="Override dataset name (if not provided, will use dataset from checkpoint metadata)")
    
    # Embedding extraction
    parser.add_argument("--extract_embeddings", action="store_true", default=True)
    parser.add_argument("--embeddings_path", type=str, 
                        default=None,
                        # default="/home/qinghang/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/transformer/exp_ant_28_reproduce/embeddings_data.npz",
                        # default="/home/qinghang/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/transformer/exp_walker_28(2)_dt1/embeddings_data.npz", # 
                        # default="/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/cleanup/transformer/20251015_231418/23.npz", # 
                        # default="/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/cleanup/transformer/20251015_235102/256.npz",
                        # default = "/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/cleanup/transformer/20251016_131954/embeddings_data.npz",
                        help="Path to precomputed embeddings (skip extraction if provided)")
    
    # Task filtering
    parser.add_argument("--task_ids", type=int, nargs="+", 
                        # default=None, 
                        # default=[0, 1, 2,  3,  4,  5,  6,  7,  8,  9, 
                        #         10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                        #         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                        #         30, 31, 32, 33, 34, 35, 36, 37, 38, 39,],
                        default=[i for i in range(28)],
                        help="Specific task IDs to evaluate")
    
    # Visualization
    parser.add_argument("--selected_tasks", type=int, nargs="+", default=None)
    parser.add_argument("--max_episodes_per_task", type=int, default=1)
    parser.add_argument("--max_points_per_task", type=int, default=5000)
    parser.add_argument("--plots", type=str, nargs="+", 
                        default=["3d_scatter", "distribution_summary", "distribution_spheres"],
                        choices=["3d_scatter", "distribution_spheres", "orthogonal_views", "distribution_summary"])
    
    return parser.parse_args()

def validate_device(device_str: str) -> str:
    """Validate device availability"""
    if device_str.startswith("cuda:"):
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU")
            return "cpu"
        
        device_id = int(device_str.split(":")[1])
        if device_id >= torch.cuda.device_count():
            print(f"Warning: CUDA device {device_id} not available, using cuda:0")
            return "cuda:0"
    
    print(f"Using device: {device_str}")
    return device_str

def load_checkpoint_and_dataset(checkpoint_path: str, device: str, dataset_name_override: Optional[str] = None):
    """Load checkpoint and reconstruct dataset"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model, metadata, state_dim, action_dim, policy_dataset, dataset_params = load_dataset_from_checkpoint_metadata(
        checkpoint_path, dataset_name_override=dataset_name_override
    )
    model = model.to(device)
    
    # Validate dimensions
    assert state_dim == model.dynamics_config.state_dim
    assert action_dim == model.dynamics_config.action_dim
    
    print(f"Model loaded - Epoch: {metadata.get('epoch', 'Unknown')}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Embedding size: {model.dynamics_config.embedding_config.embedding_size}")
    
    if dataset_name_override:
        print(f"Using override dataset: {dataset_name_override}")
    else:
        print(f"Using dataset from metadata: {dataset_params.get('dataset_name', 'Unknown')}")
    
    return model, metadata, policy_dataset, dataset_params

def handle_embeddings(model, policy_dataset, args, checkpoint_path, device):
    """Handle embedding extraction or loading"""
    if not args.extract_embeddings:
        return None
    
    print("\n" + "="*50)
    print("Processing Embeddings")
    print("="*50)
    
    # Check if embeddings path provided
    if args.embeddings_path and os.path.exists(args.embeddings_path):
        print(f"Loading existing embeddings: {args.embeddings_path}")
        try:
            embeddings_data = load_embeddings_data(args.embeddings_path)
            print(f"Loaded {len(embeddings_data['embeddings'])} embeddings")
            return args.embeddings_path
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            print("Will extract new embeddings...")
    
    # Extract new embeddings
    save_path = extract_and_save_embeddings(
        model=model,
        policy_dataset=policy_dataset,
        checkpoint_path=checkpoint_path,
        batch_size=args.batch_size,
        device=device
    )
    
    return save_path

def run_visualization(embeddings_path: str, args):
    """Run visualization analysis with improved error handling"""
    try:
        # Check if visualization module is available
        try:
            from dadp.visualization.vis_utils import check_dependencies
            if not check_dependencies():
                print("\nVisualization dependencies not available.")
                print("To enable visualization, install required packages:")
                print("  pip install plotly scikit-learn")
                return
        except ImportError:
            print("\nVisualization module not found.")
            print("Please ensure the visualization module is properly installed.")
            return
            
        # Import visualization functions
        from dadp.visualization.vis_utils import visualize_embeddings
        
        print("\n" + "="*50)
        print("Running Visualization Analysis")
        print("="*50)
        
        # Load and validate embeddings data first
        print(f"Loading embeddings from: {embeddings_path}")
        embeddings_data = load_embeddings_data(embeddings_path)
        
        # Debug information
        print(f"Embeddings shape: {embeddings_data['embeddings'].shape}")
        print(f"Task IDs shape: {embeddings_data['task_ids'].shape}")
        print(f"Task IDs type: {type(embeddings_data['task_ids'])}")
        print(f"Unique task IDs: {np.unique(embeddings_data['task_ids'])}")
        
        # Validate data consistency
        if len(embeddings_data['embeddings']) != len(embeddings_data['task_ids']):
            print(f"Error: Embeddings and task_ids length mismatch!")
            print(f"Embeddings: {len(embeddings_data['embeddings'])}, Task IDs: {len(embeddings_data['task_ids'])}")
            return
        
        # Create save directory
        if args.embeddings_path:
            save_dir = os.path.join(os.path.dirname(args.embeddings_path), "analysis")
        else:
            save_dir = os.path.join(os.path.dirname(args.checkpoint_path), "analysis")
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"Save directory: {save_dir}")
        
        # Filter plots
        vis_plots = [p for p in args.plots if p != 'distribution_summary']
        
        print(f"Will generate plots: {vis_plots}")
        
        results = visualize_embeddings(
            embeddings_data_path=embeddings_path,
            plots=vis_plots,
            selected_tasks=args.selected_tasks,
            max_episodes_per_task=args.max_episodes_per_task,
            max_points_per_task=args.max_points_per_task,
            save_dir=save_dir,
            plot_params={
                'show_points': True,
                'point_size': 3,
                'point_opacity': 0.7,
                'show_centers': True,
                'show_spheres': True,
                'sphere_opacity': 0.3,
                'center_size': 1
            }
        )
        
        # Check if visualization was successful
        if 'error' in results:
            print(f"Visualization failed: {results['error']}")
            return
        
        print(f"Analysis completed! Results saved to: {save_dir}")
        
        # List generated files
        if os.path.exists(save_dir):
            generated_files = [f for f in os.listdir(save_dir) if f.endswith('.html')]
            print(f"Generated HTML files:")
            for file in generated_files:
                full_path = os.path.join(save_dir, file)
                print(f"  - {full_path}")
        
        # Print summary
        if results.get('embedding_data'):
            embedding_data = results['embedding_data']
            print(f"Analyzed {embedding_data.num_tasks} tasks")
            print(f"Total embedding points: {len(embedding_data.all_embeddings)}")
        
        if results.get('explained_variance_ratio') is not None:
            variance_ratios = results['explained_variance_ratio']
            print(f"PCA - PC1: {variance_ratios[0]:.2%}, PC2: {variance_ratios[1]:.2%}, PC3: {variance_ratios[2]:.2%}")
        
        # Generate distribution summary if requested
        if "distribution_summary" in args.plots and results.get('embedding_data'):
            try:
                from dadp.visualization.vis_utils import create_distribution_summary_plot
                
                embedding_data = results['embedding_data']
                summary_path = os.path.join(save_dir, "distribution_summary.html")
                
                create_distribution_summary_plot(
                    embedding_data=embedding_data,
                    pca_data=results['pca_data'],
                    explained_variance_ratio=results['explained_variance_ratio'],
                    components=results['components'],
                    save_path=summary_path,
                    plot_title="Embedding Distribution Summary"
                )
                print(f"Distribution summary saved to: {summary_path}")
            except Exception as e:
                print(f"Failed to create distribution summary: {e}")
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print(f"\nError in visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Run visualization
    if args.embeddings_path:
        run_visualization(args.embeddings_path, args)
        return

    # Setup
    device = validate_device(args.device)
    seed_everything(args.seed)
    
    # Load model and dataset
    model, metadata, policy_dataset, dataset_params = load_checkpoint_and_dataset(
        args.checkpoint_path, device, dataset_name_override=args.dataset_name
    )
    
    # Create data loaders for evaluation
    from cleandiffuser.dataset.d4rl_mujoco_dataset import get_task_data
    policy_dataset = get_task_data(policy_dataset, np.array(args.task_ids)) if args.task_ids else policy_dataset
    print(f"Filtered dataset length: {len(policy_dataset)}")
    
    # Handle embeddings
    embeddings_path = handle_embeddings(
        model, policy_dataset, args, args.checkpoint_path, device
    )
    
    # Run visualization
    if embeddings_path:
        run_visualization(embeddings_path, args)

if __name__ == "__main__":
    main()
    # args = parse_args()
    # from dadp.visualization.vis_utils import create_single_task_visualization_with_additional_data_rollouts

    # additional_data_specs = [
    #     {
    #         "label": "Mixed DDIM",
    #         "path": "/home/qinghang/DomainAdaptiveDiffusionPolicy/results/exp_walker_28(2)_mixedddim_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/task_9_episode_1_embeddings.npy"
    #     },
    #     {
    #         "label": "DADP (Ours)",
    #         "path": "/home/qinghang/DomainAdaptiveDiffusionPolicy/results/exp_walker_28(2)_predict_mixddim_long_horizon_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.1_noisemixed_ddim/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/task_9_episode_3_embeddings.npy",
    #     },
    #     {
    #         "label": "Conditional",
    #         "path": "/home/qinghang/DomainAdaptiveDiffusionPolicy/results/exp_walker_28(2)_condition_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0_noisestandard/RandomWalker2d-v0/RandomWalker2d/28dynamics-v9/task_9_episode_3_embeddings.npy",
    #     },


    # ]

    # additional_datasets = []
    # for spec in additional_data_specs:
    #     data_path = spec["path"]
    #     if not os.path.exists(data_path):
    #         raise FileNotFoundError(f"Additional data file not found: {data_path}")
    #     data = np.load(data_path).astype(np.float64)
    #     additional_datasets.append({"label": spec["label"], "data": data})

    # fig = create_single_task_visualization_with_additional_data_rollouts(
    #     embeddings_data_path=args.embeddings_path,
    #     target_task_ids=list(range(3,28)),
    #     additional_data=additional_datasets,
    #     save_path="/home/qinghang/DomainAdaptiveDiffusionPolicy/visualizations/embedding_rollouts/exp_halfcheetah_task0_epi3_2d.html",
    #     max_points_per_dataset=1000,
    #     show_spheres=False,
    #     show_points=True,
    #     method="tsne",
    #     n_components=2
    # )

    # data_array_path = [
    #     './results/new model20251022_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.01_noiseembedding_guided/RandomWalker2d-v0/RandomWalker2d/40dynamics-v2/5planner_embeddings_with_denoise_history0.npy',
    #     './results/new model20251022_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.01_noiseembedding_guided/RandomWalker2d-v0/RandomWalker2d/40dynamics-v2/20planner_embeddings_with_denoise_history0.npy',
    #     './results/new model20251022_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.01_noiseembedding_guided/RandomWalker2d-v0/RandomWalker2d/40dynamics-v2/23planner_embeddings_with_denoise_history2.npy',
    #     './results/new model20251022_H20_Jump1_History16_next1_MCSS_transformer_d6_width256_joint_dp1_penalty0_bonus0_gamma0.997_adv1_weight2_guide0.01_noiseembedding_guided/RandomWalker2d-v0/RandomWalker2d/40dynamics-v2/32planner_embeddings_with_denoise_history0.npy',
    # ]
    # data_list = []
    # for path in data_array_path:
    #     data = np.load(path)  # Shape: (n_samples, embedding_dim)
    #     if data.shape[0] ==1000:
    #         data = data[16:]
    #     data = data.squeeze()
    #     data_list.append(data)

    # data_arr = np.stack(data_list, axis=0)  # Shape: (n_tasks, n_samples, embedding_dim)
    # print(f"big_arr shape: {data_arr.shape}")

    
    # from dadp.visualization.vis_utils import visualize_denoise_evolution_tsne_by_timestep
    # figs = visualize_denoise_evolution_tsne_by_timestep(
    #     task_data=data_arr[:,:,:,16:,:],  # shape [n_tasks, T, K, H, D]
    #     task_ids=list(range(data_arr.shape[0])),
    #     denoise_steps_to_plot=[0, 5, 10, 15, 20],
    #     tsne_params={"perplexity": 30, "n_iter": 1000, "learning_rate": 200, "random_state": 42},
    #     sample_timesteps_per_task=1500,
    #     save_dir="./visualizations/tsne_denoise_by_timestep3d_only_future_state",
    #     n_components=3,
    # )
    