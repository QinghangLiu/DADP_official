# dadp.py
# DADP: Dynamics-Adaptive Diffusion Policy (unified wrapper)
# Integrates: SequenceEncoder (encoder.py), LightweightDynamics (dynamics.py), DiffusionPlanner (diffusion_planner.py)
# Provides: training losses (diffusion + inverse/forward), one-step train_step, and inference (DDIM sampling + inverse)
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .dynahead import DynamicsHead

import zipfile
import json
import uuid

BIAS: int = 2

@dataclass
class EmbeddingConfig:
    """Configuration for Transformer sequence-to-embedding model"""
    embedding_size: int = 256
    norm_z: bool = True
    mask_embedding: bool = False
    pooling:str = "adaptive"
    d_state: int = 0
    d_conv: int = 0
    expand: int = 0
    bidirectional: bool = False
    nonlinearity: str = "relu"
    # Transformer parameters
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0
    pos_encoding_max_len: int = 5000
    adaptive_pooling_heads: int = 8
    adaptive_pooling_dropout: float = 0.1
    encoder_type: str = "transformer"  # Only 'transformer' supported for now
    def to_encoder_kwargs(self) -> dict:
        """Convert config to transformer encoder kwargs"""
        return {
            "embedding_size": self.embedding_size,
            "norm_z": self.norm_z,
            "mask_embedding": self.mask_embedding,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "rope_theta": self.rope_theta,
            "pos_encoding_max_len": self.pos_encoding_max_len,
            "adaptive_pooling_heads": self.adaptive_pooling_heads,
            "adaptive_pooling_dropout": self.adaptive_pooling_dropout,
        }

@dataclass 
class DynamicsConfig:
    """Configuration for DynamicsHead"""
    state_dim: int
    action_dim: int
    factor_dim: int = 0
    head_hidden: int = 256
    embedding_config: EmbeddingConfig = None
    
    def __post_init__(self):
        if self.embedding_config is None:
            self.embedding_config = EmbeddingConfig()

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Loss weights
    inverse_loss_weight: float = 1.0
    forward_loss_weight: float = 1.0
    intra_traj_consistency_loss_weight: float = 0.1
    inter_traj_consistency_loss_weight: float = 0.1
    state_loss_weight: float = 0.5
    factor_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0  # Add policy loss weight
    inter_trajectory_loss_weight: float = 0.0  # For future use
    embedding_forward_loss_weight: float = 0.0  # For future use
    consistency_window_size: float = 0.0  # For future use
    consistency_loss_weight: float = 0.0  # For future use
    # Sequence processing parameters  
    history: int = 10
    window_size: int = 1
    delta_t: int = 1
    reward_loss_weight: float = 1.0
    # Training parameters
    learning_rate: float = 3e-4
    num_epochs: int = 10
    batch_size: int = 32
    eval_interval: int = 1
    device: str = "cuda"

    # Observation parameters
    use_observation: bool = False
    
    # Cross-prediction option
    cross_prediction: bool = False
    
    # Embedding detach parameters
    detach_embedding_for_factor: bool = False
    detach_embedding_for_state: bool = False
    detach_embedding_for_policy: bool = False
    detach_embedding_for_reward: bool = False
    # Attention mask parameters
    min_visible_length: int = 2  # When equal to history, no masking is applied

class DADP(nn.Module):
    """Domain Adaptive Diffusion Policy with Transformer encoder"""
    
    def __init__(self, dynamics_config: DynamicsConfig, training_config: TrainingConfig):
        super().__init__()
        self.dynamics_config = dynamics_config
        self.training_config = training_config

        self.dynamics = DynamicsHead(
            state_dim=dynamics_config.state_dim,
            action_dim=dynamics_config.action_dim,
            factor_dim=dynamics_config.factor_dim,
            embedding_size=dynamics_config.embedding_config.embedding_size,
            head_hidden=dynamics_config.head_hidden,
            encoder_kwargs=dynamics_config.embedding_config.to_encoder_kwargs()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=training_config.learning_rate)

    def _extract_sub_batches(self, states, actions):
        """Extract sliding windows from sequences - now handles single batch"""
        # Single batch: [B, L, S]
        B, L, S = states.shape
        A = actions.shape[-1]
        
        history = self.training_config.history
        required_length = history + BIAS + self.training_config.delta_t - 1
        
        sub_states = torch.zeros(B * self.training_config.window_size, required_length, S, device=states.device, dtype=states.dtype)
        sub_actions = torch.zeros(B * self.training_config.window_size, required_length, A, device=actions.device, dtype=actions.dtype)
        
        idx = 0
        for batch_idx in range(B):
            for window_idx in range(self.training_config.window_size):
                start_pos = window_idx
                end_pos = start_pos + required_length
                
                sub_states[idx] = states[batch_idx, start_pos:end_pos, :]
                sub_actions[idx] = actions[batch_idx, start_pos:end_pos, :]
                idx += 1
        
        return sub_states, sub_actions

    def _extract_task_factors(self, batch, original_batch_size, current_sub_batch_size, device):
        task_list = batch.get("task_list", None)
        if task_list is not None:
            target_factors = torch.zeros((current_sub_batch_size, self.dynamics_config.factor_dim), device=device)
            idx = 0
            
            for batch_idx in range(original_batch_size):
                task_factor = task_list[batch_idx]
                for _ in range(self.training_config.window_size):
                    target_factors[idx] = task_factor
                    idx += 1
            target_factors = target_factors.to(torch.float32).to(device)
            return target_factors
        return None

    def _process_single_batch(self, batch):
        """Process a single batch and return processed data for loss computation"""
        device = next(self.parameters()).device
        
        # Choose input data
        if self.training_config.use_observation and "observation" in batch["obs"]:
            states = batch["obs"]["observation"].to(device)
        else:
            states = batch["obs"]["state"].to(device)

        actions = batch["act"].to(device)
        
        sub_states, sub_actions = self._extract_sub_batches(states, actions)
        
        if sub_states is None or sub_actions is None:
            return None

        current_sub_batch_size = sub_states.shape[0]
        original_batch_size = states.shape[0]

        # Extract data components
        state_history = sub_states[:, :self.training_config.history, :]
        action_history = sub_actions[:, :self.training_config.history, :]
        history_followed_state = sub_states[:, self.training_config.history, :]
        current_state = sub_states[:, -2, :]
        next_state = sub_states[:, -1, :]
        current_action = sub_actions[:, -2, :]

        # Get embeddings with attention mask based on min_visible_length
        use_random_mask = self.training_config.min_visible_length < self.training_config.history
        embedding = self.dynamics.encode_history(
            state_history, action_history,
            use_random_mask=use_random_mask,
            min_visible_length=self.training_config.min_visible_length
        )

        # Extract task factors if available
        target_factors = self._extract_task_factors(batch, original_batch_size, current_sub_batch_size, device)
        
        return {
            'embedding': embedding,
            'state_history': state_history,
            'action_history': action_history, 
            'history_followed_state': history_followed_state,
            'current_state': current_state,
            'next_state': next_state,
            'current_action': current_action,
            'target_factors': target_factors,
            'original_batch_size': original_batch_size,
            'current_sub_batch_size': current_sub_batch_size
        }

    def process_batch(self, paired_batch):
        """Process paired batch and return all necessary data for loss computation.
        Input: paired_batch with shape [batch_size, 2, seq_len, ...]
        Output: processed_batch1, processed_batch2 with shape [batch_size, seq_len, ...]
        """
        device = next(self.parameters()).device

        batch1 = {}
        batch2 = {}

        for key, value in paired_batch.items():
            if isinstance(value, dict):
                batch1[key] = {}
                batch2[key] = {}
                for sub_key, sub_value in value.items():
                    # sub_value: [batch_size, 2, seq_len, ...]
                    batch1[key][sub_key] = sub_value[:, 0, ...]  # [batch_size, seq_len, ...]
                    batch2[key][sub_key] = sub_value[:, 1, ...]
            else:
                # value: [batch_size, 2, ...]
                batch1[key] = value[:, 0, ...]
                batch2[key] = value[:, 1, ...]

        processed_batch1 = self._process_single_batch(batch1)
        processed_batch2 = self._process_single_batch(batch2)

        if processed_batch1 is None or processed_batch2 is None:
            return None

        return processed_batch1, processed_batch2

    def train_step(self, paired_batch):
        """Execute one training step - focuses purely on loss computation"""
        device = next(self.parameters()).device
        
        # Process the paired batch to get all necessary data
        processed_batch1, processed_batch2 = self.process_batch(paired_batch)

        # Extract embeddings
        embedding1 = processed_batch1['embedding']
        embedding2 = processed_batch2['embedding']

        losses1 = {}
        losses2 = {}
        
        ##### Embedding Learning Losses #####
        
        if self.training_config.cross_prediction:
            # Cross-prediction mode: use embedding1 to predict batch2's dynamics and vice versa

            # Use embedding1 to predict batch2's dynamics
            pred_next_state1 = self.dynamics.predict_next_state(embedding1, processed_batch2['current_state'], processed_batch2['current_action'])
            losses1['forward'] = DynamicsHead.forward_loss(pred_next_state1, processed_batch2['next_state'])
            
            pred_action1 = self.dynamics.predict_action(embedding1, processed_batch2['current_state'], processed_batch2['next_state'])
            losses1['inverse'] = DynamicsHead.inverse_loss(pred_action1, processed_batch2['current_action'])
            
            # Use embedding2 to predict batch1's dynamics
            pred_next_state2 = self.dynamics.predict_next_state(embedding2, processed_batch1['current_state'], processed_batch1['current_action'])
            losses2['forward'] = DynamicsHead.forward_loss(pred_next_state2, processed_batch1['next_state'])
            
            pred_action2 = self.dynamics.predict_action(embedding2, processed_batch1['current_state'], processed_batch1['next_state'])
            losses2['inverse'] = DynamicsHead.inverse_loss(pred_action2, processed_batch1['current_action'])
        else:
            # Normal mode: each embedding predicts its own batch's dynamics
            # Forward loss 1
            pred_next_state1 = self.dynamics.predict_next_state(embedding1, processed_batch1['current_state'], processed_batch1['current_action'])
            losses1['forward'] = DynamicsHead.forward_loss(pred_next_state1, processed_batch1['next_state'])

            # Inverse loss 1
            pred_action1 = self.dynamics.predict_action(embedding1, processed_batch1['current_state'], processed_batch1['next_state'])
            losses1['inverse'] = DynamicsHead.inverse_loss(pred_action1, processed_batch1['current_action'])

            # Forward loss 2
            pred_next_state2 = self.dynamics.predict_next_state(embedding2, processed_batch2['current_state'], processed_batch2['current_action'])
            losses2['forward'] = DynamicsHead.forward_loss(pred_next_state2, processed_batch2['next_state'])

            # Inverse loss 2
            pred_action2 = self.dynamics.predict_action(embedding2, processed_batch2['current_state'], processed_batch2['next_state'])
            losses2['inverse'] = DynamicsHead.inverse_loss(pred_action2, processed_batch2['current_action'])

        # Policy loss 1
        policy_embedding1 = embedding1.detach() if self.training_config.detach_embedding_for_policy else embedding1
        pred_policy_action1 = self.dynamics.predict_policy_action(policy_embedding1, processed_batch1['current_state'])
        losses1['policy'] = DynamicsHead.policy_loss(pred_policy_action1, processed_batch1['current_action'])

        # State loss 1
        state_embedding1 = embedding1.detach() if self.training_config.detach_embedding_for_state else embedding1
        pred_state1 = self.dynamics.predict_state_from_embedding(state_embedding1)
        losses1['state'] = DynamicsHead.state_loss(pred_state1, processed_batch1['history_followed_state'])

        # Factor loss 1
        losses1['factor'] = torch.tensor(0.0, device=device, requires_grad=True)
        if self.dynamics_config.factor_dim > 0 and processed_batch1['target_factors'] is not None:
            factor_embedding1 = embedding1.detach() if self.training_config.detach_embedding_for_factor else embedding1
            pred_factors1 = self.dynamics.pred_factor(factor_embedding1)
            losses1['factor'] = DynamicsHead.factor_loss(pred_factors1, processed_batch1['target_factors'])

        # Intra-trajectory consistency loss 1
        losses1['intra_traj_consistency'] = torch.tensor(0.0, device=device, requires_grad=True)
        if self.training_config.window_size > 1:
            embedding1_reshaped = embedding1.view(processed_batch1['original_batch_size'], self.training_config.window_size, -1)
            losses1['intra_traj_consistency'] = DynamicsHead.intra_traj_consistency_loss(embedding1_reshaped)
        
        # Policy loss 2
        policy_embedding2 = embedding2.detach() if self.training_config.detach_embedding_for_policy else embedding2
        pred_policy_action2 = self.dynamics.predict_policy_action(policy_embedding2, processed_batch2['current_state'])
        losses2['policy'] = DynamicsHead.policy_loss(pred_policy_action2, processed_batch2['current_action'])

        # State loss 2
        state_embedding2 = embedding2.detach() if self.training_config.detach_embedding_for_state else embedding2
        pred_state2 = self.dynamics.predict_state_from_embedding(state_embedding2)
        losses2['state'] = DynamicsHead.state_loss(pred_state2, processed_batch2['history_followed_state'])

        # Factor loss 2
        losses2['factor'] = torch.tensor(0.0, device=device, requires_grad=True)
        if self.dynamics_config.factor_dim > 0 and processed_batch2['target_factors'] is not None:
            factor_embedding2 = embedding2.detach() if self.training_config.detach_embedding_for_factor else embedding2
            pred_factors2 = self.dynamics.pred_factor(factor_embedding2)
            losses2['factor'] = DynamicsHead.factor_loss(pred_factors2, processed_batch2['target_factors'])

        # Intra-trajectory consistency loss 2
        losses2['intra_traj_consistency'] = torch.tensor(0.0, device=device, requires_grad=True)
        if self.training_config.window_size > 1:
            embedding2_reshaped = embedding2.view(processed_batch2['original_batch_size'], self.training_config.window_size, -1)
            losses2['intra_traj_consistency'] = DynamicsHead.intra_traj_consistency_loss(embedding2_reshaped)

        # Combine losses from both batches
        combined_losses = {}
        for key in losses1.keys():
            combined_losses[key] = (losses1[key] + losses2[key]) / 2.0
        
        # Inter-trajectory consistency loss
        combined_losses['inter_traj_consistency'] = DynamicsHead.inter_traj_consistency_loss(embedding1, embedding2)
        
        # Total loss with separate consistency weights
        total_loss = (
            self.training_config.inverse_loss_weight * combined_losses['inverse'] +
            self.training_config.forward_loss_weight * combined_losses['forward'] +
            self.training_config.policy_loss_weight * combined_losses['policy'] +
            self.training_config.intra_traj_consistency_loss_weight * combined_losses['intra_traj_consistency'] +
            self.training_config.inter_traj_consistency_loss_weight * combined_losses['inter_traj_consistency'] +
            self.training_config.state_loss_weight * combined_losses['state'] +
            self.training_config.factor_loss_weight * combined_losses['factor']
        )

        return {
            'total_loss': total_loss,
            'losses': combined_losses,
            'num_sub_batches': processed_batch1['current_sub_batch_size'] + processed_batch2['current_sub_batch_size'],
            'embedding1': embedding1,
            'embedding2': embedding2
        }

    def evaluate(self, test_loader: DataLoader, epoch: int, wandb: Optional[object] = None):
        """Evaluate model on test data"""
        self.eval()
        total_losses = {'total': 0.0, 'inverse': 0.0, 'forward': 0.0,
                       'intra_traj_consistency': 0.0, 'inter_traj_consistency': 0.0, 
                       'state': 0.0, 'factor': 0.0, 'policy': 0.0}
        num_batches = 0
        num_sub_batches = 0
        
        eval_progress = tqdm(test_loader, desc=f"Eval Epoch {epoch}", leave=False)
        with torch.no_grad():
            for batch in eval_progress:
                # Temporarily set min_visible_length to history for full attention during evaluation
                original_min_visible = self.training_config.min_visible_length
                self.training_config.min_visible_length = self.training_config.history
                
                step_results = self.train_step(batch)
                
                # Restore original setting
                self.training_config.min_visible_length = original_min_visible
                
                if step_results is None:
                    continue
                
                total_losses['total'] += step_results['total_loss'].item()
                for key, loss in step_results['losses'].items():
                    total_losses[key] += loss.item()
                
                num_sub_batches += step_results['num_sub_batches']
                num_batches += 1

        if num_batches > 0:
            avg_losses = {k: v / num_batches for k, v in total_losses.items()}
            
            print(f"Epoch {epoch}, Test Losses (full attention):")
            for key, loss in avg_losses.items():
                print(f"  {key.capitalize()}: {loss:.6f}")
            print(f"  Total Test Sub-batches: {num_sub_batches}")
            
            if wandb:
                try:
                    # Only log test losses under test_epoch
                    log_dict = {f"test_epoch/{key}_loss": loss for key, loss in avg_losses.items()}
                    wandb.log(log_dict)
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")
            
            return avg_losses['total']
        
        return None

    def train_embedding(self, train_loader: DataLoader, test_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None, eval_interval: Optional[int] = None,
              wandb: Optional[object] = None, log_dir: Optional[str] = None,
              save_checkpoint_epochs: int = 100):
        
        save_checkpoint_steps = max(1, (len(train_loader) // save_checkpoint_epochs))
        num_epochs = num_epochs or self.training_config.num_epochs
        eval_interval = eval_interval or self.training_config.eval_interval
        
        training_steps = 0
        best_test_loss = float('inf')
        
        # Log attention mask usage
        if self.training_config.min_visible_length < self.training_config.history:
            print(f"Training with random attention mask (min_visible_length={self.training_config.min_visible_length})")
            print("Evaluation will be performed with full attention for clean performance measurement")
        else:
            print("Training with full attention (no masking)")
        
        for epoch in trange(num_epochs, desc="Epochs"):
            # Training phase
            self.train()
            total_losses = {'total': 0.0, 'inverse': 0.0, 'forward': 0.0,
                           'intra_traj_consistency': 0.0, 'inter_traj_consistency': 0.0,
                           'state': 0.0, 'factor': 0.0, 'policy': 0.0}
            num_batches = 0
            num_sub_batches = 0
            epoch_step = 0

            train_progress = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", leave=False)
            for batch in train_progress:
                step_results = self.train_step(batch)
                if step_results is None:
                    continue

                # Backpropagation
                self.optimizer.zero_grad()
                step_results['total_loss'].backward()
                self.optimizer.step()

                # Accumulate losses
                total_losses['total'] += step_results['total_loss'].item()
                for key, loss in step_results['losses'].items():
                    total_losses[key] += loss.item()
                
                num_sub_batches += step_results['num_sub_batches']
                num_batches += 1

                # Update progress bar
                mask_indicator = " (masked)" if self.training_config.min_visible_length < self.training_config.history else ""
                train_progress.set_postfix(Total=f"{step_results['total_loss'].item():.4f}{mask_indicator}")

                # Log to wandb - only step losses
                if wandb:
                    try:
                        log_dict = {f"train_step/{key}_loss": loss.item() 
                                  for key, loss in step_results['losses'].items()}
                        wandb.log(log_dict)
                    except Exception as e:
                        print(f"Warning: Failed to log to wandb: {e}")

                training_steps += 1
                epoch_step += 1

                # Save checkpoint within epoch
                if log_dir and epoch_step % save_checkpoint_steps == 0:
                    checkpoint_path = os.path.join(log_dir, "checkpoints", 
                                                 f"epoch_{epoch + 1:04d}_step_{epoch_step:06d}.zip")
                    self.save_checkpoint(checkpoint_path, {
                        "epoch": epoch + 1, "epoch_step": epoch_step, 
                        "global_step": training_steps,
                        "train_loss": step_results['total_loss'].item()
                    })

            # Calculate average training losses
            if num_batches > 0:
                avg_losses = {k: v / num_batches for k, v in total_losses.items()}
                
                mask_info = f" (attention mask, min_visible={self.training_config.min_visible_length})" if self.training_config.min_visible_length < self.training_config.history else ""
                print(f"Epoch {epoch + 1}/{num_epochs}{mask_info}")
                for key, loss in avg_losses.items():
                    print(f"  {key.capitalize()} Loss: {loss:.6f}")
                print(f"  Total Sub-batches: {num_sub_batches}")
            
            # Evaluation
            avg_test_loss = None
            if (epoch + 1) % eval_interval == 0 and test_loader is not None:
                avg_test_loss = self.evaluate(test_loader, epoch + 1, wandb)
                
                # Save best model
                if avg_test_loss is not None and avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    if log_dir:
                        self.save_checkpoint(os.path.join(log_dir, "best_model.zip"), {
                            "epoch": epoch + 1, "best_test_loss": best_test_loss
                        })
                        if wandb:
                            try:
                                wandb.log({"model/best_test_loss": best_test_loss})
                            except Exception as e:
                                print(f"Warning: Failed to log to wandb: {e}")
            
            # Save end-of-epoch checkpoint
            if log_dir:
                checkpoint_path = os.path.join(log_dir, "checkpoints", f"epoch_{epoch + 1:04d}_final.zip")
                self.save_checkpoint(checkpoint_path, {
                    "epoch": epoch + 1, "epoch_step": epoch_step, "global_step": training_steps,
                    "train_loss": avg_losses['total'] if num_batches > 0 else None,
                    "test_loss": avg_test_loss
                })

            # Log epoch results to wandb - only training epoch losses
            if num_batches > 0 and wandb:
                try:
                    # Only log training epoch losses under train_epoch
                    epoch_logs = {f"train_epoch/{key}_loss": loss for key, loss in avg_losses.items()}
                    wandb.log(epoch_logs)
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")

        print("Training completed!")
        if wandb:
            try:
                wandb.log({"training/completed": True})
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")

    def save_checkpoint(self, checkpoint_path: str, metadata: dict):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        unique_id = str(uuid.uuid4())
        temp_model_path = checkpoint_path.replace(".zip", f"_model_temp_{unique_id}.pth")
        temp_optimizer_path = checkpoint_path.replace(".zip", f"_optimizer_temp_{unique_id}.pth")
        
        try:
            torch.save(self.state_dict(), temp_model_path)
            torch.save(self.optimizer.state_dict(), temp_optimizer_path)
            
            # Serialize metadata
            serializable_metadata = metadata.copy()
            serializable_metadata.update({
                "dynamics_config": self.dynamics_config.__dict__,
                "training_config": self.training_config.__dict__
            })
            
            # Handle nested config serialization
            if "embedding_config" in serializable_metadata["dynamics_config"]:
                embedding_config = serializable_metadata["dynamics_config"]["embedding_config"]
                if hasattr(embedding_config, '__dict__'):
                    serializable_metadata["dynamics_config"]["embedding_config"] = embedding_config.__dict__
            
            with zipfile.ZipFile(checkpoint_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(temp_model_path, arcname="model.pth")
                zf.write(temp_optimizer_path, arcname="optimizer.pth")
                zf.writestr("metadata.json", json.dumps(serializable_metadata, indent=4))
            
            print(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
            
        finally:
            for temp_path in [temp_model_path, temp_optimizer_path]:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    @staticmethod
    def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Tuple["DADP", dict]:
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        unique_id = str(uuid.uuid4())
        temp_model_path = checkpoint_path.replace(".zip", f"_model_temp_{unique_id}.pth")
        temp_optimizer_path = checkpoint_path.replace(".zip", f"_optimizer_temp_{unique_id}.pth")
        
        try:
            with zipfile.ZipFile(checkpoint_path, "r") as zf:
                with zf.open("metadata.json") as f:
                    metadata = json.load(f)
                
                for arc_name, temp_path in [("model.pth", temp_model_path), 
                                          ("optimizer.pth", temp_optimizer_path)]:
                    with zf.open(arc_name) as f:
                        with open(temp_path, "wb") as temp_file:
                            temp_file.write(f.read())
            
            # Reconstruct configs
            dynamics_config_dict = metadata["dynamics_config"]
            embedding_config = EmbeddingConfig(**dynamics_config_dict["embedding_config"])
            dynamics_config = DynamicsConfig(
                state_dim=dynamics_config_dict["state_dim"],
                action_dim=dynamics_config_dict["action_dim"],
                factor_dim=dynamics_config_dict.get("factor_dim", 0),
                head_hidden=dynamics_config_dict.get("head_hidden", 256),
                embedding_config=embedding_config
            )
            
            training_config = TrainingConfig(**metadata["training_config"])
            
            # Create and load model
            model = DADP(dynamics_config, training_config).to(device)
            model.load_state_dict(torch.load(temp_model_path, map_location=device))
            
            try:
                model.optimizer.load_state_dict(torch.load(temp_optimizer_path, map_location=device))
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
            
            print(f"Checkpoint loaded: {os.path.basename(checkpoint_path)}")
            print(f"Epoch: {metadata.get('epoch', 'Unknown')}")
            
            return model, metadata

        finally:
            for temp_path in [temp_model_path, temp_optimizer_path]:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

