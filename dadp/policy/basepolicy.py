import torch
import torch.nn as nn
from typing import Optional
import os
from abc import ABC, abstractmethod
from tqdm import tqdm

from ..dadp import DADP

class BasePolicy(nn.Module, ABC):
    """Base class for all policies"""
    
    def __init__(self, state_dim: int, action_dim: int, embedding_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
    
    @abstractmethod
    def forward(
        self, 
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        history_embedding: torch.Tensor, 
        current_state: torch.Tensor
        ) -> torch.Tensor:
        pass


class PolicyTrainer:
    """Simple trainer for policies"""
    
    def __init__(
        self,
        policy: BasePolicy,
        embedding_model: DADP,
        learning_rate: float = 3e-4,
        device: str = "cpu",
        mask_embedding: bool = False
    ):
        self.policy = policy.to(device)
        self.embedding_model = embedding_model.to(device)
        self.device = device
        self.mask_embedding = mask_embedding
        
        # Freeze embedding model
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        self.embedding_model.eval()
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.test_losses = []
        self.global_step = 0
        self.best_test_loss = float('inf')
        self.best_epoch = 0
        
        print(f"Initialized PolicyTrainer with mask_embedding: {self.mask_embedding}")
    
    def get_embedding(self, state_history, action_history, precomputed_embedding=None):
        """Get embedding (either computed, precomputed, or masked)"""
        if self.mask_embedding:
            # Use zero embedding for ablation study
            batch_size = state_history.shape[0]
            embedding_dim = self.policy.embedding_dim
            return torch.zeros(batch_size, embedding_dim, device=self.device)
        elif precomputed_embedding is not None:
            # Use precomputed embedding
            return precomputed_embedding
        else:
            # Compute embedding from scratch
            with torch.no_grad():
                return self.embedding_model.dynamics.encode_history(state_history, action_history)
    
    def train_epoch(self, train_loader, epoch, wandb=None):
        """Train for one epoch"""
        self.policy.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Handle precomputed embeddings
            if "precomputed_embedding" in batch.keys():
                original_batch = batch['original_data']
                precomputed_embedding = batch['precomputed_embedding'].to(self.device)
                
                states = original_batch["obs"]["state"].to(self.device)
                actions = original_batch["act"].to(self.device)
                
                state_history = states[:, :-2, :]
                action_history = actions[:, :-2, :]
                current_state = states[:, -2, :]
                current_action = actions[:, -2, :]
                
                # Get embedding (masked, precomputed, or computed)
                history_embedding = self.get_embedding(state_history, action_history, precomputed_embedding)
            else:
                # Original path
                states = batch["obs"]["state"].to(self.device)
                actions = batch["act"].to(self.device)
                
                state_history = states[:, :-2, :]
                action_history = actions[:, :-2, :]
                current_state = states[:, -2, :]
                current_action = actions[:, -2, :]
                
                # Get embedding (masked or computed)
                history_embedding = self.get_embedding(state_history, action_history)
            
            # Forward pass
            predicted_actions = self.policy(
                state_history=state_history,
                action_history=action_history,
                history_embedding=history_embedding, 
                current_state=current_state
            )
            
            loss = self.criterion(predicted_actions, current_action)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log to wandb every 10 steps - only loss as chart
            if wandb and self.global_step % 10 == 0:
                wandb.log({
                    "train_step_loss": loss.item(),
                    "global_step": self.global_step
                })
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}', 
                'Masked': 'Yes' if self.mask_embedding else 'No'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        # Log epoch-level metrics - only loss as chart
        if wandb:
            wandb.log({
                "train_epoch_loss": avg_loss,
                "epoch": epoch + 1
            })
        
        return avg_loss
    
    def evaluate(self, test_loader, epoch, wandb=None):
        """Evaluate on test set"""
        self.policy.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(test_loader, desc=f"Evaluating Epoch {epoch+1}", leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                # Handle precomputed embeddings
                if "precomputed_embedding" in batch.keys():
                    original_batch = batch['original_data']
                    precomputed_embedding = batch['precomputed_embedding'].to(self.device)
                    
                    states = original_batch["obs"]["state"].to(self.device)
                    actions = original_batch["act"].to(self.device)
                    
                    state_history = states[:, :-2, :]
                    action_history = actions[:, :-2, :]
                    current_state = states[:, -2, :]
                    current_action = actions[:, -2, :]
                    
                    # Get embedding (masked, precomputed, or computed)
                    history_embedding = self.get_embedding(state_history, action_history, precomputed_embedding)
                else:
                    # Original path
                    states = batch["obs"]["state"].to(self.device)
                    actions = batch["act"].to(self.device)
                    
                    state_history = states[:, :-2, :]
                    action_history = actions[:, :-2, :]
                    current_state = states[:, -2, :]
                    current_action = actions[:, -2, :]
                    
                    # Get embedding (masked or computed)
                    history_embedding = self.get_embedding(state_history, action_history)
                
                predicted_actions = self.policy(
                    state_history=state_history,
                    action_history=action_history,
                    history_embedding=history_embedding,
                    current_state=current_state
                )
                
                loss = self.criterion(predicted_actions, current_action)
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Masked': 'Yes' if self.mask_embedding else 'No'
                })
        
        avg_loss = total_loss / num_batches
        self.test_losses.append(avg_loss)
        
        # Update best model tracking
        if avg_loss < self.best_test_loss:
            self.best_test_loss = avg_loss
            self.best_epoch = epoch
        
        # Log test metrics - only loss as chart
        if wandb:
            wandb.log({
                "test_loss": avg_loss,
                "best_test_loss": self.best_test_loss,
                "epoch": epoch + 1
            })
        
        return avg_loss
    
    def save_checkpoint(self, epoch, log_dir, args, additional_metadata=None):
        """Save checkpoint similar to DADP format"""
        import zipfile
        import tempfile
        import json
        
        # Create checkpoint metadata
        metadata = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'best_test_loss': self.best_test_loss,
            'best_epoch': self.best_epoch,
            'mask_embedding': self.mask_embedding,  # Add mask_embedding to metadata
            'policy_config': {
                'state_dim': self.policy.state_dim,
                'action_dim': self.policy.action_dim,
                'embedding_dim': self.policy.embedding_dim,
                'model_type': self.policy.__class__.__name__
            },
            'training_args': vars(args)
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Create checkpoint filename
        mask_suffix = "_masked" if self.mask_embedding else ""
        checkpoint_name = f"epoch_{epoch+1:04d}_step_{self.global_step:06d}{mask_suffix}.zip"
        checkpoint_path = os.path.join(log_dir, "checkpoints", checkpoint_name)
        
        # Save checkpoint in zip format (similar to DADP)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model state dict
            model_path = os.path.join(temp_dir, "policy_model.pt")
            torch.save(self.policy.state_dict(), model_path)
            
            # Save optimizer state dict
            optimizer_path = os.path.join(temp_dir, "optimizer.pt")
            torch.save(self.optimizer.state_dict(), optimizer_path)
            
            # Save metadata
            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create zip file
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with zipfile.ZipFile(checkpoint_path, 'w') as zipf:
                zipf.write(model_path, "policy_model.pt")
                zipf.write(optimizer_path, "optimizer.pt")
                zipf.write(metadata_path, "metadata.json")
        
        print(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def train_policy(
        self,
        train_loader,
        test_loader,
        num_epochs: int,
        eval_interval: int = 1,
        wandb=None,
        log_dir: Optional[str] = None,
        save_checkpoint_epochs: int = 10,
        args=None
    ):
        """Training loop with comprehensive logging"""
        mask_status = "with MASKED embeddings" if self.mask_embedding else "with computed embeddings"
        print(f"Starting policy training for {num_epochs} epochs {mask_status}...")
        
        # Log initial hyperparameters to wandb - as config, not charts
        if wandb and args:
            wandb.config.update({
                "policy_architecture": {
                    "state_dim": self.policy.state_dim,
                    "action_dim": self.policy.action_dim,
                    "embedding_dim": self.policy.embedding_dim,
                    "model_type": self.policy.__class__.__name__
                },
                "optimizer": {
                    "type": "Adam",
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                },
                "training_config": {
                    "num_epochs": num_epochs,
                    "batch_size": getattr(args, 'batch_size', 'unknown'),
                    "eval_interval": eval_interval,
                    "save_checkpoint_epochs": save_checkpoint_epochs,
                    "mask_embedding": self.mask_embedding
                },
                "ablation_study": {
                    "mask_embedding": self.mask_embedding,
                    "embedding_type": "zero" if self.mask_embedding else "computed"
                }
            })
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, epoch, wandb)
            
            # Evaluation
            if (epoch + 1) % eval_interval == 0:
                test_loss = self.evaluate(test_loader, epoch, wandb)
                print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.6f}, Test: {test_loss:.6f}, Best: {self.best_test_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.6f}")
            
            # Save checkpoint
            if log_dir and (epoch + 1) % save_checkpoint_epochs == 0:
                self.save_checkpoint(epoch, log_dir, args)
            
            # Save best model checkpoint
            if log_dir and hasattr(self, 'best_test_loss') and (epoch + 1) % eval_interval == 0:
                current_test_loss = self.test_losses[-1] if self.test_losses else float('inf')
                if current_test_loss == self.best_test_loss:
                    best_checkpoint_path = os.path.join(log_dir, "checkpoints", "best_model.zip")
                    latest_checkpoint = self.save_checkpoint(epoch, log_dir, args)
                    # Copy the checkpoint to best_model.zip
                    import shutil
                    shutil.copy2(latest_checkpoint, best_checkpoint_path)
                    print(f"Saved best model: {best_checkpoint_path}")
        
        # Save final model
        if log_dir:
            final_checkpoint_path = self.save_checkpoint(num_epochs-1, log_dir, args, {'is_final': True})
            final_path = os.path.join(log_dir, "checkpoints", "final_model.zip")
            import shutil
            shutil.copy2(final_checkpoint_path, final_path)
            print(f"Saved final model: {final_path}")
        
        # Log final summary - as summary config, not charts
        if wandb:
            wandb.summary.update({
                "final_train_loss": self.train_losses[-1] if self.train_losses else 0,
                "final_test_loss": self.test_losses[-1] if self.test_losses else 0,
                "best_test_loss": self.best_test_loss,
                "best_epoch": self.best_epoch + 1,
                "total_epochs": num_epochs,
                "total_steps": self.global_step,
                "mask_embedding": self.mask_embedding,
                "training_mode": "MASKED" if self.mask_embedding else "COMPUTED"
            })
        
        print("Policy training completed!")
        print(f"Best test loss: {self.best_test_loss:.6f} at epoch {self.best_epoch + 1}")
        if self.train_losses:
            print(f"Final train loss: {self.train_losses[-1]:.6f}")
        if self.test_losses:
            print(f"Final test loss: {self.test_losses[-1]:.6f}")
        print(f"Training mode: {'MASKED embeddings' if self.mask_embedding else 'Computed embeddings'}")
