import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm, trange
import os
import numpy as np

from .casual_transformer_config import CausalTrainingConfig
from .casual_transformer_head import CausalDynamicsHead, save_causal_model_checkpoint


class CausalTransformerTrainer:
    """独立的 Causal Transformer 训练器"""
    
    def __init__(
        self,
        model: CausalDynamicsHead,
        training_config: CausalTrainingConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.training_config = training_config
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=training_config.learning_rate
        )
        
        # Training state
        self.current_epoch = 0
        self.best_test_loss = float('inf')
        self.global_step = 0  # 添加全局步数计数器

    def train_step(self, batch: Dict[str, torch.Tensor], wandb_logger: Optional[Any] = None, train: bool = True) -> Dict[str, float]:
        """单步训练 - 只处理3D embedding情况，并直接上传到wandb"""
        device = self.device
        
        # 处理输入数据
        states = batch["obs"]["state"].to(device)
        actions = batch["act"].to(device)
        
        # 简化数据处理
        state_history = states[:, :-2, :]     # (B, history, S)
        action_history = actions[:, :-2, :]   # (B, history, A)
        current_state = states[:, -2, :]      # (B, S)
        next_state = states[:, -1, :]         # (B, S)
        current_action = actions[:, -2, :]    # (B, A)
        
        # 1. Get embeddings using causal transformer - 始终返回3D
        embedding = self.model.encode_history(state_history, action_history)  # (B, num_segments, embedding_size)
        
        B, num_segments, embedding_size = embedding.shape
        
        # 在这里统一进行expand操作
        current_state_exp = current_state.unsqueeze(1).expand(B, num_segments, -1)   # (B, num_segments, state_dim)
        next_state_exp = next_state.unsqueeze(1).expand(B, num_segments, -1)         # (B, num_segments, state_dim)
        current_action_exp = current_action.unsqueeze(1).expand(B, num_segments, -1) # (B, num_segments, action_dim)
        
        # 2-6. 计算各种损失，传入已经扩展的维度
        pred_next_state = self.model.predict_next_state(embedding, current_state_exp, current_action_exp)
        forward_loss_per_segment = self.model.forward_loss(pred_next_state, next_state_exp)  # (num_segments,)
        
        pred_action = self.model.predict_action(embedding, current_state_exp, next_state_exp)
        inverse_loss_per_segment = self.model.inverse_loss(pred_action, current_action_exp)  # (num_segments,)
        
        pred_state = self.model.predict_state_from_embedding(embedding)
        state_loss_per_segment = self.model.state_loss(pred_state, current_state_exp)  # (num_segments,)
        
        embedding_detached = embedding.detach()
        pred_policy_action = self.model.predict_policy_action(embedding_detached, current_state_exp)
        policy_loss_per_segment = self.model.policy_loss(pred_policy_action, current_action_exp)  # (num_segments,)
        
        pred_next_state_embedding = self.model.predict_next_state_with_embedding(embedding, current_action_exp)
        embedding_forward_loss_per_segment = self.model.forward_loss(pred_next_state_embedding, next_state_exp)  # (num_segments,)
        
        # 7. Factor loss - 简化处理
        factor_loss_per_segment = torch.zeros(num_segments, device=device, requires_grad=True)
        if self.training_config.factor_loss_weight > 0 and 'task_list' in batch:
            pred_factor = self.model.pred_factor(embedding.detach())
            target_factor = batch["task_list"].to(device).float()
            
            # 扩展target_factor到3D
            target_factor = target_factor.expand(B, num_segments, -1)
            factor_loss_per_segment = self.model.factor_loss(pred_factor, target_factor)  # (num_segments,)
        
        # 计算平均损失用于反向传播
        forward_loss = forward_loss_per_segment.mean()
        inverse_loss = inverse_loss_per_segment.mean()
        state_loss = state_loss_per_segment.mean()
        policy_loss = policy_loss_per_segment.mean()
        embedding_forward_loss = embedding_forward_loss_per_segment.mean()
        factor_loss = factor_loss_per_segment.mean()
        
        # Calculate total loss
        total_loss = (
            self.training_config.inverse_loss_weight * inverse_loss +
            self.training_config.forward_loss_weight * forward_loss +
            self.training_config.state_loss_weight * state_loss +
            self.training_config.policy_loss_weight * policy_loss +
            self.training_config.embedding_forward_loss_weight * embedding_forward_loss +
            self.training_config.factor_loss_weight * factor_loss
        )
        
        # Backpropagation
        if train:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # 增加全局步数
        self.global_step += 1
        
        # 准备返回值，包括每个segment的损失
        return_dict = {
            'total_loss': total_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'forward_loss': forward_loss.item(),
            'state_loss': state_loss.item(),
            'policy_loss': policy_loss.item(),
            'embedding_forward_loss': embedding_forward_loss.item(),
            'factor_loss': factor_loss.item()
        }
        
        # 添加每个segment的损失
        for i in range(num_segments):
            return_dict[f'inverse_loss_segment_{i}'] = inverse_loss_per_segment[i].item()
            return_dict[f'forward_loss_segment_{i}'] = forward_loss_per_segment[i].item()
            return_dict[f'state_loss_segment_{i}'] = state_loss_per_segment[i].item()
            return_dict[f'policy_loss_segment_{i}'] = policy_loss_per_segment[i].item()
            return_dict[f'embedding_forward_loss_segment_{i}'] = embedding_forward_loss_per_segment[i].item()
            return_dict[f'factor_loss_segment_{i}'] = factor_loss_per_segment[i].item()
        
        # 直接上传到wandb，使用train_step_loss_name/segment_X的格式
        if wandb_logger:
            try:
                wandb_log_dict = {}
                
                # 基本损失，使用train_step前缀
                wandb_log_dict['train_step/total_loss'] = total_loss.item()
                wandb_log_dict['train_step/inverse_loss'] = inverse_loss.item()
                wandb_log_dict['train_step/forward_loss'] = forward_loss.item()
                wandb_log_dict['train_step/state_loss'] = state_loss.item()
                wandb_log_dict['train_step/policy_loss'] = policy_loss.item()
                wandb_log_dict['train_step/embedding_forward_loss'] = embedding_forward_loss.item()
                wandb_log_dict['train_step/factor_loss'] = factor_loss.item()
                
                # 每个损失类型按segment分别记录，使用 train_step_loss_name/segment_X 格式
                for i in range(num_segments):
                    # inverse loss 按 segment 分别记录
                    wandb_log_dict[f'train_step_inverse_loss/segment_{i}'] = inverse_loss_per_segment[i].item()
                    
                    # forward loss 按 segment 分别记录  
                    wandb_log_dict[f'train_step_forward_loss/segment_{i}'] = forward_loss_per_segment[i].item()
                    
                    # state loss 按 segment 分别记录
                    wandb_log_dict[f'train_step_state_loss/segment_{i}'] = state_loss_per_segment[i].item()
                    
                    # policy loss 按 segment 分别记录
                    wandb_log_dict[f'train_step_policy_loss/segment_{i}'] = policy_loss_per_segment[i].item()
                    
                    # embedding forward loss 按 segment 分别记录
                    wandb_log_dict[f'train_step_embedding_forward_loss/segment_{i}'] = embedding_forward_loss_per_segment[i].item()
                    
                    # factor loss 按 segment 分别记录
                    wandb_log_dict[f'train_step_factor_loss/segment_{i}'] = factor_loss_per_segment[i].item()
                
                # 添加步数和epoch信息
                wandb_log_dict['train_step/global_step'] = self.global_step
                wandb_log_dict['train_step/epoch'] = self.current_epoch + 1
                
                wandb_logger.log(wandb_log_dict)
                
            except Exception as e:
                print(f"Warning: Failed to log to wandb in train_step: {e}")
        
        return return_dict
    
    def train_epoch(self, train_loader: DataLoader, wandb_logger: Optional[Any] = None) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        # 获取num_segments用于初始化损失字典
        sample_batch = next(iter(train_loader))
        states = sample_batch["obs"]["state"].to(self.device)
        actions = sample_batch["act"].to(self.device)
        state_history = states[:, :-2, :]
        action_history = actions[:, :-2, :]
        sample_embedding = self.model.encode_history(state_history, action_history)
        num_segments = sample_embedding.shape[1]
        
        # 初始化损失字典
        epoch_losses = {
            'total_loss': 0.0,
            'inverse_loss': 0.0,
            'forward_loss': 0.0,
            'state_loss': 0.0,
            'policy_loss': 0.0,
            'embedding_forward_loss': 0.0,
            'factor_loss': 0.0
        }
        
        # 为每个segment添加损失统计
        for i in range(num_segments):
            epoch_losses[f'inverse_loss_segment_{i}'] = 0.0
            epoch_losses[f'forward_loss_segment_{i}'] = 0.0
            epoch_losses[f'state_loss_segment_{i}'] = 0.0
            epoch_losses[f'policy_loss_segment_{i}'] = 0.0
            epoch_losses[f'embedding_forward_loss_segment_{i}'] = 0.0
            epoch_losses[f'factor_loss_segment_{i}'] = 0.0
        
        num_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}", leave=False)
        for batch in train_progress:
            # 在train_step中直接传入wandb_logger
            losses = self.train_step(batch, wandb_logger=wandb_logger)
            
            # 直接累积损失，不再跳过任何批次
            for key, value in losses.items():
                epoch_losses[key] += value
            num_batches += 1
            
            # Update progress bar
            train_progress.set_postfix({
                "Total": f"{losses['total_loss']:.6f}",
                "Inv": f"{losses['inverse_loss']:.6f}",
                "Fwd": f"{losses['forward_loss']:.6f}",
                "Factor": f"{losses['factor_loss']:.6f}"
            })
        
        # Calculate average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        return avg_losses
    
    def evaluate(self, test_loader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                # 在评估时不传入wandb_logger
                losses = self.train_step(batch, wandb_logger=None, train=False)
                
                # 直接累积测试损失，不再跳过任何批次
                total_loss += losses['total_loss']
                num_batches += 1
        
        avg_test_loss = total_loss / num_batches
        return avg_test_loss
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None,
        wandb_logger: Optional[Any] = None
    ):
        """完整的训练循环"""
        print(f"Starting Causal Transformer training for {self.training_config.num_epochs} epochs...")
        
        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            checkpoints_dir = os.path.join(save_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
        
        for epoch in range(self.training_config.num_epochs):
            self.current_epoch = epoch
            
            # Training - 传入wandb_logger到train_epoch
            train_losses = self.train_epoch(train_loader, wandb_logger=wandb_logger)
            
            # Print training results
            print(f"\nEpoch {epoch + 1}/{self.training_config.num_epochs}")
            for key, value in train_losses.items():
                print(f"  {key}: {value:.6f}")
            
            # Evaluation
            if test_loader is not None and (epoch + 1) % self.training_config.eval_interval == 0:
                test_loss = self.evaluate(test_loader)
                print(f"  Test Loss: {test_loss:.6f}")
                
                # Save best model
                if test_loss < self.best_test_loss:
                    self.best_test_loss = test_loss
                    if save_dir:
                        best_model_path = os.path.join(save_dir, "best_causal_model.pth")
                        save_causal_model_checkpoint(
                            self.model, self.optimizer, best_model_path,
                            {
                                'epoch': epoch + 1,
                                'best_test_loss': self.best_test_loss,
                                'training_config': self.training_config.__dict__
                            }
                        )
                
                # 只记录epoch级别的汇总到wandb (train前缀)
                if wandb_logger:
                    log_dict = {f"train/{key}": value for key, value in train_losses.items()}
                    log_dict.update({
                        "test/total_loss": test_loss,
                        "epoch": epoch + 1
                    })
                    wandb_logger.log(log_dict)
            else:
                # Log only training results - epoch级别汇总
                if wandb_logger:
                    log_dict = {f"train/{key}": value for key, value in train_losses.items()}
                    log_dict["epoch"] = epoch + 1
                    wandb_logger.log(log_dict)
            
            # Save periodic checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoints_dir, f"causal_model_epoch_{epoch + 1}.pth")
                save_causal_model_checkpoint(
                    self.model, self.optimizer, checkpoint_path,
                    {
                        'epoch': epoch + 1,
                        'training_config': self.training_config.__dict__
                    }
                )
        
        print("Causal Transformer training completed!")
        return self.best_test_loss


def create_causal_trainer(
    state_dim: int,
    action_dim: int,
    factor_dim: int = 0,
    training_config: Optional[CausalTrainingConfig] = None,
    device: str = "cuda",
    **model_kwargs  # 接收所有模型参数
) -> CausalTransformerTrainer:
    """
    Factory function to create causal transformer trainer
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension  
        factor_dim: Factor dimension
        training_config: Training configuration
        device: Device for training
        **model_kwargs: Additional model arguments including:
            - embedding_size, d_model, head_hidden, separate_heads
            - total_length, num_segments
            - local_n_head, local_d_ff, local_n_layer, local_dropout
            - global_n_head, global_d_ff, global_n_layer, global_dropout
            - norm_z
        
    Returns:
        CausalTransformerTrainer instance
    """
    from .casual_transformer_head import create_causal_transformer_model
    
    if training_config is None:
        training_config = CausalTrainingConfig(device=device)
    
    # 创建模型时传递所有参数
    model = create_causal_transformer_model(
        state_dim=state_dim,
        action_dim=action_dim,
        factor_dim=factor_dim,
        **model_kwargs  # 传递所有模型参数
    )
    
    # Create trainer
    trainer = CausalTransformerTrainer(
        model=model,
        training_config=training_config,
        device=device
    )
    
    return trainer
