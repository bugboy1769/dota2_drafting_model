import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

from .model import DraftModel
from .dataset import DraftDataset
from .utils import setup_logging, save_checkpoint, load_checkpoint

class Trainer:

    def __init__(self, config:dict):
        self.config=config
        self.device=config['training']['device']
        self.logger=logging.getLogger(__name__)

        #Create Model
        self.model=DraftModel(
            num_heroes=config['model']['num_heroes'],
            embedding_dim=config['model']['embedding_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout'],
        ).to(self.device)

        #Loss functions
        self.policy_criterion=nn.CrossEntropyLoss()
        self.value_criterion=nn.BCELoss()
        self.role_criterion=nn.CrossEntropyLoss(ignore_index=0) # Ignore padding/bans

        #Optimizer
        self.optimizer=torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )

        #Scheduler
        #Dataloaders
        data_dir=Path(config['path']['data_dir'])/"processed"
        train_dataset=DraftDataset(data_dir/"train.pkl")
        val_dataset=DraftDataset(data_dir/"val.pkl")

        self.train_loader=DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=2
        )

        self.val_loader=DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=2
        )

        #Scheduler
        self.scheduler=torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['training']['learning_rate'],
            epochs=config['training']['num_epochs'],
            steps_per_epoch=len(self.train_loader)
        )

        #Training state
        self.best_val_loss=float('inf')
        self.epochs_without_improvement=0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss=0

        for batch in tqdm(self.train_loader, desc="Training"):
            hero_seq=batch['hero_sequence'].to(self.device)
            type_seq=batch['type_sequence'].to(self.device)
            team_seq=batch['team_sequence'].to(self.device)
            role_seq=batch['role_sequence'].to(self.device) # New
            valid_actions=batch['valid_actions'].to(self.device)
            target_action=batch['target_actions'].to(self.device)
            outcome=batch['outcome'].to(self.device)

            #Forward pass
            action_logits, win_prob, role_logits = self.model(hero_seq, type_seq, team_seq, valid_actions)

            #Compute losses
            policy_loss=self.policy_criterion(action_logits, target_action)
            value_loss=self.value_criterion(win_prob.squeeze(), outcome)
            
            # Role Loss: Flatten [Batch, Seq] -> [Batch*Seq]
            role_loss = self.role_criterion(role_logits.view(-1, 6), role_seq.view(-1))
            
            # Total Loss (Weighted)
            loss=policy_loss + 0.5*value_loss + 0.2*role_loss

            #Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            #Step scheduler every batch for OneCycleLR
            self.scheduler.step()

            total_loss+=loss.item()
        
        return total_loss/len(self.train_loader)
    
    def validate(self):

        self.model.eval()
        total_loss=0
        correct=0
        correct_top5=0
        total=0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                hero_seq=batch['hero_sequence'].to(self.device)
                type_seq=batch['type_sequence'].to(self.device)
                team_seq=batch['team_sequence'].to(self.device)
                role_seq=batch['role_sequence'].to(self.device) # New
                valid_actions=batch['valid_actions'].to(self.device)
                target_actions=batch['target_actions'].to(self.device)
                outcome=batch['outcome'].to(self.device)

                action_logits, win_prob, role_logits = self.model(hero_seq, type_seq, team_seq, valid_actions)

                policy_loss=self.policy_criterion(action_logits, target_actions)
                value_loss=self.value_criterion(win_prob.squeeze(), outcome)
                role_loss = self.role_criterion(role_logits.view(-1, 6), role_seq.view(-1))
                
                loss=policy_loss + 0.5*value_loss + 0.2*role_loss

                total_loss+=loss.item()
                
                # Top-1 Accuracy
                predictions=torch.argmax(action_logits, dim=1)
                correct+=(predictions==target_actions).sum().item()
                
                # Top-5 Accuracy
                _, top5_preds = torch.topk(action_logits, 5, dim=1)
                correct_top5 += (top5_preds == target_actions.unsqueeze(1)).any(dim=1).sum().item()
                
                total+=target_actions.size(0)
        
        return {
            'val_loss':total_loss/len(self.val_loader),
            'val_accuracy': correct/total,
            'val_top5_accuracy': correct_top5/total
        }
    
    def train(self):
        """Main training loop"""
        num_epochs=self.config['training']['num_epochs']
        patience=self.config['training']['early_stopping_patience']

        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            #Train
            train_loss=self.train_epoch()

            #Validate
            val_metrics=self.validate()

            #Log
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            self.logger.info(f"Val Accuracy: {val_metrics['val_accuracy']:.2%}")
            self.logger.info(f"Val Top-5 Accuracy: {val_metrics['val_top5_accuracy']:.2%}")

            #Save best model
            if val_metrics['val_loss']<self.best_val_loss:
                self.best_val_loss=val_metrics['val_loss']
                self.epochs_without_improvement=0

                model_path=Path(self.config['path']['model_dir'])/"best_model.pt"
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics, model_path)
                self.logger.info(f"Best Model Saved!")
            else:
                self.epochs_without_improvement+=1
            
            #Early stopping
            if self.epochs_without_improvement>=patience:
                self.logger.info(f"Early stopping after {epoch+1} epochs")
                break