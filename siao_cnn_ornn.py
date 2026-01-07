"""
SIAO-CNN-ORNN Model Training Pipeline

Complete training pipeline for IP-200 reactor fault classification using:
- Self-Improved Aquila Optimizer (SIAO) for hyperparameter optimization
- CNN for spatial feature extraction
- ORNN (Optimized RNN) for temporal sequence modeling

Author: SIAO-CNN-ORNN Integration Specialist
"""

import logging
from typing import Tuple, Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ORNN (Optimized RNN) Module
# =============================================================================

class OptimizedRNN(nn.Module):
    """
    Optimized RNN block for temporal sequence modeling.
    
    Uses LSTM/GRU with configurable hyperparameters optimized by SIAO.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = 'lstm'
    ):
        super(OptimizedRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # RNN layer
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, features]
        
        Returns:
            Output [batch, hidden_dim * num_directions]
        """
        # RNN forward
        output, _ = self.rnn(x)
        
        # Take last time step
        last_output = output[:, -1, :]
        
        return self.dropout(last_output)
    
    def get_output_dim(self) -> int:
        return self.hidden_dim * self.num_directions


# =============================================================================
# Complete SIAO-CNN-ORNN Model
# =============================================================================

class SIAO_CNN_ORNN(nn.Module):
    """
    Complete SIAO-optimized CNN-ORNN model for reactor fault classification.
    
    Architecture:
    Input → CNN (spatial features) → ORNN (temporal) → Classification
    """
    
    def __init__(
        self,
        input_time_steps: int = 50,
        input_features: int = 43,
        num_classes: int = 6,
        # CNN hyperparameters
        cnn_channels: Tuple[int, ...] = (32, 64, 128),
        cnn_embedding_dim: int = 256,
        # RNN hyperparameters
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 2,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        # Regularization
        dropout: float = 0.3
    ):
        super(SIAO_CNN_ORNN, self).__init__()
        
        self.input_time_steps = input_time_steps
        self.input_features = input_features
        self.num_classes = num_classes
        
        # CNN Feature Extractor
        self.cnn = self._build_cnn(
            input_features, cnn_channels, cnn_embedding_dim, dropout
        )
        
        # ORNN
        self.ornn = OptimizedRNN(
            input_dim=cnn_embedding_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            rnn_type=rnn_type
        )
        
        # Classification head
        classifier_input = self.ornn.get_output_dim()
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
        
    def _build_cnn(
        self,
        input_features: int,
        channels: Tuple[int, ...],
        embedding_dim: int,
        dropout: float
    ) -> nn.Module:
        """Build CNN feature extractor."""
        layers = []
        in_ch = 1
        
        for i, out_ch in enumerate(channels):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2) if i < len(channels) - 1 else nn.Identity()
            ])
            in_ch = out_ch
        
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, time_steps, features]
        
        Returns:
            Logits [batch, num_classes]
        """
        batch_size, time_steps, features = x.shape
        
        # Process each time step through CNN
        # Reshape: [batch * time_steps, 1, features]
        x_reshaped = x.view(batch_size * time_steps, 1, features)
        
        # CNN features: [batch * time_steps, embedding_dim]
        cnn_features = self.cnn(x_reshaped)
        
        # Reshape back: [batch, time_steps, embedding_dim]
        cnn_features = cnn_features.view(batch_size, time_steps, -1)
        
        # ORNN: [batch, rnn_output_dim]
        rnn_output = self.ornn(cnn_features)
        
        # Classify
        logits = self.classifier(rnn_output)
        
        return logits


# =============================================================================
# SIAO Hyperparameter Optimizer
# =============================================================================

class SIAOModelOptimizer:
    """
    Uses SIAO to optimize CNN-ORNN hyperparameters.
    
    Optimizes:
    - CNN embedding dimension
    - RNN hidden dimension
    - RNN layers
    - Dropout rate
    - Learning rate
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_classes: int = 6,
        device: str = 'cuda'
    ):
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.LongTensor(y_train)
        self.X_val = torch.FloatTensor(X_val)
        self.y_val = torch.LongTensor(y_val)
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Training config
        self.epochs_per_eval = 10  # Quick evaluation
        self.batch_size = 32
        
        logger.info(f"SIAOModelOptimizer initialized on {self.device}")
    
    def _decode_params(self, params: np.ndarray) -> Dict:
        """
        Decode continuous parameters to hyperparameter dict.
        
        params: [cnn_embed, rnn_hidden, rnn_layers, dropout, lr]
        """
        return {
            'cnn_embedding_dim': int(64 + params[0] * (512 - 64)),  # 64-512
            'rnn_hidden_dim': int(32 + params[1] * (256 - 32)),     # 32-256
            'rnn_num_layers': int(1 + params[2] * 3),               # 1-4
            'dropout': 0.1 + params[3] * 0.5,                       # 0.1-0.6
            'learning_rate': 10 ** (-4 + params[4] * 2)             # 1e-4 to 1e-2
        }
    
    def _train_and_evaluate(self, params: np.ndarray) -> float:
        """
        Train model with given hyperparameters and return validation loss.
        """
        try:
            hp = self._decode_params(params)
            
            # Create model
            model = SIAO_CNN_ORNN(
                input_time_steps=self.X_train.shape[1],
                input_features=self.X_train.shape[2],
                num_classes=self.num_classes,
                cnn_embedding_dim=hp['cnn_embedding_dim'],
                rnn_hidden_dim=hp['rnn_hidden_dim'],
                rnn_num_layers=hp['rnn_num_layers'],
                dropout=hp['dropout']
            ).to(self.device)
            
            # Create data loaders
            train_dataset = TensorDataset(self.X_train, self.y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])
            
            model.train()
            for epoch in range(self.epochs_per_eval):
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_x = self.X_val.to(self.device)
                val_y = self.y_val.to(self.device)
                outputs = model(val_x)
                val_loss = criterion(outputs, val_y).item()
                
                # Accuracy
                preds = outputs.argmax(dim=1).cpu().numpy()
                acc = accuracy_score(self.y_val.numpy(), preds)
            
            # Return negative accuracy (minimize → maximize accuracy)
            return 1 - acc  # RMSE-like: lower is better
            
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            return 1.0  # Worst case
    
    def optimize(
        self,
        pop_size: int = 20,
        max_iter: int = 30
    ) -> Tuple[Dict, float, Dict]:
        """
        Run SIAO optimization to find best hyperparameters.
        
        Returns:
            Tuple of (best_hyperparams, best_accuracy, info)
        """
        from siao_optimizer import SelfImprovedAquilaOptimizer
        
        logger.info("=" * 60)
        logger.info("Starting SIAO Hyperparameter Optimization")
        logger.info("=" * 60)
        
        # SIAO configuration
        dim = 5  # 5 hyperparameters
        lb = np.zeros(dim)
        ub = np.ones(dim)
        
        siao = SelfImprovedAquilaOptimizer(
            objective_func=self._train_and_evaluate,
            dim=dim,
            lb=lb,
            ub=ub,
            pop_size=pop_size,
            max_iter=max_iter,
            chaos_method='combined',
            minimize=True  # Minimize error
        )
        
        best_params, best_error, info = siao.optimize()
        
        best_hp = self._decode_params(best_params)
        best_accuracy = 1 - best_error
        
        logger.info("=" * 60)
        logger.info("SIAO Optimization Complete!")
        logger.info(f"Best Accuracy: {best_accuracy:.4f}")
        logger.info(f"Best Hyperparameters: {best_hp}")
        logger.info("=" * 60)
        
        return best_hp, best_accuracy, info


# =============================================================================
# Training Functions
# =============================================================================

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict]:
    """
    Train the SIAO-CNN-ORNN model.
    
    Returns:
        Trained model and training history
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Data
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_x = X_val.to(device)
            val_y = y_val.to(device)
            outputs = model(val_x)
            val_loss = criterion(outputs, val_y).item()
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val.numpy(), preds)
        
        # Record
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    logger.info(f"Training complete. Best Val Accuracy: {best_val_acc:.4f}")
    
    return model, history


def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate trained model.
    
    Returns:
        Dictionary with accuracy, F1 score, confusion matrix
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_test = torch.FloatTensor(X_test).to(device)
    y_test_np = y_test
    
    with torch.no_grad():
        outputs = model(X_test)
        preds = outputs.argmax(dim=1).cpu().numpy()
    
    accuracy = accuracy_score(y_test_np, preds)
    f1 = f1_score(y_test_np, preds, average='weighted')
    cm = confusion_matrix(y_test_np, preds)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': preds
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training history."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(history['val_acc'], label='Val Accuracy', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    """Plot confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib/seaborn not available")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_full_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int = 6,
    optimize_hyperparams: bool = True,
    epochs: int = 100
) -> Tuple[nn.Module, Dict, Dict]:
    """
    Run complete SIAO-CNN-ORNN training pipeline.
    
    Args:
        X: Input data [samples, time_steps, features]
        y: Labels [samples]
        num_classes: Number of classes
        optimize_hyperparams: Whether to use SIAO for hyperparameter optimization
        epochs: Training epochs
    
    Returns:
        Trained model, training history, evaluation results
    """
    logger.info("=" * 60)
    logger.info("SIAO-CNN-ORNN Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input shape: {X.shape}")
    logger.info(f"Classes: {num_classes}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Hyperparameter optimization
    if optimize_hyperparams:
        optimizer = SIAOModelOptimizer(
            X_train, y_train, X_val, y_val, num_classes
        )
        best_hp, _, _ = optimizer.optimize(pop_size=15, max_iter=20)
    else:
        best_hp = {
            'cnn_embedding_dim': 256,
            'rnn_hidden_dim': 128,
            'rnn_num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 1e-3
        }
    
    # Create model with best hyperparameters
    model = SIAO_CNN_ORNN(
        input_time_steps=X.shape[1],
        input_features=X.shape[2],
        num_classes=num_classes,
        cnn_embedding_dim=best_hp['cnn_embedding_dim'],
        rnn_hidden_dim=best_hp['rnn_hidden_dim'],
        rnn_num_layers=best_hp['rnn_num_layers'],
        dropout=best_hp['dropout']
    )
    
    # Train
    model, history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs,
        learning_rate=best_hp['learning_rate']
    )
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    return model, history, results


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("SIAO-CNN-ORNN Training Pipeline")
    print("Run this in Colab after loading your processed data!")
