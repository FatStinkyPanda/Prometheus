# backend/core/consciousness/self_improving_engine.py

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import torch
import torch.nn as nn
from collections import deque

from backend.utils.logger import Logger
from backend.memory.base_memory import BaseMemory

class AdaptiveAttentionNetwork(nn.Module):
    """Neural network for learning attention patterns over unlimited context."""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention for context importance
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feedforward network for importance scoring
        self.importance_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Context compression network
        self.compression_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through attention and importance scoring.
        
        Returns:
            - importance_scores: Tensor of shape (batch, seq_len, 1)
            - compressed_representation: Tensor of shape (batch, seq_len, input_dim // 2)
        """
        # Apply self-attention
        attn_output, attn_weights = self.attention(x, x, x, attn_mask=mask)
        
        # Calculate importance scores
        importance_scores = self.importance_network(attn_output)
        
        # Generate compressed representations
        compressed = self.compression_network(attn_output)
        
        return importance_scores, compressed

class SelfImprovingEngine:
    """
    A self-improving engine that learns from interactions to optimize
    context management and response generation over time.
    """
    
    def __init__(self, consciousness, memory: BaseMemory):
        self.logger = Logger(__name__)
        self.consciousness = consciousness
        self.memory = memory
        
        # Learning components
        self.attention_model = AdaptiveAttentionNetwork()
        self.optimizer = torch.optim.Adam(self.attention_model.parameters(), lr=0.001)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.learning_metrics = {
            'total_interactions': 0,
            'successful_predictions': 0,
            'context_efficiency': 0.0,
            'user_satisfaction': 0.0
        }
        
        # Pattern recognition
        self.interaction_patterns = {}
        self.context_patterns = {}
        
        # Continuous learning parameters
        self.batch_size = 32
        self.learning_interval = 100  # Learn every N interactions
        self.model_checkpoint_interval = 1000
        
        self.logger.info("SelfImprovingEngine initialized")
        
    async def learn_from_interaction(
        self,
        interaction_data: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None
    ):
        """Learn from a completed interaction to improve future performance."""
        self.learning_metrics['total_interactions'] += 1
        
        # Extract learning signals
        learning_signals = self._extract_learning_signals(interaction_data, feedback)
        
        # Update performance history
        self.performance_history.append({
            'timestamp': datetime.utcnow(),
            'signals': learning_signals,
            'feedback': feedback
        })
        
        # Perform learning if we have enough data
        if len(self.performance_history) >= self.batch_size:
            if self.learning_metrics['total_interactions'] % self.learning_interval == 0:
                await self._perform_learning_cycle()
                
        # Save checkpoint periodically
        if self.learning_metrics['total_interactions'] % self.model_checkpoint_interval == 0:
            await self._save_model_checkpoint()
            
    def _extract_learning_signals(
        self,
        interaction_data: Dict[str, Any],
        feedback: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract signals for learning from interaction data."""
        signals = {
            'input_length': interaction_data.get('total_input_tokens', 0),
            'output_length': interaction_data.get('total_output_tokens', 0),
            'context_retrievals': len(interaction_data.get('context_used', [])),
            'processing_time': interaction_data.get('processing_time', 0),
            'compression_ratio': interaction_data.get('compression_ratio', 1.0)
        }
        
        # Add feedback signals if available
        if feedback:
            signals['user_rating'] = feedback.get('rating', 0.5)
            signals['relevance_score'] = feedback.get('relevance', 0.5)
            signals['helpful_chunks'] = feedback.get('helpful_chunks', [])
            signals['irrelevant_chunks'] = feedback.get('irrelevant_chunks', [])
            
        return signals
        
    async def _perform_learning_cycle(self):
        """Perform a learning cycle to update the attention model."""
        self.logger.info("Starting learning cycle")
        
        # Prepare training batch
        batch_data = list(self.performance_history)[-self.batch_size:]
        
        # Convert to tensors
        embeddings = []
        importance_targets = []
        
        for data in batch_data:
            # Get embeddings from interaction
            if 'embeddings' in data['signals']:
                embeddings.append(data['signals']['embeddings'])
                
            # Calculate importance based on feedback
            importance = self._calculate_importance_target(data)
            importance_targets.append(importance)
            
        if not embeddings:
            return
            
        # Stack tensors
        embeddings_tensor = torch.stack([torch.tensor(e) for e in embeddings])
        importance_tensor = torch.stack([torch.tensor(i) for i in importance_targets])
        
        # Training step
        self.attention_model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted_importance, compressed = self.attention_model(embeddings_tensor)
        
        # Calculate loss
        loss = nn.MSELoss()(predicted_importance.squeeze(), importance_tensor)
        
        # Add compression regularization
        compression_loss = torch.mean(torch.abs(compressed))
        total_loss = loss + 0.1 * compression_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Update metrics
        self.learning_metrics['context_efficiency'] = 1.0 - total_loss.item()
        
        self.logger.info(f"Learning cycle complete. Loss: {total_loss.item():.4f}")
        
    def _calculate_importance_target(self, data: Dict[str, Any]) -> float:
        """Calculate target importance based on feedback and signals."""
        importance = 0.5  # Base importance
        
        signals = data['signals']
        feedback = data.get('feedback', {})
        
        # Adjust based on user feedback
        if feedback:
            if 'rating' in feedback:
                importance = feedback['rating']
            
            # Boost importance for helpful chunks
            if signals.get('chunk_id') in feedback.get('helpful_chunks', []):
                importance = min(importance + 0.3, 1.0)
                
            # Reduce importance for irrelevant chunks
            if signals.get('chunk_id') in feedback.get('irrelevant_chunks', []):
                importance = max(importance - 0.3, 0.0)
                
        # Adjust based on performance signals
        if signals.get('processing_time', 0) > 5.0:
            importance *= 0.9  # Penalize slow processing
            
        if signals.get('compression_ratio', 1.0) < 0.5:
            importance *= 1.1  # Reward efficient compression
            
        return float(np.clip(importance, 0.0, 1.0))
        
    async def predict_importance(
        self,
        embeddings: List[np.ndarray],
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Predict importance scores for a list of embeddings."""
        if not embeddings:
            return []
            
        self.attention_model.eval()
        
        with torch.no_grad():
            embeddings_tensor = torch.stack([torch.tensor(e) for e in embeddings])
            importance_scores, _ = self.attention_model(embeddings_tensor.unsqueeze(0))
            
        scores = importance_scores.squeeze().cpu().numpy()
        
        # Apply context-based adjustments
        if context:
            scores = self._apply_context_adjustments(scores, context)
            
        return scores.tolist()
        
    def _apply_context_adjustments(
        self,
        scores: np.ndarray,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """Apply context-based adjustments to importance scores."""
        # Boost scores for recent context
        if context.get('is_recent'):
            scores *= 1.2
            
        # Boost scores for question context
        if context.get('contains_question'):
            scores *= 1.5
            
        # Apply user preference patterns
        if context.get('user_preferences'):
            # This would apply learned user-specific patterns
            pass
            
        return np.clip(scores, 0.0, 1.0)
        
    async def optimize_context_retrieval(
        self,
        query: str,
        available_contexts: List[Dict[str, Any]],
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Optimize context retrieval using learned patterns.
        
        This method uses the learned attention model to select the most
        relevant context pieces within the token budget.
        """
        if not available_contexts:
            return []
            
        # Get embeddings for all contexts
        embeddings = []
        for ctx in available_contexts:
            if 'embedding' in ctx:
                embeddings.append(ctx['embedding'])
            else:
                # Generate embedding if not available
                # This would use the embedding model
                embeddings.append(np.random.randn(768))  # Placeholder
                
        # Predict importance scores
        importance_scores = await self.predict_importance(embeddings)
        
        # Sort contexts by importance
        scored_contexts = list(zip(available_contexts, importance_scores))
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        
        # Select contexts within token budget
        selected_contexts = []
        current_tokens = 0
        
        for ctx, score in scored_contexts:
            ctx_tokens = len(ctx.get('content', '').split())
            if current_tokens + ctx_tokens <= max_tokens:
                selected_contexts.append(ctx)
                current_tokens += ctx_tokens
            else:
                # Try to compress and fit
                compressed = self._compress_context(ctx, max_tokens - current_tokens)
                if compressed:
                    selected_contexts.append(compressed)
                break
                
        return selected_contexts
        
    def _compress_context(
        self,
        context: Dict[str, Any],
        max_tokens: int
    ) -> Optional[Dict[str, Any]]:
        """Compress a context to fit within token limit."""
        content = context.get('content', '')
        if not content:
            return None
            
        # Simple compression by taking first N tokens
        # In production, use more sophisticated compression
        tokens = content.split()
        if len(tokens) <= max_tokens:
            return context
            
        compressed_content = ' '.join(tokens[:max_tokens])
        compressed_context = context.copy()
        compressed_context['content'] = compressed_content
        compressed_context['compressed'] = True
        
        return compressed_context
        
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_interactions': self.learning_metrics['total_interactions'],
            'model_performance': {
                'context_efficiency': self.learning_metrics['context_efficiency'],
                'average_importance_accuracy': self._calculate_importance_accuracy(),
                'compression_effectiveness': self._calculate_compression_effectiveness()
            },
            'learning_progress': {
                'total_learning_cycles': self.learning_metrics['total_interactions'] // self.learning_interval,
                'improvement_rate': self._calculate_improvement_rate()
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
        
    def _calculate_importance_accuracy(self) -> float:
        """Calculate accuracy of importance predictions."""
        if not self.performance_history:
            return 0.0
            
        # Compare predicted importance with actual user feedback
        accuracies = []
        for data in list(self.performance_history)[-100:]:
            if 'predicted_importance' in data['signals'] and data.get('feedback'):
                predicted = data['signals']['predicted_importance']
                actual = data['feedback'].get('rating', 0.5)
                accuracy = 1.0 - abs(predicted - actual)
                accuracies.append(accuracy)
                
        return np.mean(accuracies) if accuracies else 0.0
        
    def _calculate_compression_effectiveness(self) -> float:
        """Calculate effectiveness of context compression."""
        if not self.performance_history:
            return 0.0
            
        ratios = []
        for data in list(self.performance_history)[-100:]:
            if 'compression_ratio' in data['signals']:
                # Good compression maintains quality while reducing size
                ratio = data['signals']['compression_ratio']
                quality = data.get('feedback', {}).get('rating', 0.5)
                effectiveness = (1.0 - ratio) * quality  # Higher compression with high quality
                ratios.append(effectiveness)
                
        return np.mean(ratios) if ratios else 0.0
        
    def _calculate_improvement_rate(self) -> float:
        """Calculate the rate of improvement over time."""
        if len(self.performance_history) < 100:
            return 0.0
            
        # Compare recent performance with older performance
        old_performance = list(self.performance_history)[:50]
        new_performance = list(self.performance_history)[-50:]
        
        old_scores = [d.get('feedback', {}).get('rating', 0.5) for d in old_performance]
        new_scores = [d.get('feedback', {}).get('rating', 0.5) for d in new_performance]
        
        old_avg = np.mean(old_scores) if old_scores else 0.5
        new_avg = np.mean(new_scores) if new_scores else 0.5
        
        return (new_avg - old_avg) / max(old_avg, 0.1)  # Percentage improvement
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for system improvement."""
        recommendations = []
        
        # Check context efficiency
        if self.learning_metrics['context_efficiency'] < 0.7:
            recommendations.append(
                "Context retrieval efficiency is below optimal. "
                "Consider adjusting the attention model architecture or increasing training frequency."
            )
            
        # Check improvement rate
        improvement_rate = self._calculate_improvement_rate()
        if improvement_rate < 0.05:
            recommendations.append(
                "Learning rate has plateaued. "
                "Consider introducing new learning signals or adjusting hyperparameters."
            )
            
        # Check compression effectiveness
        compression_effectiveness = self._calculate_compression_effectiveness()
        if compression_effectiveness < 0.6:
            recommendations.append(
                "Context compression could be more effective. "
                "Consider implementing more sophisticated compression algorithms."
            )
            
        if not recommendations:
            recommendations.append("System is performing optimally. Continue monitoring.")
            
        return recommendations
        
    async def _save_model_checkpoint(self):
        """Save a checkpoint of the attention model."""
        checkpoint = {
            'model_state_dict': self.attention_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_metrics': self.learning_metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # In production, save to a proper model store
        self.logger.info(f"Model checkpoint saved at interaction {self.learning_metrics['total_interactions']}")
        
    async def load_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Load a model checkpoint."""
        self.attention_model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        self.learning_metrics = checkpoint_data['learning_metrics']
        
        self.logger.info("Model checkpoint loaded successfully")