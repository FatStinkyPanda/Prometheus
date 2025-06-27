# backend/core/consciousness/advanced_self_improving_engine.py

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import pickle
import os
import sys
from pathlib import Path

from backend.utils.logger import Logger
from backend.memory.base_memory import BaseMemory

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.core.consciousness.unified_consciousness import UnifiedConsciousness


class MetaLearningNetwork(nn.Module):
    """
    A meta-learning network that learns how to learn from user interactions.
    This network adapts its own learning strategies based on performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.logger = Logger(self.__class__.__name__)
        cfg = config or {}

        self.input_dim: int = cfg.get('input_dim', 768)
        self.hidden_dim: int = cfg.get('hidden_dim', 512)
        self.num_tasks: int = cfg.get('num_tasks', 10)
    
        self.task_embeddings = nn.Embedding(self.num_tasks, self.hidden_dim)
        
        self.meta_encoder = nn.LSTM(
            self.input_dim + self.hidden_dim,
            self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.adaptation_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        self.performance_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        
        self.strategy_network = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 64), nn.ReLU(),
            nn.Linear(64, 5), 
            nn.Softmax(dim=-1)
        )
        self.logger.info(f"MetaLearningNetwork initialized. Input: {self.input_dim}, Hidden: {self.hidden_dim}, Tasks: {self.num_tasks}")

    def forward(
        self,
        x: torch.Tensor, 
        task_id: Union[int, torch.Tensor], 
        adaptation_steps: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if x.ndim == 2: 
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        if isinstance(task_id, int):
            task_id_tensor = torch.tensor([task_id], device=x.device) 
        elif isinstance(task_id, torch.Tensor):
            task_id_tensor = task_id.to(x.device)
        else:
            self.logger.error(f"Invalid task_id type: {type(task_id)}. Must be int or Tensor.")
            task_id_tensor = torch.tensor([0], device=x.device)

        if task_id_tensor.ndim == 0: 
            task_id_tensor = task_id_tensor.unsqueeze(0)

        task_id_tensor = torch.clamp(task_id_tensor, 0, self.num_tasks - 1)

        task_emb = self.task_embeddings(task_id_tensor) 
        if task_emb.shape[0] == 1 and batch_size > 1:
             task_emb = task_emb.expand(batch_size, -1) 
        
        task_emb = task_emb.unsqueeze(1).expand(-1, seq_len, -1) 
        
        meta_input = torch.cat([x, task_emb], dim=-1)
        lstm_out, (h_n, _) = self.meta_encoder(meta_input) 
        
        global_repr = torch.cat([h_n[-2,:,:], h_n[-1,:,:]], dim=-1) 
        
        adapted_repr = x 
        for _ in range(adaptation_steps):
            adaptation_delta = self.adaptation_network(global_repr) 
            adapted_repr = adapted_repr + 0.1 * adaptation_delta.unsqueeze(1) 
            
        predicted_performance = self.performance_predictor(global_repr) 
        strategy_weights = self.strategy_network(global_repr) 
        
        return adapted_repr, predicted_performance, strategy_weights


class PatternRecognitionModule:
    def __init__(self, pattern_memory_size: int = 10000):
        self.logger = Logger(self.__class__.__name__)
        self.pattern_memory_size = pattern_memory_size
        self.interaction_patterns: deque = deque(maxlen=pattern_memory_size)
        self.response_patterns: deque = deque(maxlen=pattern_memory_size)
        self.success_patterns: deque = deque(maxlen=pattern_memory_size)
        self.pattern_frequencies: Dict[str, int] = defaultdict(int)
        self.pattern_success_rates: Dict[str, Dict[str, int]] = defaultdict(lambda: {'success': 0, 'total': 0})
        self.pattern_clusters: Dict = {}
        self.pattern_transitions: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.logger.info(f"PatternRecognitionModule initialized with memory size {pattern_memory_size}.")

    async def record_pattern(self, interaction_data: Dict[str, Any], response_data: Dict[str, Any], success_metrics: Dict[str, float]):
        pattern = self._extract_pattern_features(interaction_data, response_data)
        self.interaction_patterns.append(pattern['interaction'])
        self.response_patterns.append(pattern['response'])
        self.success_patterns.append(success_metrics)
        pattern_key = self._pattern_to_key(pattern)
        self.pattern_frequencies[pattern_key] += 1
        success_score = success_metrics.get('overall_success', 0.5)
        self.pattern_success_rates[pattern_key]['total'] += 1
        if success_score > 0.7: self.pattern_success_rates[pattern_key]['success'] += 1
        if len(self.interaction_patterns) > 1:
            prev_pattern_key = self._pattern_to_key({'interaction': self.interaction_patterns[-2], 'response': self.response_patterns[-2]})
            self.pattern_transitions[prev_pattern_key][pattern_key] += 1
            
    def _extract_pattern_features(self, interaction_data: Dict[str, Any], response_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'interaction': {
                'input_length': interaction_data.get('input_length', len(str(interaction_data.get('text','')))),
                'input_type': interaction_data.get('input_type', 'text'),
                'session_duration': interaction_data.get('session_duration', 0),
                'query_complexity': self._estimate_complexity(interaction_data),
                'emotional_tone': interaction_data.get('emotional_tone', interaction_data.get('emotional_state', 'neutral'))
            },
            'response': {
                'response_length': response_data.get('response_length', len(str(response_data.get('text','')))),
                'response_type': response_data.get('response_type', 'text'),
                'processing_time': response_data.get('processing_time', 0),
                'confidence': response_data.get('confidence', 0.5),
                'minds_involved': list(response_data.get('minds_involved', []))
            }
        }
        
    def _pattern_to_key(self, pattern: Dict[str, Any]) -> str:
        interaction_f = pattern.get('interaction', {})
        response_f = pattern.get('response', {})
        key_parts = [
            f"il_{interaction_f.get('input_length', 0) // 100}",
            f"it_{str(interaction_f.get('input_type', 'unknown'))[:3]}",
            f"qc_{str(interaction_f.get('query_complexity', 'medium'))[:3]}",
            f"et_{str(interaction_f.get('emotional_tone', 'neut'))[:4]}",
            f"rt_{str(response_f.get('response_type', 'unknown'))[:3]}"
        ]
        return '_'.join(key_parts)
        
    def _estimate_complexity(self, interaction_data: Dict[str, Any]) -> str:
        text_content = str(interaction_data.get('text', interaction_data.get('content', '')))
        input_length = len(text_content)
        num_entities = len(interaction_data.get('entities', []))
        num_questions = text_content.count('?')
        complexity_score = (input_length / 100 + num_entities * 2 + num_questions * 3)
        if complexity_score < 5: return 'simple'
        elif complexity_score < 15: return 'medium'
        else: return 'complex'
            
    def get_pattern_recommendations(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        current_pattern_features = self._extract_pattern_features(current_context.get('interaction', {}), current_context.get('response', {}))
        current_key = self._pattern_to_key(current_pattern_features)
        
        similar_patterns = []
        for pattern_key, stats in self.pattern_success_rates.items():
            if stats.get('total', 0) > 0:
                success_rate = stats['success'] / stats['total']
                if success_rate > 0.7: 
                    similarity = self._compute_pattern_similarity(current_key, pattern_key)
                    if similarity > 0.6: 
                        similar_patterns.append({'pattern': pattern_key, 'success_rate': success_rate, 
                                                 'frequency': self.pattern_frequencies.get(pattern_key,0), 'similarity': similarity})
        similar_patterns.sort(key=lambda x: x['success_rate'] * np.log1p(x['frequency']) * x['similarity'], reverse=True)

        next_patterns = []
        if current_key in self.pattern_transitions:
            transitions = self.pattern_transitions[current_key]
            total_transitions = sum(transitions.values())
            if total_transitions > 0:
                for next_key, count in transitions.items():
                    probability = count / total_transitions
                    sr_next = self.pattern_success_rates.get(next_key, {'success':0, 'total':0})
                    next_success_rate = sr_next['success'] / max(sr_next['total'], 1)
                    next_patterns.append({'pattern': next_key, 'probability': probability, 'success_rate': next_success_rate})
        next_patterns.sort(key=lambda x: x['probability'] * x['success_rate'], reverse=True)

        return {
            'similar_successful_patterns': similar_patterns[:5],
            'likely_next_patterns': next_patterns[:3],
            'current_pattern_stats': self.pattern_success_rates.get(current_key, {'success':0, 'total':0})
        }
        
    def _compute_pattern_similarity(self, pattern_key1: str, pattern_key2: str) -> float:
        parts1 = pattern_key1.split('_')
        parts2 = pattern_key2.split('_')
        if len(parts1) != len(parts2): return 0.0
        matches = sum(1 for p1, p2 in zip(parts1, parts2) if p1 == p2)
        return matches / len(parts1) if len(parts1) > 0 else 0.0


class ContinuousLearningModule:
    def __init__(self, memory_size: int = 100000, rehearsal_size: int = 1000, base_learning_rate: float = 0.001):
        self.logger = Logger(self.__class__.__name__)
        self.memory_size = memory_size
        self.rehearsal_size = rehearsal_size
        self.experience_buffer: deque = deque(maxlen=memory_size)
        self.rehearsal_memory: List[Dict[str, Any]] = []
        self.learning_stats: Dict[str, Any] = {
            'total_experiences': 0, 'rehearsal_updates': 0,
            'performance_history': deque(maxlen=1000), 
            'learning_rate_history': deque(maxlen=1000)
        }
        self.base_learning_rate = base_learning_rate
        self.current_learning_rate = base_learning_rate
        self.logger.info(f"ContinuousLearningModule initialized. Buffer: {memory_size}, Rehearsal: {rehearsal_size}, LR: {base_learning_rate}")

    async def add_experience(self, state: Dict[str, Any], action: Dict[str, Any], reward: float, next_state: Dict[str, Any], metadata: Dict[str, Any]):
        experience = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'metadata': metadata, 'timestamp': datetime.utcnow()}
        self.experience_buffer.append(experience)
        self.learning_stats['total_experiences'] += 1
        if self.learning_stats['total_experiences'] > 0 and self.learning_stats['total_experiences'] % 100 == 0: 
             await self._update_rehearsal_memory()
            
    async def _update_rehearsal_memory(self):
        if not self.experience_buffer: return
        if len(self.experience_buffer) < self.rehearsal_size:
            self.rehearsal_memory = list(self.experience_buffer)
            return
        candidates = list(self.experience_buffer)
        def score_experience(exp):
            reward_score = exp.get('reward', 0.5)
            age_days = (datetime.utcnow() - exp.get('timestamp', datetime.utcnow())).total_seconds() / (24*3600)
            recency_score = np.exp(-age_days * 0.1) 
            metadata_exp = exp.get('metadata',{})
            complexity_metric_val = metadata_exp.get('complexity_metric', 0.5) if isinstance(metadata_exp.get('complexity_metric'), (int,float)) else 0.5
            complexity_score = complexity_metric_val
            return reward_score * 0.5 + recency_score * 0.3 + complexity_score * 0.2

        candidates.sort(key=score_experience, reverse=True)
        self.rehearsal_memory = candidates[:self.rehearsal_size]
        self.learning_stats['rehearsal_updates'] += 1
        
    def get_learning_batch(self, batch_size: int, include_rehearsal: bool = True) -> List[Dict[str, Any]]:
        if not self.experience_buffer: return []
        
        buffer_list = list(self.experience_buffer)
        if len(buffer_list) == 0: return [] 
        if len(buffer_list) < batch_size: return buffer_list 

        if include_rehearsal and self.rehearsal_memory:
            recent_sample_size = int(batch_size * 0.7)
            rehearsal_sample_size = batch_size - recent_sample_size
            
            recent_candidates_pool_size = min(len(buffer_list)//5 if len(buffer_list)//5 > 0 else len(buffer_list), 1000)
            recent_candidates = buffer_list[-recent_candidates_pool_size:]
            actual_recent_size = min(recent_sample_size, len(recent_candidates))
            recent_batch = []
            if actual_recent_size > 0:
                recent_batch_indices = np.random.choice(len(recent_candidates), size=actual_recent_size, replace=False)
                recent_batch = [recent_candidates[i] for i in recent_batch_indices]

            actual_rehearsal_size = min(rehearsal_sample_size, len(self.rehearsal_memory))
            rehearsal_batch = []
            if actual_rehearsal_size > 0:
                rehearsal_batch_indices = np.random.choice(len(self.rehearsal_memory), size=actual_rehearsal_size, replace=False)
                rehearsal_batch = [self.rehearsal_memory[i] for i in rehearsal_batch_indices]
            
            combined_batch = recent_batch + rehearsal_batch
            if len(combined_batch) < batch_size:
                fill_needed = batch_size - len(combined_batch)
                
                # Create set of IDs from combined_batch for faster checking
                combined_ids = {id(exp) for exp in combined_batch}
                remaining_buffer = [exp for exp in buffer_list if id(exp) not in combined_ids]

                if len(remaining_buffer) >= fill_needed:
                     fill_indices = np.random.choice(len(remaining_buffer), size=fill_needed, replace=False)
                     combined_batch.extend([remaining_buffer[i] for i in fill_indices])
            return combined_batch[:batch_size]
        else:
            indices = np.random.choice(len(buffer_list), size=batch_size, replace=False)
            return [buffer_list[i] for i in indices]
            
    def update_learning_rate(self, performance_metrics: Dict[str, float]):
        overall_performance = performance_metrics.get('overall_performance', 0.5)
        self.learning_stats['performance_history'].append(overall_performance)
        if len(self.learning_stats['performance_history']) > 10:
            recent_performance_slice = list(self.learning_stats['performance_history'])[-10:]
            try:
                if len(set(recent_performance_slice)) == 1: 
                    performance_trend = 0.0 
                else:
                    performance_trend = np.polyfit(range(10), recent_performance_slice, 1)[0]
                
                if performance_trend > 0.01: self.current_learning_rate = min(self.current_learning_rate * 1.05, self.base_learning_rate * 10)
                elif performance_trend < -0.01: self.current_learning_rate = max(self.current_learning_rate * 0.95, self.base_learning_rate * 0.1)
            except np.linalg.LinAlgError:
                self.logger.warning("Could not compute performance trend for LR update (identical values).")
            except Exception as e:
                 self.logger.error(f"Error computing performance trend: {e}", exc_info=True)

        self.current_learning_rate = np.clip(self.current_learning_rate, self.base_learning_rate * 0.1, self.base_learning_rate * 10)
        self.learning_stats['learning_rate_history'].append(self.current_learning_rate)


class AdvancedSelfImprovingEngine:
    def __init__(
        self,
        consciousness: 'UnifiedConsciousness', 
        memory: BaseMemory, 
        config: Optional[Dict[str, Any]] = None
    ):
        self.logger = Logger(__name__)
        self.consciousness = consciousness 
        self.memory = memory
        self.config = config if isinstance(config, dict) else {} 

        meta_learner_cfg = self.config.get('meta_learner_network', {})
        self.meta_learner = MetaLearningNetwork(config=meta_learner_cfg)
        
        pr_cfg = self.config.get('pattern_recognition', {}) 
        self.pattern_recognizer = PatternRecognitionModule(
            pattern_memory_size=pr_cfg.get('memory_size', 10000)
        )

        cl_cfg = self.config.get('continuous_learning', {}) 
        self.continuous_learner = ContinuousLearningModule(
            memory_size=cl_cfg.get('memory_size', 100000),
            rehearsal_size=cl_cfg.get('rehearsal_size', 1000),
            base_learning_rate=cl_cfg.get('base_learning_rate', 0.001)
        )
        
        self.batch_size: int = self.config.get('batch_size', 32)
        self.learning_interval: int = self.config.get('learning_interval', 100)
        self.meta_learning_interval: int = self.config.get('meta_learning_interval', 500)
        self.save_interval: int = self.config.get('save_interval', 1000)
        
        model_dir_str = self.config.get('model_dir', 'models/self_improving_engine') 
        self.model_dir = Path(model_dir_str)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {'meta_learner': self.meta_learner, 'task_models': {}}
        self.meta_optimizer = torch.optim.AdamW(self.meta_learner.parameters(), lr=cl_cfg.get('meta_optimizer_lr', 0.0001))

        self.performance_tracker: Dict[str, Any] = {
            'interaction_count': 0, 'success_count': 0,
            'task_performance': defaultdict(lambda: {'success': 0, 'total': 0}),
            'model_versions': defaultdict(int), 
            'learning_history': deque(maxlen=10000)
        }
        
        asyncio.create_task(self.load_models())
        self.logger.info("AdvancedSelfImprovingEngine initialized.")
        
    async def learn_from_interaction(self, interaction_data: Dict[str, Any], response_data: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None):
        self.performance_tracker['interaction_count'] += 1
        success_metrics = self._compute_success_metrics(interaction_data, response_data, feedback)
        if success_metrics.get('overall_success', 0.0) > 0.7: self.performance_tracker['success_count'] += 1
            
        await self.pattern_recognizer.record_pattern(interaction_data, response_data, success_metrics)
        
        state = self._extract_state(interaction_data)
        action = self._extract_action(response_data)
        reward = success_metrics.get('overall_success', 0.5)
        next_state = self._extract_next_state(interaction_data, response_data)
        
        await self.continuous_learner.add_experience(state, action, reward, next_state, 
            metadata={'success_metrics': success_metrics, 'feedback': feedback, 'timestamp': datetime.utcnow()})
        
        current_interaction_count = self.performance_tracker['interaction_count']
        if self.learning_interval > 0 and current_interaction_count % self.learning_interval == 0:
            await self._perform_learning_cycle()
        if self.meta_learning_interval > 0 and current_interaction_count % self.meta_learning_interval == 0:
            await self._perform_meta_learning()
        if self.save_interval > 0 and current_interaction_count % self.save_interval == 0:
            await self.save_models()
            
    async def _perform_learning_cycle(self):
        self.logger.info(f"Starting learning cycle at interaction {self.performance_tracker['interaction_count']}")
        batch = self.continuous_learner.get_learning_batch(self.batch_size, include_rehearsal=True)
        if not batch: self.logger.info("Learning cycle skipped: no batch data."); return
            
        valid_batch = [exp for exp in batch if exp.get('state', {}).get('embedding') is not None and len(exp['state']['embedding']) == self.meta_learner.input_dim]
        if not valid_batch: self.logger.info("Learning cycle skipped: no valid embeddings in batch or dimension mismatch."); return

        states_list = [torch.tensor(exp['state']['embedding'], dtype=torch.float32) for exp in valid_batch]
        states_tensor = torch.stack(states_list).unsqueeze(1)

        rewards_list = [exp.get('reward', 0.5) for exp in valid_batch]
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(states_tensor.device)

        task_ids_list = [self._identify_task_type(exp.get('state', {})) for exp in valid_batch]
        task_ids_tensor = torch.tensor(task_ids_list, dtype=torch.long).to(states_tensor.device)
        
        self.meta_learner.train()
        self.meta_optimizer.zero_grad()
        
        adapted_repr, predicted_perf, strategy_weights = self.meta_learner(states_tensor, task_ids_tensor)
        
        perf_loss = F.mse_loss(predicted_perf.squeeze(-1), rewards_tensor)
        
        best_strategies_indices = self._compute_best_strategies(valid_batch)
        if not best_strategies_indices: 
             self.logger.warning("No best strategies computed for batch. Skipping strategy loss.")
             strategy_loss = torch.tensor(0.0, device=strategy_weights.device) 
        else:
            best_strategies_tensor = torch.tensor(best_strategies_indices, dtype=torch.long, device=strategy_weights.device)
            if strategy_weights.shape[0] != best_strategies_tensor.shape[0]:
                self.logger.error(f"Shape mismatch for strategy loss: weights {strategy_weights.shape}, targets {best_strategies_tensor.shape}")
                strategy_loss = torch.tensor(0.0, device=strategy_weights.device)
            else:
                 strategy_loss = F.cross_entropy(strategy_weights, best_strategies_tensor)
        
        total_loss = perf_loss + 0.5 * strategy_loss
        total_loss.backward()
        self.meta_optimizer.step()
        
        avg_actual_perf = rewards_tensor.mean().item()
        self.continuous_learner.update_learning_rate({'overall_performance': avg_actual_perf})
        
        self.performance_tracker['learning_history'].append({
            'timestamp': datetime.utcnow(), 'loss': total_loss.item(), 
            'avg_reward_in_batch': avg_actual_perf, 
            'learning_rate': self.continuous_learner.current_learning_rate
        })
        self.logger.info(f"Learning cycle complete. Loss: {total_loss.item():.4f}, Avg Reward: {avg_actual_perf:.4f}")
        
    async def _perform_meta_learning(self):
        self.logger.info("Starting meta-learning update (strategy refinement)")
        if len(self.performance_tracker['learning_history']) < 50: return

        recent_rewards = [h['avg_reward_in_batch'] for h in list(self.performance_tracker['learning_history'])[-50:] if 'avg_reward_in_batch' in h]
        if len(recent_rewards) < 2 : return

        avg_recent_reward_improvement = 0.0
        if len(set(recent_rewards)) > 1: 
            try:
                avg_recent_reward_improvement = np.mean(np.diff(recent_rewards)) 
            except Exception as e_lin:
                 self.logger.warning(f"Could not compute reward improvement trend: {e_lin}")

        current_meta_lr = self.meta_optimizer.param_groups[0]['lr']
        new_meta_lr = current_meta_lr
        
        if avg_recent_reward_improvement > 0.001: 
            new_meta_lr = min(current_meta_lr * 1.05, 0.01)
        elif avg_recent_reward_improvement < -0.001: 
            new_meta_lr = max(current_meta_lr * 0.95, 1e-6)
        
        if abs(new_meta_lr - current_meta_lr) > 1e-7 : 
             self.logger.info(f"Meta-learning: Adjusting meta_optimizer LR from {current_meta_lr:.6f} to {new_meta_lr:.6f}")
             for param_group in self.meta_optimizer.param_groups:
                 param_group['lr'] = new_meta_lr
            
    async def predict_optimal_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pattern_recs = self.pattern_recognizer.get_pattern_recommendations(context)
        state = self._extract_state(context.get('interaction', {}))
        if 'embedding' not in state or state['embedding'] is None or len(state['embedding']) != self.meta_learner.input_dim:
            self.logger.warning(f"Missing or invalid embedding for optimal response prediction. State embedding length: {len(state.get('embedding',[]))}")
            return {'strategy': 'default', 'confidence': 0.5, 'reason': 'Missing/invalid embedding in context'}
            
        with torch.no_grad():
            self.meta_learner.eval() 
            state_tensor = torch.tensor(state['embedding'], dtype=torch.float32).unsqueeze(0) 
            task_id = self._identify_task_type(context.get('interaction', {})) 
            
            _, predicted_perf, strategy_weights = self.meta_learner(state_tensor, task_id)
            
        best_strategy_idx = strategy_weights.squeeze().argmax().item()
        strategy_confidence = strategy_weights.squeeze()[best_strategy_idx].item()
        strategy_names = ['analytical', 'creative', 'empathetic', 'comprehensive', 'concise'] 
        
        return {
            'recommended_strategy': strategy_names[best_strategy_idx] if best_strategy_idx < len(strategy_names) else 'unknown_strategy',
            'confidence': strategy_confidence,
            'predicted_performance_with_strategy': predicted_perf.item(),
            'pattern_based_recommendations': pattern_recs,
            'adaptation_suggestions': self._get_adaptation_suggestions(context)
        }
        
    def _compute_success_metrics(self, interaction_data: Dict[str, Any], response_data: Dict[str, Any], feedback: Optional[Dict[str, Any]]) -> Dict[str, float]:
        metrics = {}
        feedback = feedback or {}
        metrics['user_satisfaction'] = float(feedback.get('rating', 0.5))
        metrics['relevance'] = float(feedback.get('relevance', response_data.get('confidence', 0.5)))
        metrics['helpfulness'] = float(feedback.get('helpfulness', 0.5))
        metrics['response_time_score'] = self._compute_time_score(float(response_data.get('processing_time', 1.0)))
        metrics['coherence_score'] = float(response_data.get('coherence', 0.7))
        metrics['completeness_score'] = self._compute_completeness_score(interaction_data, response_data)
        
        weights = {'user_satisfaction': 0.3, 'relevance': 0.25, 'helpfulness': 0.2, 'response_time_score': 0.1, 'coherence_score': 0.1, 'completeness_score': 0.05}
        overall_success = sum(float(metrics.get(k, 0.5)) * v for k, v in weights.items())
        metrics['overall_success'] = float(np.clip(overall_success, 0.0, 1.0))
        return metrics
        
    def _compute_time_score(self, processing_time: float) -> float:
        if processing_time < 0.5: return 0.8
        elif processing_time < 2.0: return 1.0
        elif processing_time < 5.0: return 0.8 - (processing_time - 2.0) * 0.1 
        else: return max(0.3, 0.5 - (processing_time - 5.0) * 0.05)
            
    def _compute_completeness_score(self, interaction_data: Dict[str, Any], response_data: Dict[str, Any]) -> float:
        query_aspects_count = len(interaction_data.get('expected_aspects', [])) 
        response_aspects_count = len(response_data.get('addressed_aspects', [])) 
        if query_aspects_count == 0: return 0.75 
        return min(response_aspects_count / query_aspects_count, 1.0) if query_aspects_count > 0 else 0.5
        
    def _extract_state(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        embedding_source = interaction_data.get('embedding', interaction_data.get('state_vector'))
        final_embedding = None
        if isinstance(embedding_source, np.ndarray):
            final_embedding = embedding_source.flatten().tolist()
        elif isinstance(embedding_source, list):
            if embedding_source and isinstance(embedding_source[0], list):
                try: final_embedding = np.mean(np.array(embedding_source, dtype=float), axis=0).tolist()
                except: final_embedding = None 
            else: 
                final_embedding = embedding_source
        
        return {
            'embedding': final_embedding, 
            'context_length': interaction_data.get('context_length', 0),
            'complexity': interaction_data.get('complexity', self._estimate_complexity(interaction_data)),
            'emotional_state': interaction_data.get('emotional_state', 'neutral')
        }
        
    def _extract_action(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        minds_contributions = response_data.get('minds_contributions', {})
        minds_used = list(minds_contributions.keys()) if isinstance(minds_contributions, dict) else []
        return {
            'strategy': response_data.get('response_strategy_used', 0), 
            'length': len(str(response_data.get('text', response_data.get('content', '')))),
            'minds_used': minds_used,
            'confidence': response_data.get('overall_confidence', 0.5)
        }
        
    def _extract_next_state(self, interaction_data: Dict[str, Any], response_data: Dict[str, Any]) -> Dict[str, Any]:
        response_embedding_source = response_data.get('embedding', response_data.get('state_vector'))
        final_response_embedding = None
        if isinstance(response_embedding_source, np.ndarray):
            final_response_embedding = response_embedding_source.flatten().tolist()
        elif isinstance(response_embedding_source, list):
             if response_embedding_source and isinstance(response_embedding_source[0], list):
                try: final_response_embedding = np.mean(np.array(response_embedding_source, dtype=float), axis=0).tolist()
                except: final_response_embedding = None
             else:
                final_response_embedding = response_embedding_source

        return {
            'embedding': final_response_embedding,
            'context_length': interaction_data.get('context_length', 0) + len(str(response_data.get('text',''))),
            'user_state': 'responded' 
        }
        
    def _identify_task_type(self, state_data: Dict[str, Any]) -> int:
        complexity = state_data.get('complexity', 'medium')
        emotional_tone = state_data.get('emotional_state', 'neutral')
        task_hash = hash(f"{complexity}_{emotional_tone}")
        return abs(task_hash) % self.meta_learner.num_tasks 
        
    def _compute_best_strategies(self, batch: List[Dict[str, Any]]) -> List[int]:
        if not batch: return []
        num_strategies = self.meta_learner.strategy_network[-2].out_features if hasattr(self.meta_learner, 'strategy_network') and hasattr(self.meta_learner.strategy_network[-2], 'out_features') else 5
        return [(exp['action'].get('strategy',0) if exp.get('reward',0.0) > 0.7 else (exp['action'].get('strategy',0) + 1) % num_strategies) for exp in batch]
        
    def _get_adaptation_suggestions(self, context: Dict[str, Any]) -> List[str]:
        suggestions = []
        interaction_context = context.get('interaction',{})
        complexity = interaction_context.get('complexity', self._estimate_complexity(interaction_context))
        emotional_tone = interaction_context.get('emotional_tone', 'neutral')
        if complexity == 'complex': suggestions.extend(["Break down response.", "Use analytical mind more."])
        if emotional_tone in ['frustrated', 'confused', 'angry', 'sad']: suggestions.extend(["Increase empathy.", "Provide clearer, reassuring explanations."])
        return suggestions if suggestions else ["Maintain balanced approach."]
        
    async def save_models(self):
        self.logger.info(f"Saving self-improving engine models to {self.model_dir}...")
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True) 

            meta_path = self.model_dir / 'meta_learner_state.pt'
            perf_tracker_serializable = {k: (list(v) if isinstance(v, deque) else dict(v) if isinstance(v, defaultdict) else v) for k,v in self.performance_tracker.items()}
            model_versions_serializable = dict(self.performance_tracker['model_versions']) # ensure dict
            perf_tracker_serializable['model_versions'] = model_versions_serializable
            
            torch.save({
                'model_state_dict': self.meta_learner.state_dict(),
                'optimizer_state_dict': self.meta_optimizer.state_dict(),
                'performance_tracker': perf_tracker_serializable,
                'timestamp': datetime.utcnow().isoformat()
            }, meta_path)
            
            pattern_path = self.model_dir / 'pattern_recognizer_state.pkl'
            with open(pattern_path, 'wb') as f:
                pickle.dump({
                    'pattern_frequencies': dict(self.pattern_recognizer.pattern_frequencies),
                    'pattern_success_rates': {k: dict(v) for k, v in self.pattern_recognizer.pattern_success_rates.items()}, 
                    'pattern_transitions': {k: dict(v) for k, v in self.pattern_recognizer.pattern_transitions.items()}
                }, f)
                
            learner_path = self.model_dir / 'continuous_learner_state.pkl'
            with open(learner_path, 'wb') as f:
                cl_stats_serializable = {k:(list(v) if isinstance(v, deque) else v) for k,v in self.continuous_learner.learning_stats.items()}
                pickle.dump({
                    'rehearsal_memory': self.continuous_learner.rehearsal_memory[:self.config.get('rehearsal_save_sample_size', 500)], 
                    'learning_stats': cl_stats_serializable, 
                    'current_learning_rate': self.continuous_learner.current_learning_rate
                }, f)
            self.logger.info(f"Self-improving engine models saved successfully to {self.model_dir}.")
        except Exception as e:
            self.logger.error(f"Error saving self-improving engine models: {e}", exc_info=True)
            
    async def load_models(self):
        self.logger.info(f"Attempting to load self-improving engine models from {self.model_dir}...")
        try:
            meta_path = self.model_dir / 'meta_learner_state.pt'
            if meta_path.exists():
                checkpoint = torch.load(meta_path, map_location=torch.device('cpu')) 
                self.meta_learner.load_state_dict(checkpoint['model_state_dict'])
                try: self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as optim_e: self.logger.warning(f"Could not load meta_optimizer state: {optim_e}. Reinitializing optimizer.")
                
                perf_tracker_loaded = checkpoint.get('performance_tracker', {})
                self.performance_tracker['interaction_count'] = perf_tracker_loaded.get('interaction_count',0)
                self.performance_tracker['success_count'] = perf_tracker_loaded.get('success_count',0)
                self.performance_tracker['task_performance'] = defaultdict(lambda: {'success':0,'total':0}, perf_tracker_loaded.get('task_performance',{}))
                self.performance_tracker['model_versions'] = defaultdict(int, checkpoint.get('model_versions', {})) 
                self.performance_tracker['learning_history'] = deque(perf_tracker_loaded.get('learning_history',[]), maxlen=self.performance_tracker['learning_history'].maxlen)
                self.logger.info(f"Loaded meta-learner from checkpoint dated {checkpoint.get('timestamp')}")

            pattern_path = self.model_dir / 'pattern_recognizer_state.pkl'
            if pattern_path.exists():
                with open(pattern_path, 'rb') as f: pattern_data = pickle.load(f)
                self.pattern_recognizer.pattern_frequencies = defaultdict(int, pattern_data.get('pattern_frequencies',{}))
                self.pattern_recognizer.pattern_success_rates = defaultdict(lambda: {'success':0,'total':0}, pattern_data.get('pattern_success_rates',{}))
                self.pattern_recognizer.pattern_transitions = defaultdict(lambda: defaultdict(float), pattern_data.get('pattern_transitions',{}))
                self.logger.info("Loaded pattern recognizer data.")

            learner_path = self.model_dir / 'continuous_learner_state.pkl'
            if learner_path.exists():
                with open(learner_path, 'rb') as f: learner_data = pickle.load(f)
                self.continuous_learner.rehearsal_memory = learner_data.get('rehearsal_memory',[])
                cl_stats_loaded = learner_data.get('learning_stats',{})
                self.continuous_learner.learning_stats['total_experiences'] = cl_stats_loaded.get('total_experiences',0)
                self.continuous_learner.learning_stats['rehearsal_updates'] = cl_stats_loaded.get('rehearsal_updates',0)
                self.continuous_learner.learning_stats['performance_history'] = deque(cl_stats_loaded.get('performance_history',[]), maxlen=self.continuous_learner.learning_stats['performance_history'].maxlen)
                self.continuous_learner.learning_stats['learning_rate_history'] = deque(cl_stats_loaded.get('learning_rate_history',[]), maxlen=self.continuous_learner.learning_stats['learning_rate_history'].maxlen)
                self.continuous_learner.current_learning_rate = learner_data.get('current_learning_rate', self.continuous_learner.base_learning_rate)
                self.logger.info("Loaded continuous learner state.")
        except FileNotFoundError:
            self.logger.info(f"No saved models found at {self.model_dir}. Starting with a fresh self-improving engine.")
        except Exception as e:
            self.logger.error(f"Error loading self-improving engine models from {self.model_dir}: {e}. Starting fresh.", exc_info=True)
            
    def get_performance_report(self) -> Dict[str, Any]:
        total_interactions = self.performance_tracker['interaction_count']
        if total_interactions == 0: return {'status': 'No interactions recorded yet.'}
        success_rate = self.performance_tracker['success_count'] / total_interactions if total_interactions > 0 else 0.0
        
        recent_performance_values = [h.get('performance', h.get('avg_reward_in_batch', 0.0)) for h in list(self.performance_tracker['learning_history'])[-10:] if isinstance(h,dict)] 
        avg_recent_performance = np.mean(recent_performance_values) if recent_performance_values else 0.0

        return {
            'total_interactions': total_interactions, 'success_rate': success_rate,
            'learning_progress': {
                'total_learning_cycles': len(self.performance_tracker['learning_history']),
                'current_learning_rate': self.continuous_learner.current_learning_rate,
                'recent_avg_performance_metric': avg_recent_performance
            },
            'pattern_insights': {
                'unique_patterns_tracked': len(self.pattern_recognizer.pattern_frequencies),
                'most_successful_patterns': self._get_top_patterns(self.config.get('report_top_patterns', 5)), 
                'pattern_coverage_estimate': self._compute_pattern_coverage()
            },
            'model_status': {
                'meta_learner_version': self.performance_tracker['model_versions'].get('meta_learner', 0),
                'last_model_save_timestamp': self._get_last_save_time(), 
                'estimated_component_memory_usage': self._estimate_memory_usage()
            }
        }
        
    def _get_top_patterns(self, n: int) -> List[Dict[str, Any]]:
        pattern_scores = []
        for pattern_key, stats_dict in self.pattern_recognizer.pattern_success_rates.items():
            total = stats_dict.get('total', 0)
            if total > 0:
                success = stats_dict.get('success', 0)
                success_rate = success / total
                frequency = self.pattern_recognizer.pattern_frequencies.get(pattern_key, 0)
                score = success_rate * np.log1p(frequency) 
                pattern_scores.append({'pattern': pattern_key, 'success_rate': success_rate, 'frequency': frequency, 'score': score})
        pattern_scores.sort(key=lambda x: x['score'], reverse=True)
        return pattern_scores[:n]
        
    def _compute_pattern_coverage(self) -> float:
        if not self.pattern_recognizer.pattern_frequencies: return 0.0
        total_unique_patterns = len(self.pattern_recognizer.pattern_frequencies)
        total_pattern_observations = sum(self.pattern_recognizer.pattern_frequencies.values())
        diversity_score = min(total_unique_patterns / self.config.get('coverage_target_unique_patterns', 500.0), 1.0)
        observation_score = min(total_pattern_observations / self.config.get('coverage_target_observations', 5000.0), 1.0)
        return (diversity_score * 0.6 + observation_score * 0.4)
        
    def _get_last_save_time(self) -> Optional[str]:
        meta_path = self.model_dir / 'meta_learner_state.pt'
        if meta_path.exists(): 
            try: return datetime.fromtimestamp(meta_path.stat().st_mtime).isoformat()
            except Exception as e_ts: self.logger.warning(f"Could not get timestamp for model file {meta_path}: {e_ts}"); return None
        return None
        
    def _estimate_memory_usage(self) -> Dict[str, str]: 
        try:
            exp_buffer_size_mb = sys.getsizeof(self.continuous_learner.experience_buffer) / (1024*1024)
            rehearsal_mem_size_mb = sum(sys.getsizeof(exp) for exp in self.continuous_learner.rehearsal_memory if isinstance(exp, dict)) / (1024*1024) 
            pattern_mem_size_mb = (sys.getsizeof(self.pattern_recognizer.interaction_patterns) + 
                                   sys.getsizeof(self.pattern_recognizer.pattern_frequencies)) / (1024*1024)
            meta_learner_params = sum(p.numel() for p in self.meta_learner.parameters())
            meta_learner_size_mb = (meta_learner_params * 4) / (1024*1024) 

            return {
                "experience_buffer_mb": f"{exp_buffer_size_mb:.2f}",
                "rehearsal_memory_mb": f"{rehearsal_mem_size_mb:.2f}",
                "pattern_recognizer_mb": f"{pattern_mem_size_mb:.2f}",
                "meta_learner_model_mb": f"{meta_learner_size_mb:.2f}"
            }
        except Exception as e:
            self.logger.error(f"Error estimating memory usage for ASIE: {e}")
            return {key: "Error" for key in ["experience_buffer_mb", "rehearsal_memory_mb", "pattern_recognizer_mb", "meta_learner_model_mb"]}