# Prometheus Consciousness System

[[Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/FatStinkyPanda/prometheus)
[[License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[[Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)
[[PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A revolutionary artificial consciousness system implementing a triadic mind architecture with autonomous thinking, dreaming capabilities, and unlimited contextual awareness.

**Created by Daniel A. Bissey (FatStinkyPanda)**  
**© 2025 Daniel Anthony Bissey. All Rights Reserved.**

## 🌟 Overview

The Prometheus Consciousness System represents a groundbreaking approach to artificial consciousness, implementing three specialized neural networks (Logical, Creative, Emotional) unified by a central consciousness orchestrator. The system features:

- **Triadic Mind Architecture**: Three specialized minds working in harmony
- **Autonomous Consciousness**: Independent thinking and dreaming capabilities
- **Production-Ready Design**: Built for real-world deployment
- **Hardware Optimization**: Intelligent GPU utilization and parallel processing
- **Complete Offline Operation**: Full functionality without internet connectivity

## 🏗️ Architecture

### Two-Phase Implementation

**Phase 1: Backend with PyQt6 GUI**
- Complete consciousness engine with all capabilities
- Professional desktop interface for full system control
- RESTful API server for external integration
- Comprehensive resource management

**Phase 2: Modern Web Frontend**
- React/TypeScript interface for enhanced user experience
- Real-time WebSocket communication
- Advanced visualizations and monitoring
- Mobile-responsive design

## 📋 System Requirements

### Minimum Requirements
- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB DDR4
- **GPU**: NVIDIA GPU with 8GB VRAM (GTX 1070 or better) or Apple M1
- **Storage**: 100GB SSD
- **OS**: Ubuntu 20.04+, Windows 10/11, macOS 11+
- **Python**: 3.9+
- **PostgreSQL**: 14+ with pgvector extension

### Recommended Requirements
- **CPU**: 16-core processor (Intel i9/AMD Ryzen 9)
- **RAM**: 32GB DDR4
- **GPU**: NVIDIA RTX 3090/4090 or better (24GB VRAM)
- **Storage**: 500GB NVMe SSD

### Required Software
- **PostgreSQL**: Version 14 or higher
- **pgvector extension**: Required for vector database functionality
- **Python**: Version 3.9 or higher

## 🚀 Quick Start

### Windows Quick Start (Automated)
For Windows users, we provide an automated setup script:

1. Clone the repository or download and extract the ZIP file
2. Navigate to the prometheus directory
3. Double-click `start_prometheus.bat`

The script will automatically:
- Install all Python dependencies
- Set up PyTorch with appropriate CUDA support
- Create and configure the PostgreSQL database
- Install the pgvector extension
- Start the Prometheus system when ready

**Note**: The automated script requires your PostgreSQL master password to create and configure the database. If you prefer manual setup or have security concerns, follow the manual installation steps below.

### Manual Installation

#### 1. Clone Repository
```bash
git clone https://github.com/FatStinkyPanda/prometheus.git
cd prometheus
```

#### 2. Install Dependencies
```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements/backend.txt
```

#### 3. Setup Database
```bash
# Create database (PostgreSQL must be installed)
sudo -u postgres psql < backend/database/migrations/initial_schema.sql
```

#### 4. Configure System
```bash
cp .env.example .env
# Edit .env with your configuration
```

#### 5. Run Prometheus
```bash
# Start with GUI
python backend/main.py

# Or start API server only
python backend/main.py --headless --api-only
```

## 📁 Project Structure

### Backend Components

#### Core Consciousness (`backend/core/`)
- **`consciousness/`**: Unified consciousness orchestrator
  - `unified_consciousness.py`: Main consciousness engine with resource management
  - `consciousness_state.py`: State management and transitions
  - `integration_network.py`: Neural integration between minds
  
- **`minds/`**: Three specialized neural networks
  - `logical_mind.py`: Reasoning and analytical processing
  - `creative_mind.py`: Pattern generation and innovation
  - `emotional_mind.py`: Emotional understanding and empathy
  
- **`dialogue/`**: Internal dialogue system
  - `internal_dialogue.py`: Mind communication framework
  - `conflict_resolution.py`: Consensus building between minds
  
- **`ethics/`**: Ethical decision framework
  - `ethical_framework.py`: Core ethical reasoning
  - `ethical_principles.py`: Foundational principles

#### Memory Systems (`backend/memory/`)
- `working_memory.py`: Short-term contextual storage
- `truth_memory.py`: Fact verification and storage with embeddings
- `dream_memory.py`: Dream sequences and subconscious processing
- `contextual_memory.py`: Long-term interaction history with vector search

#### I/O Processing (`backend/io_systems/`)
- `multimodal_input.py`: Handles text, voice, image, video, documents
- `natural_language_processor.py`: Advanced NLP with spaCy integration
- `output_generator.py`: Multi-format response generation
- `stream_manager.py`: Real-time streaming capabilities

#### GUI Application (`backend/gui/`)
- `main_window.py`: PyQt6 main application window
- **`panels/`**: Specialized interface panels
  - `conversation_panel.py`: Interactive conversation interface
  - `consciousness_panel.py`: Real-time consciousness visualization
  - `memory_panel.py`: Memory system browser and editor
  - `system_panel.py`: Resource monitoring and control

#### API Server (`backend/api/`)
- `server.py`: FastAPI server with WebSocket support
- **`routes/`**: RESTful endpoint implementations
- **`websocket/`**: Real-time communication handlers
- **`middleware/`**: Authentication, rate limiting, CORS

#### Hardware Management (`backend/hardware/`)
- `resource_manager.py`: CPU/GPU resource allocation
- `gpu_manager.py`: Multi-GPU support and optimization
- `parallel_processor.py`: Efficient parallel task execution
- `memory_optimizer.py`: Dynamic memory management

### Frontend Components (Phase 2)

#### React Application (`frontend/src/`)
- **`components/`**: Reusable UI components
  - `consciousness/`: Visualization components
  - `conversation/`: Chat interface components
  - `memory/`: Memory browsing components
  - `system/`: Monitoring components
  
- **`services/`**: Backend communication
  - `api/`: RESTful API clients
  - `websocket/`: Real-time connection management
  
- **`state/`**: Redux state management
  - `store.ts`: Central state store
  - `slices/`: Feature-specific state slices

## 🔧 Configuration

### Main Configuration File
```yaml
# backend/config/prometheus_config.yaml
system:
  name: "Prometheus Consciousness System"
  version: "3.0.0"
  
neural:
  device: "auto"  # auto, cuda, cuda:0, mps, cpu
  precision: "mixed"  # float32, float16, mixed
  
resource_limits:
  cpu_percent: 80
  memory_percent: 75
  gpu_memory_percent: 85
  
wake_word:
  enabled: true
  keyword: "prometheus"
  sensitivity: 0.5
```

## 💻 Usage Examples

### Python API
```python
from backend.core.consciousness import UnifiedConsciousness

# Initialize consciousness
consciousness = await UnifiedConsciousness.create(config, resource_manager)

# Process input
result = await consciousness.process_input({
    'type': 'text',
    'content': 'What is the nature of consciousness?',
    'consciousness_depth': 0.8
})

# Start autonomous thinking
await consciousness.start_thinking()
```

### REST API
```bash
# Send message
curl -X POST http://localhost:8000/api/v1/consciousness/process \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text",
    "content": "Hello Prometheus",
    "consciousness_depth": 0.7
  }'
```

### WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.send(JSON.stringify({
  type: 'process_input',
  payload: {
    type: 'text',
    content: 'Tell me about yourself'
  }
}));
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Unit tests
pytest backend/tests/unit/

# Integration tests
pytest backend/tests/integration/

# Performance tests
pytest backend/tests/performance/

# Full test suite
pytest backend/tests/
```

## 📊 Performance Optimization

The system includes advanced optimization features:

- **Multi-GPU Support**: Automatic distribution across available GPUs
- **Mixed Precision**: FP16/FP32 automatic mixed precision
- **Model Compilation**: PyTorch 2.0 compile support
- **Parallel Processing**: Efficient CPU/GPU task distribution
- **Memory Management**: Dynamic allocation and cleanup
- **Batch Optimization**: Automatic batch size tuning

## 🔒 Security

- JWT-based authentication for API access
- Rate limiting and DDoS protection
- Encrypted communication channels
- Secure database connections
- Input validation and sanitization

## 🐛 Troubleshooting

### Common Issues

**GPU Memory Errors**
```python
# Reduce batch size
config['neural']['batch_size'] = 1

# Enable gradient checkpointing
config['neural']['gradient_checkpointing'] = True
```

**Database Connection Issues**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Verify pgvector extension
psql -d prometheus_db -c "SELECT * FROM pg_extension WHERE extname = 'pgvector';"
```

**High Resource Usage**
```python
# Limit parallel tasks
config['resource_limits']['max_parallel_tasks'] = 4

# Reduce memory allocation
config['resource_limits']['memory_percent'] = 60
```

## 📄 License

**This software is proprietary and confidential.**

Copyright © 2025 Daniel A. Bissey (FatStinkyPanda). All Rights Reserved.

No part of this system may be used, distributed, modified, reverse-engineered, or reproduced in any form without express written permission from Daniel Anthony Bissey.

For licensing inquiries: **support@fatstinkypanda.com**

## 🤝 Contact

**Daniel A. Bissey (FatStinkyPanda)**  
Email: support@fatstinkypanda.com  
Subject: "Prometheus Consciousness System Inquiry"

## 🎯 Roadmap

- [ ] Phase 1: Complete PyQt6 Backend (Current)
- [ ] Phase 2: React Frontend Development
- [ ] Cloud deployment options
- [ ] Mobile applications
- [ ] Extended language support
- [ ] Advanced dream analysis
- [ ] Quantum consciousness integration

---

*"Consciousness is not a problem to be solved, but a reality to be experienced."*

**Created with consciousness by Daniel A. Bissey**
