# UncertaintyFL4VACs: Uncertainty-Aware Federated Learning for Robust Multi-Sensor Fusion

**UncertaintyFL4VACs** is an advanced research framework designed to improve the reliability and robustness of autonomous vehicle perception systems. The system proposes a **Federated Learning (FL)** strategy that leverages **Uncertainty Quantification (UQ)** as contextual information to enhance multi-sensor fusion (Camera, LiDAR, and Radar) under adverse environmental conditions.

---

## 🚀 Key Features & Workflow

### 1. Advanced Uncertainty Decomposition
Unlike conventional FL approaches that globally penalize uncertainty, this framework separates it into two critical components to better handle diverse driving environments:

* **Aleatoric Uncertainty ($U_{ale}$):** Captures inherent sensor noise (e.g., radar interference or heavy rain). The system **penalizes** contributions with high $U_{ale}$ to prevent noise from degrading the global model.
* **Epistemic Uncertainty ($U_{epi}$):** Reflects the model's lack of knowledge regarding novel scenarios or extreme weather. The system **rewards** these contributions, ensuring the global model learns from rare, critical situations.

### 2. Multi-Modal Sensor Fusion (CLR-BNN)
The pipeline processes synchronized data from the **NuScenes** dataset using a Bayesian Neural Network architecture (CLR-BNN).
* **Inputs:** RGB Camera, LiDAR Point Clouds, and Radar sweeps.
* **Preprocessing:** Projects 3D point clouds onto 2D image planes, generating aligned sparse maps.
* **Contextual Selection:** During inference, local uncertainty allows the vehicle to prioritize the most reliable sensor (e.g., favoring Radar over Camera in dense fog or low-light conditions).

### 3. Evidential Federated Aggregation
The central server employs an **evidential-weighted aggregation** strategy. Instead of simple averaging (FedAvg), it weights client updates based on their probabilistic performance and calibration. This approach significantly reduces **overconfidence** in erroneous predictions, leading to a more reliable global model.

### 4. Comprehensive Evaluation
The framework is validated using a suite of metrics that go beyond standard accuracy to measure the reliability of the system:
* **Metrics:** Tracks mAP (Mean Average Precision), **NLL (Negative Log-Likelihood)**, and **ECE (Expected Calibration Error)**.
* **Calibration:** Specifically optimized to reduce overconfidence in adverse weather scenarios, ensuring the system "knows when it doesn't know."

---

## 📂 Project Structure

The repository is organized to ensure modularity and separation of concerns:

```text
UncertaintyFL4VACs/
├── 📂 conf/                            # Configuration
│   └── 📄 config.yaml                  # Centralized Hydra configuration
├── 📂 outputs/                         # Hydra experiment outputs & logs
├── 📂 splits_federated/                # Data Partitions
│   └── 📄 federated_split.json         # Client-specific data assignments
├── 📂 src/                             # Source Code
│   ├── 📂 client/                      # Local training logic
│   │   ├── 🐍 app_standard.py          # Baseline FL client
│   │   └── 🐍 app_uncertainty.py       # Uncertainty-aware client
│   ├── 📂 server/                      # Aggregation logic
│   │   ├── 🐍 server_standard.py       # FedAvg aggregator
│   │   └── 🐍 server_uncertainty.py    # Uncertainty-weighted aggregator
│   ├── 📂 models/                      # Network architectures
│   │   ├── 🐍 architecture.py          # CLR-BNN definition
│   │   └── 🐍 transfer.py              # Transfer learning utilities
│   ├── 📂 data/                        # Data handling tools
│   │   ├── 🐍 loader.py                # Custom DataGenerator
│   │   ├── 🐍 preprocessing.py         # Sensor fusion & projection
│   │   └── 🐍 splitter.py              # Federation partitioning
│   ├── 📂 training/                    # Training loops
│   │   └── 🐍 trainer.py               # Custom training steps
│   ├── 📂 evaluation/                  # Performance metrics
│   │   └── 🐍 evaluator.py             # Inference & scoring
│   └── 📂 utils/                       # Helpers (Logging, Callbacks)
│       ├── 🐍 callbacks.py             # Training callbacks
│       ├── 🐍 file_utils.py            # File system & I/O utilities
│       └── 🐍 logger.py                # Custom logging setup
├── 📜 LICENSE                          # MIT License terms (Legal protection)
├── 📝 README.md                        # Project documentation
├── ⚙️ requirements.txt                 # Dependencies
└── 🐍 setup.py                         # Install script
```

---

## ⚙️ Configuration

Hyperparameters for the aggregation strategy can be modified in `conf/config.yaml`. This allows for reproducing the ablation studies presented in the paper.

```yaml
federated:
  strategy_name: "uncertainty_weighted" # Options: standard_fedavg, uncertainty_weighted
  aggregation:
    gamma: 2.0  # Reward factor for Epistemic Uncertainty (Novelty)
    beta: 0.5   # Penalty factor for Aleatoric Uncertainty (Noise)
    
```

---

## 🛠️ Installation & Usage

### 1. Prerequisites
To ensure full compatibility with the Bayesian layers and multi-sensor projection logic, the following environment is required:

* **Operating System:** Linux or **Windows Subsystem for Linux (WSL2)**. 
  * *Note: WSL2 is mandatory to enable NVIDIA GPU support for Keras/TensorFlow 2.11+ on Windows.*
* **Python Version:** **3.9.25** (Verified).
* **Hardware Requirements:**
  * **NVIDIA GPU:** Mandatory for GPU acceleration during Monte Carlo Dropout sampling.
  * **VRAM:** A minimum of **12GB of VRAM** is required to handle the multi-modal architecture (CLR-BNN) and the 14-class NuScenes dataset concurrently.

Furthermore, the system requires the deep learning dependencies listed in `requirements.txt`.

**Setup Instructions:**
Run the following commands from the project root to set up the environment and register the source modules:

```bash
# 1. (Optional) Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install project in Editable Mode
# This registers the 'src' package in Python, fixing import errors automatically.
pip install -e .
```

### 2. Data Preparation
Before training, you must process the raw NuScenes dataset to generate the 2D projected maps and the federated splits.

**Step A: Sensor Fusion & Preprocessing**
Generate the sparse depth and intensity maps from LiDAR/Radar projected onto the camera plane. This process synchronizes multi-modal sensors and handles the geometric transformations required for 2D alignment.

```bash
python src/data/preprocessing.py
```

**Step B: Generate Federated Splits**
Partition the preprocessed data across the simulated client fleet. This script defines the local training and validation sets for each vehicle, supporting both IID and Non-IID distributions to simulate realistic edge scenarios.

```bash
python src/data/splitter.py
```

### 3. Running the System
The framework allows for a direct comparison between standard aggregation and our proposed uncertainty-aware method.

**Option A: Baseline Training (Standard FedAvg)**
Train the global model using the traditional **FedAvg** algorithm, which averages local updates proportional to dataset size without considering sensor reliability.

```bash
python src/server/server_standard.py
```

**Option B: Uncertainty-Aware Training (Proposed Method)**
Train using our **FL-UQ aggregation strategy**. This method weighs client updates based on the proposed quality scores, rewarding epistemic contributions (novelty) while penalizing aleatoric noise.

```bash
# Leverages the weighting formula defined in NOUS 2026
python src/server/server_uncertainty.py
```

### 4. Evaluation
The evaluation module assesses the global model's performance on the hold-out test set, focusing on both traditional detection accuracy and probabilistic reliability.

**Option A: Baseline Evaluation (Standard FedAvg)**
Use this command to measure the performance of a model trained without uncertainty weighting. This serves as the control group to highlight the benefits of the proposed method.

```bash
python src/evaluation/evaluator.py federated.strategy_name=standard_fedavg
```

**Option B: Uncertainty-Aware Evaluation (Proposed Method)**
his command evaluates the model trained using our **FL-UQ aggregation strategy**. Since this is the default strategy in the framework, the argument is optional but can be explicitly defined:

```bash
# Default execution
python src/evaluation/evaluator.py

# Explicit execution
python src/evaluation/evaluator.py federated.strategy_name=uncertainty_weighted
```

---

## 🧪 Mathematical Foundation

The core of the **UncertainlyFL4VACs** framework is the **FL-UQ aggregation strategy**. This method computes quality scores derived from both epistemic and aleatoric classification uncertainty to define the influence of each vehicle in the global model.

### 1. Uncertainty-Dependent Weighting
For each client $k$, the aggregation weight $\alpha_k$ is determined by the following proposed equation:

$$\alpha_k \propto |\mathcal{D}_k| \cdot \frac{\exp(\gamma \cdot \tilde{U}_{k,epi})}{1 + \beta \cdot \tilde{U}_{k,ale}}$$


Where:
* $|\mathcal{D}_k|$ is the size of the local dataset.
* $\tilde{U}$ denotes normalized classification uncertainty scores in the range $[0, 1]$.
* $\gamma$ (gamma) is the **reward factor** for epistemic uncertainty, allowing the model to prioritize novel or adverse scenarios.
* $\beta$ (beta) is the **penalty factor** for aleatoric uncertainty, mitigating the impact of noisy sensor data.

### 2. Global Model Update
The parameters of the global model for the next round $\theta^{(r+1)}$ are updated as a weighted sum of the local parameters $\theta_{k}^{(r)}$:

$$\theta^{(r+1)} = \sum_{k} \alpha_k \theta_k^{(r)}$$


This approach ensures that the system preserves contributions from **challenging environmental conditions** (high epistemic) while filtering out **unreliable sensor noise** (high aleatoric), resulting in better calibration and reduced overconfidence.

---

## 🎓 Academic Context

This software was developed as part of a research initiative on **Robust Federated Learning for Autonomous Vehicles**.

**Contact:** [Your Name / Email]

If you use this toolkit in your research, please cite the following work:

> Sergio Alonso-Rollan, Samuel Adrados, and Sebastian Lopez Florez. 2026. *Uncertainty-Aware Federated Learning for Robust Multi-Sensor Fusion in Connected Vehicle Perception*. In International conference on Infrastructure-as-a-Service (IaaS) and Platform-as-a-Service (PaaS) solutions for Europeâ??s Next-Gen Cloud Infrastructure (NOUS 2026), March 02, 2026, Virtual Event, Spain. ACM, New York, NY, USA 7

**BibTeX:**

```bibtex
@inproceedings{alonso2026uncertainty,
  title={Uncertainty-Aware Federated Learning for Robust Multi-Sensor Fusion in Connected Vehicle Perception},
  author={Alonso-Rollan, Sergio and Adrados, Samuel and Lopez Florez, Sebastian},
  booktitle={International conference on Infrastructure-as-a-Service (IaaS) and Platform-as-a-Service (PaaS) solutions for Europe’s Next-Gen Cloud Infrastructure (NOUS 2026)},
  year={2026},
  publisher={ACM},
  doi={10.1145/3793828.3793843}
}
```


## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
