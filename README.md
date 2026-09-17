# QKD Security Performance Simulation using BB84 Protocol and E91 Protocol

Cloning the repository
git clone https://github.com/Face2Faze/fyp-quantum-key-distribution.git
cd fyp-quantum-key-distribution

## Requirements
- Python 3.11 or higher

## Dependencies
Install all required libraries using pip:

```bash
pip install qiskit qiskit-aer numpy matplotlib cryptography
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Face2Faze/fyp-quantum-key-distribution.git
cd fyp-quantum-key-distribution
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
```

3. Activate the virtual environment:

Windows:
```bash
.venv\Scripts\activate
```

Mac / Linux:
```bash
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install qiskit qiskit-aer numpy matplotlib cryptography
```

## Running the Simulation

Full simulation (2-3 hours):
```bash
python protocol_simulation.py
```

Demo version (2-4 minutes):
```bash
python protocol_simulation_demo.py
```

## Output
Results are saved automatically to:
- results/   — CSV files
- graphs/    — PNG graphs

## Notes
- No internet connection or IBM Quantum account required
- All quantum circuits run locally on Qiskit AerSimulator
