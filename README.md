# Deep Learning - Neural Network for Database Weight Assignment

Documentation site: https://docs.start.qolab.ai

A Python deep learning project that processes database files and assigns weights using neural networks.

## Project Structure

```
deeplearning/
├── data/                   # Raw and processed data files
├── models/                 # Trained model checkpoints
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and preprocessing utilities
│   ├── model.py            # Neural network architecture
│   ├── train.py            # Training script
│   └── utils.py            # Helper functions
├── notebooks/
│   └── exploration.ipynb   # Jupyter notebook for experimentation
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/train.py --data_path data/ --epochs 100
```

### PERT + Keras Demo
```bash
python src/pert_keras_demo.py --data_source auto --target_column actual_cost
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Dependencies

- PyTorch
- pandas
- numpy
- scikit-learn
- jupyter
- matplotlib

## License

MIT
