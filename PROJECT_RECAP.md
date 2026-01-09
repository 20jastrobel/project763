# PROJECT RECAP: Deep Learning for Moving Cost Estimation
## Comprehensive Technical Documentation for Deep Research LLM

**Project Name:** `deeplearning` (project763)  
**Repository:** 20jastrobel/project763  
**Branch:** main  
**Last Updated:** January 8, 2026  
**Platform:** macOS with Python 3.12.4 (venv)

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Business Domain Context](#2-business-domain-context)
3. [Dataset Specification](#3-dataset-specification)
4. [Data Generation Pipeline](#4-data-generation-pipeline)
5. [Neural Network Architecture](#5-neural-network-architecture)
6. [Training Infrastructure](#6-training-infrastructure)
7. [Feature Engineering Details](#7-feature-engineering-details)
8. [File Structure & Codebase Map](#8-file-structure--codebase-map)
9. [Dependencies & Environment](#9-dependencies--environment)
10. [Model Outputs & Interpretability](#10-model-outputs--interpretability)
11. [Known Issues & Technical Debt](#11-known-issues--technical-debt)
12. [Research Questions & Future Directions](#12-research-questions--future-directions)
13. [Appendices](#13-appendices)

---

## 1. EXECUTIVE SUMMARY

This project implements a **neural network system for predicting moving costs and labor hours** for a residential moving company ("Badger Brothers Moving"). The core objective is to learn **feature importance weights** that determine which attributes of a move (home size, item inventory, complexity factors) most significantly impact the final cost and duration.

### Primary Goals:
1. **Predict `actual_hours`**: Clock-hours required for a moving crew to complete a job
2. **Predict `actual_cost`**: Total dollar cost charged to the customer
3. **Learn feature weights**: Identify which input variables drive predictions (interpretable ML)

### Technical Approach:
- PyTorch-based neural networks with **learnable feature importance weights**
- Three model architectures: WeightAssignmentNetwork, AutoEncoder, AttentionWeightNetwork
- Synthetic dataset with 501 moves, 36 item types, 52+ features
- Training with MSE loss, Adam optimizer, early stopping, learning rate scheduling

---

## 2. BUSINESS DOMAIN CONTEXT

### 2.1 Domain: Residential Moving Services

The data simulates a moving company that provides:
- **Labor services**: Loading, unloading, packing, disassembly/assembly
- **Transportation**: Moving items between locations
- **Specialty handling**: Pianos, safes, fragile items

### 2.2 Pricing Model (Simulated)

The synthetic cost function follows this structure:

```
cost = (duration_hours × crew_size × hourly_rate_per_mover) 
     + base_trip_fee 
     + specialty_fee (50 × specialty_items)
     + disassembly_fee (15 × disassembly_items)
     + fragile_fee (2 × fragile_items)
     + noise_factor
```

### 2.3 Duration Model (Simulated)

Work units (crew-hours) are calculated as:

```
work_units = 1.5 
           + 0.0015 × sqft 
           + 0.028 × total_items 
           + 0.05 × fragile_items
           + 0.08 × disassembly_items 
           + 0.18 × specialty_items
           + 0.015 × bulky_items 
           + 0.35 × max(floors - 1, 0)

duration_hours = (work_units × noise) / crew_size + 0.30
```

**Note:** These formulas represent the "ground truth" data generation logic. The neural network's task is to **learn these relationships from data** without knowing the explicit formulas.

---

## 3. DATASET SPECIFICATION

### 3.1 Data Files Overview

| File | Records | Columns | Purpose |
|------|---------|---------|---------|
| `moves.csv` | 501 | 18 | Primary move records with features and targets |
| `move_items.csv` | 11,806 | 3 | Junction table: move_id → item_code → qty |
| `item_catalog.csv` | 36 | 8 | Item metadata (volume, difficulty, flags) |
| `moves_ml_wide.csv` | 501 | 52 | Denormalized: one row per move with item counts as columns |
| `badger_mock.db` | N/A | N/A | SQLite database with normalized schema |

### 3.2 Primary Dataset: `moves.csv`

#### Schema:
```
move_id              INTEGER   Primary key (1-501)
home_sqft            INTEGER   Home square footage (350-3500, mode ~1200)
years_lived          FLOAT     Years customer lived at residence (0-30)
num_rooms            INTEGER   Number of rooms (1-10)
num_floors           INTEGER   Number of floors (1-4)
crew_size            INTEGER   Moving crew size (2-5)
total_items          INTEGER   Total item count for move
total_volume_cuft    FLOAT     Total cubic feet of all items
fragile_items        INTEGER   Count of fragile items (TVs, mirrors, art, lamps)
bulky_items          INTEGER   Count of items ≥25 cubic feet
disassembly_items    INTEGER   Count of items requiring disassembly
specialty_items      INTEGER   Count of specialty items (piano, safe, treadmill)
avg_difficulty       FLOAT     Weighted average item difficulty (1.0-5.0 scale)
hourly_rate_per_mover FLOAT    Hourly labor rate ($50-85)
base_trip_fee        FLOAT     Fixed trip fee ($0-150)
actual_hours         FLOAT     TARGET: Clock-hours for move completion
actual_cost          FLOAT     TARGET: Total cost charged
created_at           TEXT      ISO 8601 timestamp
```

#### Statistical Summary (from sample):
- `home_sqft`: Range 564-2997, typical 1200-2000
- `years_lived`: Range 0.7-18.0, typical 2-8 years
- `num_rooms`: Range 1-8, typical 3-5
- `crew_size`: Range 2-5, typical 3-4
- `total_items`: Range 56-243
- `actual_hours`: Range 1.89-4.02 hours
- `actual_cost`: Range $427.63-$1,283.44

### 3.3 Item Catalog: `item_catalog.csv`

36 distinct item types across 7 categories:

| Category | Items | Examples |
|----------|-------|----------|
| **box** | 4 | BOX_S, BOX_M, BOX_L, BOX_WARD |
| **furniture** | 18 | SOFA_2S, SOFA_3S, SECTIONAL, DINING_TBL, BED_FRAME, DRESSER, DESK |
| **electronics** | 3 | TV_40, TV_60, TV_75 |
| **appliance** | 4 | FRIDGE, WASHER, DRYER, DISHWASHER |
| **fragile** | 2 | MIRROR_L, ART_L |
| **decor** | 2 | LAMP, RUG |
| **specialty** | 3 | PIANO_UP, SAFE_SM, TREADMILL |

#### Item Attributes:
```
item_code           TEXT      Unique identifier (e.g., "SOFA_3S")
item_name           TEXT      Human-readable name
category            TEXT      Category classification
volume_cuft         FLOAT     Volume in cubic feet (1.5-95.0)
difficulty          INTEGER   Handling difficulty (1-5 scale)
requires_disassembly INTEGER  Boolean: requires disassembly
is_fragile          INTEGER   Boolean: fragile item
is_specialty        INTEGER   Boolean: specialty handling required
```

### 3.4 Wide-Format ML Dataset: `moves_ml_wide.csv`

**52 columns** structured as:
- 5 home features: `home_sqft`, `years_lived`, `num_rooms`, `num_floors`, `crew_size`
- 8 aggregate features: `total_items`, `total_volume_cuft`, `fragile_items`, `bulky_items`, `disassembly_items`, `specialty_items`, `avg_difficulty`
- 2 target variables: `actual_hours`, `actual_cost`
- 36 item count columns: One column per item_code with quantity

This format is **ready for direct neural network input** without joins.

---

## 4. DATA GENERATION PIPELINE

### 4.1 Generator Script: `generate_mock_data.py`

**860 lines of Python** implementing:

#### 4.1.1 Home Feature Generation

```python
def draw_home_sqft(rng):
    # Triangular distribution: low=350, high=3500, mode=1200
    return int(rng.triangular(350, 3500, 1200))

def draw_years_lived(rng):
    # Gamma-like distribution with 8% chance of long tenure
    y = rng.gammavariate(2.0, 3.0)  # mean ~6
    if rng.random() < 0.08:
        y += rng.uniform(8, 18)
    return round(min(max(y, 0.0), 30.0), 1)

def draw_num_rooms(rng, sqft):
    # Linear relationship with sqft plus noise
    base = sqft / 420.0 + rng.uniform(-0.6, 0.6)
    return int(min(max(round(base), 1), 10))

def draw_num_floors(rng):
    # Probability distribution: 62% 1-floor, 28% 2-floor, 8% 3-floor, 2% 4-floor
    r = rng.random()
    if r < 0.62: return 1
    if r < 0.90: return 2
    if r < 0.98: return 3
    return 4
```

#### 4.1.2 Item Allocation Logic

**Boxes:**
```python
boxes_total = (sqft / 18.0) + (years * 2.2) + (rooms * 3.0) + noise
# Distribution: 25% small, 45% medium, 25% large, 5% wardrobe
```

**Furniture (based on inferred bedrooms):**
```python
bedrooms = round(rooms * 0.45 + random_offset)
items["MATTRESS"] += bedrooms
items["BED_FRAME"] += bedrooms
items["BOX_SPRING"] += bedrooms (85% probability)
items["DRESSER"] += bedrooms + offset
items["NIGHTSTAND"] += bedrooms * 2 + offset
```

**Living Room (conditional on room count):**
- COFFEE_TBL: 90% probability
- END_TBL: 1-3 count
- SOFA_3S: Always if rooms ≥ 2
- SECTIONAL: 18% if sqft > 1600

**Specialty Items (low probability):**
- PIANO_UP: 4%
- SAFE_SM: 8%
- TREADMILL: 12%

#### 4.1.3 Crew Size Selection

```python
def choose_crew_size(sqft, total_volume, total_items, floors):
    crew = 2
    if sqft > 1200 or total_volume > 800 or total_items > 95:
        crew += 1
    if sqft > 2200 or total_volume > 1200 or total_items > 140:
        crew += 1
    if sqft > 3000 or total_volume > 1600:
        crew += 1
    if floors >= 3:
        crew += 1
    return min(max(crew, 2), 5)
```

### 4.2 Reproducibility

- **Seed:** 42 (configurable via `--seed` argument)
- **Deterministic:** Same seed produces identical output
- **Scalable:** `--n` argument controls dataset size (default 500)

---

## 5. NEURAL NETWORK ARCHITECTURE

### 5.1 WeightAssignmentNetwork (Primary Model)

**Purpose:** Learn feature importance weights while predicting targets.

```python
class WeightAssignmentNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], output_dim=1, dropout=0.2):
        # Learnable feature weights (key innovation)
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        
        # Network architecture
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Apply softmax-normalized feature weights
        weighted_x = x * F.softmax(self.feature_weights, dim=0)
        return self.network(weighted_x)
    
    def get_feature_weights(self):
        return F.softmax(self.feature_weights, dim=0).detach()
```

**Architecture Details:**
- **Input:** Variable (depends on features used, typically 52 for wide format)
- **Hidden Layers:** [128, 64, 32] neurons with BatchNorm + ReLU + Dropout(0.2)
- **Output:** Single value (hours or cost prediction)
- **Feature Weighting:** Softmax-normalized learnable parameters

### 5.2 AutoEncoder

**Purpose:** Unsupervised learning of compact feature representations.

```python
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        # Encoder: input_dim -> 128 -> 64 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder: latent_dim -> 64 -> 128 -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
```

**Use Cases:**
- Dimensionality reduction
- Feature learning before supervised training
- Anomaly detection (high reconstruction error)

### 5.3 AttentionWeightNetwork

**Purpose:** Use self-attention to discover feature interactions.

```python
class AttentionWeightNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_heads=4):
        self.embed_dim = 64
        self.feature_embedding = nn.Linear(1, self.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # Embed each feature: (batch, features) -> (batch, features, embed_dim)
        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        # Self-attention
        attended, attention_weights = self.attention(x, x, x)
        # Output
        output = self.fc(attended.view(batch_size, -1))
        return output, attention_weights
```

**Advantage:** Returns attention weights showing feature-to-feature interactions.

---

## 6. TRAINING INFRASTRUCTURE

### 6.1 Data Loading Pipeline

```python
class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def preprocess(self, df, target_column=None, fit_scaler=True):
        # Separate features and target
        if target_column in df.columns:
            targets = df[target_column].values
            features_df = df.drop(columns=[target_column])
        else:
            targets = None
            features_df = df
        
        # Numeric columns only, standardized
        numeric_df = features_df.select_dtypes(include=[np.number])
        if fit_scaler:
            features = self.scaler.fit_transform(numeric_df.values)
        else:
            features = self.scaler.transform(numeric_df.values)
        
        return features, targets
```

### 6.2 Training Configuration

**Default Hyperparameters:**
```
epochs:          100
batch_size:      32
learning_rate:   0.001
train_split:     0.8 (80% train, 20% validation)
early_stopping:  patience=15
lr_scheduler:    ReduceLROnPlateau(patience=5, factor=0.5)
loss:            MSELoss
optimizer:       Adam
seed:            42
```

### 6.3 Training Loop

```python
def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader, processor = get_data_loaders(
        args.data_path,
        target_column=args.target_column,
        batch_size=args.batch_size,
        train_split=args.train_split
    )
    
    # Create model
    model = create_model(args.model_type, input_dim=input_dim, output_dim=args.output_dim)
    
    # Training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=args.patience)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            save_model(model, args.save_path, 'best_model.pt')
        
        if early_stopping(val_loss):
            break
```

### 6.4 Command-Line Interface

```bash
python src/train.py \
    --data_path data/moves_ml_wide.csv \
    --target_column actual_cost \
    --model_type weight_assignment \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --save_path models/
```

---

## 7. FEATURE ENGINEERING DETAILS

### 7.1 Raw Features (from moves.csv)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `home_sqft` | Continuous | 350-3500 | Square footage of home |
| `years_lived` | Continuous | 0-30 | Years at residence |
| `num_rooms` | Discrete | 1-10 | Number of rooms |
| `num_floors` | Discrete | 1-4 | Number of floors |
| `crew_size` | Discrete | 2-5 | Size of moving crew |

### 7.2 Aggregate Features (computed from items)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `total_items` | Discrete | 56-260 | Sum of all item quantities |
| `total_volume_cuft` | Continuous | 500-1600 | Total cubic feet |
| `fragile_items` | Discrete | 0-15 | Items marked is_fragile=1 |
| `bulky_items` | Discrete | 0-25 | Items with volume ≥25 cuft |
| `disassembly_items` | Discrete | 0-10 | Items requiring disassembly |
| `specialty_items` | Discrete | 0-3 | Items marked is_specialty=1 |
| `avg_difficulty` | Continuous | 1.0-2.0 | Weighted average difficulty |

### 7.3 Pricing Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `hourly_rate_per_mover` | Continuous | $50-85 | Labor rate per person per hour |
| `base_trip_fee` | Continuous | $0-150 | Fixed trip charge |

### 7.4 Item Count Features (36 columns in wide format)

Each item code becomes a column with the quantity for that move:
```
BOX_S, BOX_M, BOX_L, BOX_WARD, SOFA_2S, SOFA_3S, SECTIONAL, 
COFFEE_TBL, END_TBL, TV_40, TV_60, TV_75, DINING_TBL, DINING_CHAIR, 
BAR_STOOL, MATTRESS, BOX_SPRING, BED_FRAME, DRESSER, NIGHTSTAND, 
BOOKCASE, DESK, OFFICE_CHAIR, FILE_CAB, FRIDGE, WASHER, DRYER, 
DISHWASHER, MIRROR_L, ART_L, LAMP, RUG, PIANO_UP, SAFE_SM, TREADMILL
```

### 7.5 Target Variables

| Target | Type | Range | Description |
|--------|------|-------|-------------|
| `actual_hours` | Continuous | 1.0-16.0 | Clock-hours for completion |
| `actual_cost` | Continuous | $200-$2000+ | Total charge to customer |

---

## 8. FILE STRUCTURE & CODEBASE MAP

```
deeplearning/
├── .github/
│   └── copilot-instructions.md    # Project context for AI assistants
├── .venv/                         # Python virtual environment
├── data/
│   └── .gitkeep                   # Placeholder for user data
├── models/
│   ├── .gitkeep
│   └── pert_estimator_model.keras # Pre-trained Keras model (legacy?)
├── notebooks/
│   └── exploration.ipynb          # Jupyter notebook for EDA & training
├── src/
│   ├── __init__.py                # Package marker
│   ├── data_loader.py             # DataProcessor, DatabaseDataset, get_data_loaders
│   ├── model.py                   # WeightAssignmentNetwork, AutoEncoder, AttentionWeightNetwork
│   ├── train.py                   # Training script with CLI
│   ├── pert_keras_demo.py         # Keras PERT estimation demo (legacy?)
│   └── utils.py                   # set_seed, save_model, load_model, EarlyStopping, AverageMeter
├── badger_mock.db                 # SQLite database (normalized schema)
├── generate_mock_data.py          # 860-line data generation script
├── item_catalog.csv               # 36 item types with metadata
├── move_items.csv                 # 11,806 move-item associations
├── moves.csv                      # 501 moves with features and targets
├── moves_ml_wide.csv              # Denormalized ML-ready format
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── README (1).md                  # Duplicate/backup README
```

---

## 9. DEPENDENCIES & ENVIRONMENT

### 9.1 Python Environment

```
Python Version: 3.12.4
Environment:    venv (.venv/)
OS:             macOS
```

### 9.2 requirements.txt

```
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
jupyter>=1.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### 9.3 Optional Dependencies (for full functionality)

```
openpyxl     # Excel file support
pyarrow      # Parquet file support
```

---

## 10. MODEL OUTPUTS & INTERPRETABILITY

### 10.1 Feature Weight Extraction

After training `WeightAssignmentNetwork`:

```python
weights = model.get_feature_weights().cpu().numpy()
for i, (col, w) in enumerate(zip(processor.feature_columns, weights)):
    print(f"{col}: {w:.4f}")
```

### 10.2 Expected High-Weight Features

Based on the data generation formulas, these features should receive higher weights:
1. `total_items` (0.028 coefficient in work_units)
2. `specialty_items` (0.18 coefficient - highest impact)
3. `disassembly_items` (0.08 coefficient)
4. `fragile_items` (0.05 coefficient)
5. `num_floors` (0.35 coefficient for floors > 1)
6. `crew_size` (divides work_units, affects cost multiplier)

### 10.3 Attention Visualization

For `AttentionWeightNetwork`:

```python
output, attention_weights = model(inputs)
# attention_weights shape: (batch, num_heads, features, features)
# Visualize feature-to-feature attention patterns
```

### 10.4 Model Checkpoints

Saved to `models/` directory:
- `best_model.pt` - Lowest validation loss
- `final_model.pt` - Final epoch model
- `training_history.json` - Loss curves per epoch

---

## 11. KNOWN ISSUES & TECHNICAL DEBT

### 11.1 Type Hints

Several functions have type hint issues flagged by Pylance:
```python
# In data_loader.py
def preprocess(self, df: pd.DataFrame, target_column: str = None, ...):
    # Should be: target_column: Optional[str] = None
```

### 11.2 Import Resolution

Lint errors for unresolved imports (torch, numpy, pandas) indicate the virtual environment may not be fully configured in the IDE.

### 11.3 Data Split Method

Current split is sequential (`df.iloc[:train_size]`), not random. This could introduce bias if data has temporal ordering.

**Recommended fix:**
```python
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
```

### 11.4 Legacy Files

- `pert_keras_demo.py` - Appears to be from a different approach (Keras PERT estimation)
- `pert_estimator_model.keras` - Pre-trained model, unclear if still relevant
- `README (1).md` - Duplicate file

### 11.5 Missing Functionality

- No model inference/prediction script
- No hyperparameter tuning framework
- No cross-validation
- No test set (only train/val split)

---

## 12. RESEARCH QUESTIONS & FUTURE DIRECTIONS

### 12.1 Feature Importance Analysis

**Question:** Do the learned feature weights match the known data generation coefficients?

**Approach:**
1. Train WeightAssignmentNetwork on `actual_hours` prediction
2. Extract learned weights
3. Compare to ground truth coefficients from `generate_mock_data.py`
4. Quantify weight ranking correlation (Spearman's rho)

### 12.2 Multi-Target Learning

**Question:** Can a single model predict both hours and cost effectively?

**Approach:**
- Modify `output_dim=2`
- Use weighted multi-task loss
- Compare to single-target models

### 12.3 Attention Analysis

**Question:** Which features attend to each other in cost prediction?

**Approach:**
- Train AttentionWeightNetwork
- Visualize attention heatmaps
- Identify feature interaction patterns

### 12.4 Generalization Testing

**Question:** How well does the model generalize to edge cases?

**Test scenarios:**
- Moves with unusual item compositions (all boxes, no furniture)
- Very large homes (>3000 sqft)
- High specialty item counts
- Synthetic "adversarial" moves

### 12.5 Real-World Transfer

**Question:** If real moving data becomes available, what domain adaptation is needed?

**Considerations:**
- Distribution shift in home sizes, item frequencies
- Regional pricing differences
- Seasonal patterns
- Data quality/missing values

### 12.6 Model Improvements

**Potential enhancements:**
1. **Embedding layers** for categorical features (item codes, categories)
2. **Graph neural networks** for item relationship modeling
3. **Uncertainty quantification** (Bayesian NN, MC Dropout)
4. **Ensemble methods** combining multiple architectures
5. **Regularization** (L1 for sparse feature selection)

---

## 13. APPENDICES

### Appendix A: Complete Item Catalog

| Code | Name | Category | Volume (cuft) | Difficulty | Disassembly | Fragile | Specialty |
|------|------|----------|---------------|------------|-------------|---------|-----------|
| BOX_S | Small box | box | 1.5 | 1 | 0 | 0 | 0 |
| BOX_M | Medium box | box | 3.0 | 1 | 0 | 0 | 0 |
| BOX_L | Large box | box | 4.5 | 1 | 0 | 0 | 0 |
| BOX_WARD | Wardrobe box | box | 8.0 | 2 | 0 | 0 | 0 |
| SOFA_2S | Loveseat | furniture | 45.0 | 3 | 0 | 0 | 0 |
| SOFA_3S | 3-seat sofa | furniture | 60.0 | 3 | 0 | 0 | 0 |
| SECTIONAL | Sectional sofa | furniture | 95.0 | 4 | 1 | 0 | 0 |
| COFFEE_TBL | Coffee table | furniture | 12.0 | 2 | 0 | 0 | 0 |
| END_TBL | End table | furniture | 6.0 | 2 | 0 | 0 | 0 |
| TV_40 | TV (up to 40") | electronics | 6.0 | 2 | 0 | 1 | 0 |
| TV_60 | TV (41-65") | electronics | 9.0 | 3 | 0 | 1 | 0 |
| TV_75 | TV (66"+) | electronics | 14.0 | 4 | 0 | 1 | 0 |
| DINING_TBL | Dining table | furniture | 40.0 | 3 | 1 | 0 | 0 |
| DINING_CHAIR | Dining chair | furniture | 6.0 | 1 | 0 | 0 | 0 |
| BAR_STOOL | Bar stool | furniture | 4.0 | 1 | 0 | 0 | 0 |
| MATTRESS | Mattress | furniture | 35.0 | 3 | 0 | 0 | 0 |
| BOX_SPRING | Box spring | furniture | 25.0 | 2 | 0 | 0 | 0 |
| BED_FRAME | Bed frame | furniture | 25.0 | 3 | 1 | 0 | 0 |
| DRESSER | Dresser | furniture | 30.0 | 3 | 0 | 0 | 0 |
| NIGHTSTAND | Nightstand | furniture | 6.0 | 1 | 0 | 0 | 0 |
| BOOKCASE | Bookcase | furniture | 18.0 | 2 | 0 | 0 | 0 |
| DESK | Desk | furniture | 22.0 | 3 | 1 | 0 | 0 |
| OFFICE_CHAIR | Office chair | furniture | 10.0 | 2 | 0 | 0 | 0 |
| FILE_CAB | File cabinet | furniture | 12.0 | 3 | 0 | 0 | 0 |
| FRIDGE | Refrigerator | appliance | 32.0 | 4 | 0 | 0 | 0 |
| WASHER | Washer | appliance | 25.0 | 4 | 0 | 0 | 0 |
| DRYER | Dryer | appliance | 25.0 | 3 | 0 | 0 | 0 |
| DISHWASHER | Dishwasher | appliance | 18.0 | 3 | 0 | 0 | 0 |
| MIRROR_L | Large mirror | fragile | 8.0 | 3 | 0 | 1 | 0 |
| ART_L | Large framed art | fragile | 6.0 | 3 | 0 | 1 | 0 |
| LAMP | Lamp | decor | 3.0 | 1 | 0 | 1 | 0 |
| RUG | Rug | decor | 6.0 | 1 | 0 | 0 | 0 |
| PIANO_UP | Upright piano | specialty | 60.0 | 5 | 0 | 0 | 1 |
| SAFE_SM | Safe (small) | specialty | 18.0 | 5 | 0 | 0 | 1 |
| TREADMILL | Treadmill | specialty | 30.0 | 5 | 1 | 0 | 1 |

### Appendix B: SQLite Database Schema

```sql
CREATE TABLE item_catalog (
    item_code TEXT PRIMARY KEY,
    item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    volume_cuft REAL NOT NULL,
    difficulty INTEGER NOT NULL,
    requires_disassembly INTEGER NOT NULL CHECK (requires_disassembly IN (0,1)),
    is_fragile INTEGER NOT NULL CHECK (is_fragile IN (0,1)),
    is_specialty INTEGER NOT NULL CHECK (is_specialty IN (0,1))
);

CREATE TABLE moves (
    move_id INTEGER PRIMARY KEY,
    home_sqft INTEGER NOT NULL,
    years_lived REAL NOT NULL,
    num_rooms INTEGER NOT NULL,
    num_floors INTEGER NOT NULL,
    crew_size INTEGER NOT NULL,
    total_items INTEGER NOT NULL,
    total_volume_cuft REAL NOT NULL,
    fragile_items INTEGER NOT NULL,
    bulky_items INTEGER NOT NULL,
    disassembly_items INTEGER NOT NULL,
    specialty_items INTEGER NOT NULL,
    avg_difficulty REAL NOT NULL,
    hourly_rate_per_mover REAL NOT NULL,
    base_trip_fee REAL NOT NULL,
    actual_hours REAL NOT NULL,
    actual_cost REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE move_items (
    move_id INTEGER NOT NULL,
    item_code TEXT NOT NULL,
    qty INTEGER NOT NULL CHECK (qty >= 0),
    PRIMARY KEY (move_id, item_code),
    FOREIGN KEY (move_id) REFERENCES moves(move_id) ON DELETE CASCADE,
    FOREIGN KEY (item_code) REFERENCES item_catalog(item_code)
);

CREATE INDEX idx_move_items_move_id ON move_items(move_id);
CREATE INDEX idx_move_items_item_code ON move_items(item_code);
```

### Appendix C: Sample Training Command

```bash
# Activate virtual environment
source .venv/bin/activate

# Train model predicting cost
python src/train.py \
    --data_path moves_ml_wide.csv \
    --target_column actual_cost \
    --model_type weight_assignment \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --patience 15 \
    --save_path models/cost_model/

# Train model predicting hours
python src/train.py \
    --data_path moves_ml_wide.csv \
    --target_column actual_hours \
    --model_type weight_assignment \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --patience 15 \
    --save_path models/hours_model/
```

### Appendix D: Ground Truth Cost Formula

```python
# From generate_mock_data.py - generate_targets()

work_units = (
    1.5
    + 0.0015 * sqft
    + 0.028 * total_items
    + 0.05 * fragile_items
    + 0.08 * disassembly_items
    + 0.18 * specialty_items
    + 0.015 * bulky_items
    + 0.35 * max(floors - 1, 0)
)

noise = math.exp(rng.gauss(0, 0.18))
crew_hours = work_units * noise
duration_hours = crew_hours / crew_size + 0.30

hourly_rate_per_mover = uniform(50, 85)
base_trip_fee = uniform(0, 150)

specialty_fee = 50.0 * specialty_items
disassembly_fee = 15.0 * disassembly_items
fragile_fee = 2.0 * fragile_items

cost = (duration_hours * crew_size * hourly_rate_per_mover 
      + base_trip_fee 
      + specialty_fee 
      + disassembly_fee 
      + fragile_fee)
cost *= (1.0 + gauss(0, 0.07))  # Additional noise
cost = max(200.0, cost)
```

---

## DOCUMENT END

**Document Version:** 1.0  
**Generated:** January 8, 2026  
**For:** Deep Research LLM Analysis  
**Contact:** 20jastrobel (GitHub)
