# Transformer-based NLP for Italian Clinical EHRs

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Transformer-based structuring of Italian electronic health records with application in cardiac settings**

Repository for assessing transformer-based Named Entity Recognition (NER) models for structuring Italian Electronic Health Records (EHRs), focusing on clinical feature extraction and patient classification in cardiac settings.

## 📋 Overview

This project implements and evaluates multiple NER models for extracting structured clinical information from Italian cardiac patient anamneses:

- **SpaCy** (Best performance: 97% F1-score)
- **Flair** 
- **MultiCoNER** (based on [MultiCoNER baseline](https://github.com/amzn/multiconer-baseline))
- **Baseline** (Dictionary-based approach)

### Clinical Features Extracted

The models identify 12 clinical entities from Italian EHRs:

| Feature | Label | Type |
|---------|-------|------|
| Hypertension | HY/NOHY | Boolean |
| Dyslipidemia | DY/NODY | Boolean |
| Diabetes | DB/NODB | Boolean |
| COPD | COPD/NOCOPD | Boolean |
| NYHA Class | NYHA | Integer (1-4) |
| Ejection Fraction | EF | Integer (0-100) |
| Sinus Rhythm | RS/NORS | Boolean |
| Atrial Fibrillation | FA/NOFA | Boolean |
| Atrial Flutter | FL/NOFL | Boolean |
| Pacemaker | PM/NOPM | Boolean |
| Left Bundle Branch Block | BBSx/NOBBSx | Boolean |
| Right Bundle Branch Block | BBDx/NOBBDx | Boolean |

### Performance

| Model | Dataset | F1-Score | Features |
|-------|---------|----------|----------|
| SpaCy | AM (test) | 97.00% | 12 |
| SpaCy | EVD-100 | 97.13% | 12 |
| SpaCy | STEMI | 88.29% | 3 |

**Downstream Task**: Amyloidosis classification using extracted features achieved 66.70% F1-score with NER-annotated data vs. 66.99% with clinician-annotated data, demonstrating NLP's capability to replace manual annotation.

## 🚀 Setup Environment

### Prerequisites
- Python 3.9
- CUDA-capable GPU (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/saramazz/NLP_Italian_EHRs.git
cd NLP_Italian_EHRs
```

2. **Create and activate virtual environment**
```bash
python3.9 -m venv .env
source .env/bin/activate  # On Linux/Mac
# .env\Scripts\activate   # On Windows
```

3. **Install required packages**
```bash
pip install -r requirements.txt
# For MultiCoNER-specific requirements:
pip install -r multiconer-baseline/custom_requirements.txt
```

### Check GPU availability
```bash
nvidia-smi
```

## 📁 Project Structure
```
NLP_Italian_EHRs/
│
├── baseline/                        # Dictionary-based baseline model
│   ├── baseline.py                 # Main baseline implementation
│   ├── baseline_dictionary.json    # Medical terms dictionary v1
│   ├── baseline_dictionary_v2.json # Medical terms dictionary v2
│   └── Makefile.txt                # Baseline-specific commands
│
├── flair/                          # Flair NER implementation
│   ├── Makefile                    # Flair-specific build commands
│   ├── ner.py                      # NER model implementation
│   ├── run_ner.py                  # Training/inference runner
│   └── to_iob.py                   # IOB format converter
│
├── multiconer-baseline/            # MultiCoNER transformer implementation
│   ├── model/                      # Model architectures
│   ├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── custom_requirements.txt    # MultiCoNER dependencies
│   ├── evaluate.py                # Model evaluation
│   ├── fine_tune.py               # Fine-tuning script
│   ├── log.py                     # Logging utilities
│   ├── predict_tags.py            # Inference script
│   ├── requirements.txt           # Base requirements
│   └── train_model.py             # Main training script
│
├── spacy/                          # SpaCy NER implementation
│   ├── Makefile                    # SpaCy-specific commands
│   ├── base_config.cfg            # Base configuration
│   ├── config.cfg                 # Training configuration
│   ├── config_dbmdz.cfg           # DBMDZ model config
│   ├── predict.py                 # Inference script
│   ├── predict_dbmdz_*.iob        # Prediction outputs
│   └── to_iob.py                  # IOB format converter
│
├── scripts/                        # Utility scripts
│   ├── agreement_annotators.py    # Inter-annotator agreement
│   ├── conlleval.py              # CoNLL evaluation
│   ├── create_trainset.py        # Dataset preparation
│   ├── custom_eval.py            # Custom evaluation metrics
│   ├── extract_ananmesis.py      # Extract anamnesis from EHRs
│   ├── merge.py                  # Merge annotations
│   ├── post_annotation.py        # Post-processing
│   ├── score.py                  # Scoring utilities
│   └── stats.py                  # Dataset statistics
│
├── tokenizer/                     # Custom tokenization
│   ├── __init__.py
│   └── tokenizer.py              # Tokenizer implementation
│
├── data/                         # Dataset directory (user-provided)
│   ├── train.*.conll            # Training data
│   ├── dev.*.conll              # Development data
│   └── test.*.conll             # Test data
│
├── experiments/                  # Training outputs
│   └── [model_name]/            # Model-specific results
│
├── .gitignore                   # Git ignore rules
├── Makefile                     # Root build automation
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## 📊 Data Format

**Note**: Clinical datasets are not publicly available due to privacy regulations (GDPR). Only code is provided. Records were anonymized at source by FTGM staff following GDPR-compliant protocols before research use.

### Expected Data Structure

If you have your own Italian clinical data, place files in the `data` directory:
```
data/
├── anamnesi.a.iob    # Annotator A annotations
├── anamnesi.b.iob    # Annotator B annotations
├── anamnesi.txt      # Raw text
├── train.a.conll     # Training set (annotator A)
├── train.b.conll     # Training set (annotator B)
├── dev.a.conll       # Development set (annotator A)
├── dev.b.conll       # Development set (annotator B)
├── test.a.conll      # Test set (annotator A)
└── test.b.conll      # Test set (annotator B)
```

### Data Format Specification

Input files should be in **IOB format** (Inside-Outside-Beginning):
- Each line contains: `TOKEN TAG`
- Empty lines separate sentences
- Tags follow pattern: `B-FEATURE`, `I-FEATURE`, `O`

Example:
```
Il O
paziente O
è O
diabetico B-DB
. O

Nega O
ipertensione B-NOHY
. O
```

## 🛠️ Usage

### Using the Baseline Model

The baseline model uses a dictionary-based approach:
```bash
cd baseline/
python baseline.py --input ../data/anamnesi.txt --dictionary baseline_dictionary_v2.json
```

### Using SpaCy

Train SpaCy model:
```bash
cd spacy/
python -m spacy train config.cfg --output ./output --paths.train ../data/train.conll --paths.dev ../data/dev.conll
```

Inference:
```bash
python predict.py --model ./output/model-best --input ../data/test.txt
```

### Using Flair
```bash
cd flair/
python run_ner.py --train ../data/train.conll --dev ../data/dev.conll --test ../data/test.conll
```

### Using MultiCoNER

See the main Makefile usage section below for detailed MultiCoNER commands.

## 🔧 Makefile Usage (MultiCoNER)

The root Makefile simplifies MultiCoNER model training and inference.

### Available Variables

Default values:
```makefile
DATASET_DIR=data
LAN=it
CORPUS=b
TRAINSET=$(DATASET_DIR)/train.$(CORPUS).conll
DEVSET=$(DATASET_DIR)/dev.$(CORPUS).conll
TESTSET=$(DATASET_DIR)/test.$(CORPUS).conll
GPU=2
N_GPUS=1
EPOCHS=20
MODEL_NAME=xlm_roberta_base
ENCODER_MODEL=xlm-roberta-base
OUT_DIR=experiments
BATCH=64
LR=0.0001
```

### Prepare Dataset

Create train/dev/test splits:
```bash
# For corpus A
make CORPUS=a data/train.a.conll

# For corpus B
make CORPUS=b data/train.b.conll
```

### Training a Model

**Basic training:**
```bash
make experiments/xlm_roberta_base_lr0.0001_ep20_batch64
```

**Customize parameters:**
```bash
# Dry run to see command
make GPU=3 MODEL_NAME=test_model experiments/test_model_lr0.0001_ep20_batch64 -n

# Run with custom GPU and model name
make GPU=3 MODEL_NAME=test_model experiments/test_model_lr0.0001_ep20_batch64
```

**Background execution with logging:**
```bash
make GPU=3 MODEL_NAME=test_model experiments/test_model_lr0.0001_ep20_batch64 &> make.out &
```

### Inference on New Data
```bash
make GPU=0 ../data/your_anamnesis_output_spacy.iob
```

**Options:**
- `-B`: Force rebuild target file
- `-n`: Show commands without executing (dry run)
- `GPU=-1`: Use CPU instead of GPU

## 🔬 Research Context

This work was conducted at:
- **Biorobotics Institute**, Scuola Superiore Sant'Anna, Pisa, Italy
- **Fondazione Toscana Gabriele Monasterio** (FTGM), Pisa, Italy
- **Erasmus Medical Center**, Rotterdam, The Netherlands

**Dataset**: 2,235 patient anamneses from FTGM (405 for training, 100 EVD-100, 1,730 STEMI)

**Ethics Approval**: FTGM Ethics Committee (Decree No. 3854, 02/12/2023, Area Vasta Nord Ovest)

## 📄 Citation

If you use this code in your research, please cite:
```bibtex
@article{mazzucato2025transformer,
  title={Transformer-based structuring of Italian electronic health records with application in cardiac settings},
  author={Mazzucato, Sara and Bandini, Andrea and Sartiano, Daniele and Vergaro, Giuseppe and Dalmiani, Stefano and Emdin, Michele and Micera, Silvestro and Oddo, Calogero Maria and Passino, Claudio and Moccia, Sara},
  journal={Journal of Medical Systems},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sara Mazzucato** - Corresponding Author - [sara.mazzucato.phd@gmail.com](mailto:sara.mazzucato.phd@gmail.com)
- Andrea Bandini, Daniele Sartiano, Giuseppe Vergaro, Stefano Dalmiani, Michele Emdin, Silvestro Micera, Calogero Maria Oddo, Claudio Passino, Sara Moccia

## 🙏 Acknowledgments

This research was supported by the "Proximity Care Project" by Scuola Superiore Sant'Anna – Interdisciplinary Center "Health Science", funded by Fondazione Cassa di Risparmio di Lucca.

## ⚠️ Data Privacy & Availability

**Clinical data is NOT publicly available** due to privacy regulations and GDPR compliance. The repository provides only the code implementation. 

Records used in the original study were:
- Anonymized at source by FTGM clinical staff
- De-identified following GDPR-compliant protocols
- Approved by institutional ethics committee

Researchers interested in applying these methods should use their own institutional data following appropriate ethical and privacy guidelines.

## 💡 Use Cases

This code can be adapted for:
- Italian clinical NLP applications
- Multi-class NER with presence/absence/not-mentioned classification
- Clinical feature extraction from unstructured text
- Healthcare NLP research in non-English languages

## 🔧 Troubleshooting

**Common issues:**
- GPU memory errors: Reduce `BATCH` size
- CUDA not found: Install CUDA toolkit or use `GPU=-1` for CPU
- Import errors: Verify all requirements are installed

---

**Keywords**: Electronic Health Records, Natural Language Processing, Named Entity Recognition, Transformers, Italian Clinical Text, Cardiac Amyloidosis, SpaCy, BERT, XLM-RoBERTa
