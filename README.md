# Transformer-based NLP for Italian Clinical EHRs

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Transformer-based structuring of Italian electronic health records with application in cardiac settings**

Repository for assessing transformer-based Named Entity Recognition (NER) models for structuring Italian Electronic Health Records (EHRs), focusing on clinical feature extraction and patient classification in cardiac settings.

## 📋 Overview

This project implements and evaluates three transformer-based NER models for extracting structured clinical information from Italian cardiac patient anamneses:

- **SpaCy** (Best performance: 97% F1-score)
- **Flair** 
- **MultiCoNER** (based on [MultiCoNER baseline](https://github.com/amzn/multiconer-baseline))

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
pip install -r multiconer-baseline/custom_requirements.txt
```

### Check GPU availability
```bash
nvidia-smi
```

## 📊 Data Format

**Note**: Clinical datasets are not publicly available due to privacy regulations (GDPR). Only code is provided. Records were anonymized at source by FTGM staff following GDPR-compliant protocols before research use.

### Expected Data Structure

If you have your own Italian clinical data, place files in the `data` directory:
```
NLP_Italian_EHRs/ner$ ls data/
anamnesi.a.iob  anamnesi.b.iob  anamnesi.txt  
dev.a.conll     dev.b.conll
train.a.conll   train.b.conll
test.a.conll    test.b.conll
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

## 🛠️ Makefile Usage

The Makefile simplifies command execution for training and inference.

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

This executes:
```bash
export CUDA_VISIBLE_DEVICES="2"
train_model.py --iob_tagging ris \
  --train data/train.b.conll \
  --dev data/dev.b.conll \
  --out_dir experiments/xlm_roberta_base_lr0.0001_ep20_batch64 \
  --model_name xlm_roberta_base \
  --gpus 1 \
  --epochs 20 \
  --encoder_model xlm-roberta-base \
  --batch_size 64 \
  --lr 0.0001
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

Example with options:
```bash
make GPU=0 ../data/your_anamnesis_output_spacy.iob -B -n
```

## 📁 Project Structure
```
.
├── data/                    # Dataset directory (user-provided)
├── ner/                     # NER model implementations
│   ├── spacy/              # SpaCy model
│   ├── flair/              # Flair model
│   └── multiconer/         # MultiCoNER baseline
├── utils/                   # Utility functions
│   └── data_conversion.py  # Excel to .form conversion
├── models/                  # Trained model weights
├── experiments/            # Training output directory
├── Makefile               # Build automation
├── requirements.txt       # Python dependencies
└── README.md
```

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

- **Sara Mazzucato** - Corresponding Author - [sara.mazzucato@santannapisa.it](mailto:sara.mazzucato.phd@gmail.com)
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
