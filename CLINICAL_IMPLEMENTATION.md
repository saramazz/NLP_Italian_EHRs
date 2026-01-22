# Clinical Implementation Guide for Cardiac Amyloidosis Screening

This document provides practical guidance for healthcare institutions seeking to 
deploy the NLP models described in Mazzucato et al. (2026) for automated cardiac 
amyloidosis (CA) screening from Italian clinical narratives.

> **Disclaimer**: This is a research-stage tool. Multi-center clinical validation 
> and usability testing are required before implementation in clinical workflows.

## Overview

Three deployment pathways are available, ranging from rapid prototyping to 
production-grade systems. Choose based on your institution's IT infrastructure, 
data-sharing policies, and clinical requirements.

---

## Three Deployment Pathways

### Pathway 1: Zero-Shot Screening (Fastest, Lowest Cost)

**Best for**: Rapid screening without annotation infrastructure; privacy-critical 
institutions; limited IT resources.

**Implementation Steps**:
1. Install open-source model locally:
```bash
   ollama pull qwen2.5-7b-instruct
```
   Or download MedGemma-27B-IT from Hugging Face

2. Load Italian anamnesis EHR exports (plain text or CSV)

3. Run zero-shot binary classification on each record using the provided Python 
   wrapper (see `scripts/` directory)

4. Output: CA presence prediction (yes/no) + confidence score (0–1)

**Specifications**:
- **Deployment time**: 2–4 hours
- **Cost**: €0
- **Performance**: F1=0.92
- **Infrastructure**: Standard hospital server (CPU-capable; GPU optional)
- **Data handling**: Fully on-premises (no external APIs; fully GDPR-compliant)

**Advantages**:
- No API dependencies or data sharing
- Rapid deployment
- Complete privacy (data never leaves hospital servers)

**Limitations**:
- No feature-level interpretability
- Predictions may be harder for clinicians to explain to patients

---

### Pathway 2: Structured Extraction + Supervised Classification

**Best for**: Institutions requiring feature-level interpretability; clinician 
review of extracted variables; regulatory auditing.

**Implementation Steps**:
1. Extract 21 clinical variables from anamneses using SpaCy or Qwen model
   - See `spacy/` and `scripts/` directories for code

2. Load extracted features into pre-trained classifier (XGBoost or Logistic Regression)

3. Generate output: CA risk score, confidence interval, and top 5 discriminative 
   features per patient

4. Clinicians can review which clinical variables drove the CA classification

**Specifications**:
- **Deployment time**: 4–6 hours
- **Cost**: €0
- **Performance**: F1=0.80 (classification)
- **Infrastructure**: Standard hospital server
- **Data handling**: Fully on-premises

**Advantages**:
- Feature-level interpretability (which variables matter?)
- Clinicians can review and override automated decisions
- Good for regulatory auditing and compliance

**Limitations**:
- Slightly lower performance than zero-shot approaches
- Requires variable extraction (1–2% error rate on difficult entities like DSM, IPC)

---

### Pathway 3: Proprietary API Integration (Highest Accuracy, Ongoing Costs)

**Best for**: Institutions prioritizing maximum accuracy; those already using 
GPT-4 APIs; organizations with external data-sharing agreements.

**Implementation Steps**:
1. Configure GPT-4.1-mini API endpoint with hospital IT/data governance
2. Set up secure API calls with encrypted data transmission
3. Batch process anamnesis records through API with structured JSON prompts
4. Store returned predictions securely

**Specifications**:
- **Integration time**: 2–3 hours
- **Ongoing cost**: €2–5 per 1,000 patient records
  - Example: €200–1,000/year for hospitals screening 100,000 patients annually
- **Performance**: F1=0.96 (highest)
- **Data handling**: Requires data-sharing agreement; data transmitted to external API

**Advantages**:
- Highest extraction and classification accuracy
- Minimal local infrastructure required
- Easy to scale

**Limitations**:
- Recurring operational costs
- Data leaves hospital servers (requires GDPR/HIPAA compliance review)
- Dependent on third-party API availability

---

## Hospital Infrastructure Integration

All three pathways can be integrated with existing hospital systems:

### EHR Export Interface
- Accept Italian anamnesis text exports from your EHR system
- Batch processing: Daily, weekly, or on-demand
- Format: Plain text or CSV with patient ID + anamnesis text

### Prediction Storage
- Store CA risk predictions in hospital-approved database
- Include: Patient ID, timestamp, risk score, confidence interval, model version
- Retention: Follow institutional data governance policies

### Audit & Compliance Logging
- Log all predictions and model confidence scores
- Track clinician reviews and overrides (if applicable)
- Enable regulatory auditing for compliance with GDPR, institutional policies

### Clinical Decision Support (CDSS) Integration (Optional)
- Display CA risk scores in clinician-facing dashboards
- Alert logic: High-risk patients (e.g., risk score > 0.8) for clinician review
- Integration with existing cardiology CDSS workflows

---

## Resource Requirements by Pathway

| Resource | Pathway 1 | Pathway 2 | Pathway 3 |
|----------|----------|----------|----------|
| **GPU** | Optional | Optional | Not needed |
| **Local Storage** | <5 GB | <5 GB | <1 GB |
| **Network** | None (offline) | None (offline) | Required (API) |
| **IT Effort** | Minimal | Moderate | Moderate |
| **Data Governance Review** | Yes | Yes | Yes* |
| **One-time Cost** | €0 | €0 | €0 |
| **Annual Cost (1000 pts)** | €0 | €0 | €200–500 |

*Pathway 3 requires more extensive review for external data transfer.

---

## Getting Started Checklist

### Step 1: Technical Feasibility (IT Team)
- [ ] Review hospital IT infrastructure and server availability
- [ ] Confirm CPU/GPU availability (optional for Pathways 1–2)
- [ ] Verify network requirements (Pathway 3 only)

### Step 2: Data Governance Review (IT + Privacy Officer)
- [ ] Confirm Italian anamnesis data can be exported from EHR
- [ ] Ensure compliance with GDPR and institutional policies
- [ ] Define data retention and deletion policies
- [ ] For Pathway 3: Review external API data-sharing requirements

### Step 3: Clinical Validation (Cardiology Department)
- [ ] Identify pilot user group (3–5 cardiologists)
- [ ] Define clinician feedback mechanisms
- [ ] Establish workflow for CA risk score review and override
- [ ] Plan user training and documentation

### Step 4: Technical Setup (IT/Data Science Team)
- Follow setup instructions in main [README.md](README.md)
- Choose your preferred pathway
- Configure database storage and audit logging

### Step 5: Pilot Testing (2–4 weeks)
- [ ] Process first 100–500 patient records
- [ ] Clinician usability testing
- [ ] Compare AI predictions to manual chart review (validation)
- [ ] Gather feedback and refine workflows

---

## Technical Support

For technical questions:
- Open an issue on [GitHub](https://github.com/saramazz/NLP_Italian_EHRs)

For clinical or implementation questions:
- Email: sara.mazzucato.phd@gmail.com

---

## Important Disclaimers & Regulatory Considerations

⚠️ **Research Tool**: This is a research prototype, not a FDA-cleared or 
CE-marked medical device.

⚠️ **External Validation Required**: Multi-center prospective validation is 
needed before clinical implementation. Single-center, retrospective studies do 
not establish clinical utility.

⚠️ **Clinician Review Mandatory**: All automated CA risk scores should be reviewed 
by qualified cardiologists before clinical decision-making.

⚠️ **Regulatory Compliance**: Ensure compliance with:
- **GDPR** (EU): Data protection for patient records
- **HIPAA** (USA): If applicable
- **Local institutional** policies on AI/ML in clinical workflows

⚠️ **Liability**: The authors and institutions assume no liability for deployment 
or clinical outcomes. Your institution is responsible for clinical validation, 
risk assessment, and regulatory compliance.

---

## References

See Mazzucato et al. (2026) "Detecting Cardiac Amyloidosis in Italian Cardiology 
Reports: Structured Variable Extraction versus Direct Free-Text Analysis" for 
full methodology, validation results, and performance metrics.

**Repository**: https://github.com/saramazz/NLP_Italian_EHRs

**License**: MIT

---

**Last updated**: January 2025
**Version**: 1.0
