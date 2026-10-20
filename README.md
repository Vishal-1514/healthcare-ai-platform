# 🩺 Multi-Modal Medical Diagnostic Platform

An integrated healthcare platform combining computer vision for dermatological screening, time-series neural networks for neuromuscular signal analysis, and a large language model for automated symptom triage.

---

## 🏗️ System Architecture & Core Modules

1. **Visual Dermatological Screening (ResNet-50)**
   * **Task:** Classification of 23 distinct skin disease categories.
   * **Framework:** PyTorch
   * **Architecture:** Pre-trained ResNet-50 backbone fine-tuned with custom classification heads, dropout regularization (0.4), and automated data augmentation.

2. **Electromyography (EMG) Neuromuscular Analysis (1D CNN-LSTM)**
   * **Task:** 3-class time-series signal classification (`Healthy`, `Neuropathy`, `Myopathy`).
   * **Framework:** Keras / TensorFlow
   * **Architecture:** Hybrid 1D-CNN feature extractor combined with LSTM sequence modeling layers to analyze raw 1,000-sample EMG waveforms.

3. **Conversational Symptom Assistant (Gemini API)**
   * **Task:** Patient symptom intake and clinical triage guidance.
   * **Engine:** Google Gemini LLM via `google-genai` SDK.

---

## 🚀 Local Installation & Setup

### Prerequisites
* Python 3.10+
* Virtual Environment (`venv` or `conda`)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/healthcare-ai-platform.git](https://github.com/YOUR_USERNAME/healthcare-ai-platform.git)
   cd healthcare-ai-platform