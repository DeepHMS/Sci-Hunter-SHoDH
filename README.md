<div align="center">
  <img src="Sci-hunter_header.png" width="80%" alt="Sci-Hunter Header">
</div>

# Sci-Hunter 🧬🔍

**Automated Biomedical Literature Mining and Semantic Harmonization**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

**Sci-Hunter** is an automated and highly customizable literature mining tool designed to query the PubMed database, extract relevant manuscripts using complex keyword logic, and perform advanced text matching to identify specific user-defined features. 

The application is built upon the foundation of **SHoDH (Semantic Harmonizer for Optimized Document Harvesting)**, a computational framework originally developed by **Dr. Deeptarup Biswas** in 2022. It is engineered to streamline high-throughput systematic reviews and integrate seamlessly into advanced biomedical analysis workflows.

---

## 🌟 Key Features

* **NCBI PubMed API Integration:** Direct linkage to PubMed via `Biopython` for dynamic fetching of manuscript metadata based on customizable chronological and publication type filters.
* **Advanced Semantic Harmonization:** Utilizes Levenshtein distance-based algorithms (`thefuzz`) to perform deep text-matching of user keywords against titles and abstracts, accounting for spelling variations and hyphenations.
* **Feature-Specific Mapping:** Upload custom `.csv` lists of genes, proteins, or other clinical features to cross-reference against thousands of extracted manuscripts.
* **Interactive Visual Analytics:** Generates dynamic timeline trends, interactive Venn diagrams, and cross-over bar plots utilizing `Plotly`.
* **Secure Access Portal:** Built-in gateway utilizing Google Service Accounts (`gspread`) to log user access and maintain an authenticated session state.
* **Comprehensive Data Export:** Clean, downloadable CSV outputs summarizing metadata, frequency hits, similarity percentages, and calculated confidence scores.

---

## ⚙️ Installation & Local Setup

To run Sci-Hunter locally or deploy it on a cloud platform (e.g., Hugging Face Spaces, Streamlit Community Cloud), follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/Sci-Hunter.git](https://github.com/yourusername/Sci-Hunter.git)
cd Sci-Hunter
