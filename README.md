# LLM Stylistic Profiling & Bias Audit: A Data-Centric Corpus Linguistics Showcase

This project aims to design and execute an end-to-end workflow for evaluating Large Language Model (LLM) outputs using corpus linguistic methods. It combines traditional corpus analysis techniques with modern LLM evaluation approaches to profile style, diversity, consistency, and audit bias and toxicity.

## Project Goals

The primary goals of this project are to:
1.  Collect and curate an evaluation corpus of LLM outputs across multiple registers and demographic prompts.
2.  Quantitatively profile the style, diversity, and consistency of these outputs using corpus-linguistic methods (e.g., lexical diversity, syntactic complexity).
3.  Audit bias and toxicity in LLM outputs using open fairness benchmarks and corpus statistics.
4.  Demonstrate an LLM-in-the-loop annotation pipeline to accelerate manual corpus work.
5.  Produce interactive visualizations and a written report suitable for a portfolio.

## Project Structure

The project is organized as follows:

```
.
├── data/
│   ├── raw/          # Original, unmodified datasets (e.g., downloaded corpora)
│   └── processed/    # Cleaned, transformed, or sampled datasets for analysis
├── docs/             # Project documentation (Plan.md, rules.mdc)
├── notebooks/        # Jupyter notebooks for experimentation and analysis
│   └── corpus_analysis_demo.ipynb # Demo notebook for core functionalities
├── src/
│   ├── analysis/     # Python scripts for analysis (e.g., llm_analysis.py)
│   ├── utils/        # Utility functions (e.g., corpus_utils.py)
│   └── visualization/# Python scripts for generating visualizations (e.g., visualization.py)
├── requirements.txt  # Project dependencies
├── LICENSE           # Project's MIT License
└── README.md         # This file: overview and setup instructions
```

## Datasets

This project utilizes several types of datasets:

1.  **Reference Corpora for Stylistic Baselines:**
    *   **Brown Corpus:** Accessed via NLTK. See setup instructions for download.
    *   **COCA Sample ( планируется):** To be downloaded from Kaggle and placed in `data/raw/coca_sample/`.
    *   **Reddit Comments Corpus ( планируется):** To be downloaded from Kaggle and placed in `data/raw/reddit_comments/`.

2.  **Bias Evaluation Datasets ( планируется):**
    *   **StereoSet:** To be sourced from Hugging Face Datasets and relevant files placed in `data/raw/stereoset/`.
    *   **HolisticBias:** To be sourced from Hugging Face Datasets and relevant files placed in `data/raw/holisticbias/`.
    *   **RealToxicityPrompts:** To be sourced from Hugging Face Datasets and relevant files placed in `data/raw/realtoxicityprompts/`.

3.  **Annotation Datasets ( планируется):**
    *   **Switchboard Dialog Act Corpus:** To be sourced from UC Berkeley and relevant files placed in `data/raw/switchboard/`.
    *   **Reddit Politeness Corpus:** To be sourced from Kaggle and relevant files placed in `data/raw/reddit_politeness/`.

4.  **LLM Outputs (to be generated):**
    *   Outputs generated from LLMs (e.g., Llama-2, GPT-3.5/4) based on designed prompts. These will be saved, likely as CSV or JSON, in `data/raw/llm_outputs/` initially, and processed versions in `data/processed/llm_outputs/`.

## Python Code Implementation

The core logic of the project is implemented in Python modules within the `src/` directory:

*   `src/utils/corpus_utils.py`: Contains utility functions for loading corpora (e.g., `load_brown_corpus()`), calculating lexical diversity (`calculate_lexical_diversity()`), finding collocations (`find_collocations()`), getting word frequencies (`get_word_frequencies()`), and calculating basic text statistics (`calculate_basic_stats()`).
*   `src/analysis/llm_analysis.py`: Implements the `LLMAnalyzer` class, which is responsible for analyzing LLM outputs against a reference corpus. It includes methods for calculating corpus statistics, analyzing individual LLM outputs, and comparing multiple outputs.
*   `src/visualization/visualization.py`: Provides functions for creating various plots to visualize analysis results, such as lexical diversity comparisons (`plot_lexical_diversity_comparison()`), word frequency comparisons (`plot_word_frequency_comparison()`), word clouds (`create_wordcloud()`), metric distributions (`plot_metric_distribution()`), and correlation matrices (`plot_correlation_matrix()`).

A demonstration of how to use these modules can be found in `notebooks/corpus_analysis_demo.ipynb`.

## Required Packages

The project relies on several Python packages. A full list with specific versions is available in `requirements.txt`. Key packages include:

```
nltk>=3.6.7
pandas>=2.2.0
numpy>=1.26.4
scipy>=1.12.0
spacy>=3.7.0
spacy-llm>=0.1.0
lexicalrichness>=0.2.0
textstat>=0.7.3
matplotlib>=3.8.0
seaborn>=0.13.0
wordcloud>=1.9.0
plotly>=5.18.0
transformers>=4.51.3
datasets>=2.19.1
evaluate>=0.4.3
jupyterlab>=3.5.3
label-studio==1.11.0
openai>=1.79.0
```
*(This is a subset, see `requirements.txt` for all packages)*

## Setup and How to Run

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and Activate Conda Environment:**
    A Conda environment named `llm-corpus-audit` with Python 3.11 is recommended.
    ```bash
    conda create -n llm-corpus-audit python=3.11 -y
    conda activate llm-corpus-audit
    ```

3.  **Install Dependencies:**
    Install all required packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data (Brown Corpus):**
    The Brown Corpus is used as a reference. NLTK downloads its data to a central directory (e.g., `~/nltk_data` on Linux/macOS, or `%APPDATA%/nltk_data` on Windows).
    ```python
    # Run this in a Python interpreter or as a script
    import nltk
    nltk.download('brown')
    nltk.download('punkt') # For tokenization
    nltk.download('averaged_perceptron_tagger') # For POS tagging if needed later
    ```
    To ensure the project can access these resources consistently, you can either:
    *   Let NLTK manage its default path. The `corpus_utils.py` script will use NLTK's default loading mechanisms.
    *   (Optional) If you prefer to have NLTK data within the project's `data/raw/nltk_data` directory, you can manually set the `NLTK_DATA` environment variable to point to `./data/raw/nltk_data` before running Python scripts, or configure NLTK within Python to look at this path. You would then need to manually copy the `corpora/brown` and other downloaded resources from the default NLTK path to `./data/raw/nltk_data/`.

5.  **Download spaCy Models:**
    A small English model from spaCy is useful for various NLP tasks.
    ```bash
    python -m spacy download en_core_web_sm
    ```

6.  **Download Other Datasets:**
    *   For datasets like COCA Sample, Reddit Comments, StereoSet, etc., download them from their respective sources (Kaggle, Hugging Face).
    *   Place the raw data files into the appropriate subdirectories within `data/raw/` as specified in the "Datasets" section above (e.g., `data/raw/coca_sample/`, `data/raw/stereoset/`).

7.  **Run the Demo Notebook:**
    *   Start Jupyter Lab:
        ```bash
        jupyter lab
        ```
    *   Navigate to the `notebooks/` directory and open `corpus_analysis_demo.ipynb`.
    *   Execute the cells in the notebook to see a demonstration of the corpus analysis utilities and visualization functions.

## Analysis Pipeline Overview

The project follows a structured analysis pipeline:

1.  **Data Collection & Curation:**
    *   Design prompt templates for LLM output generation.
    *   Generate outputs from selected LLMs.
    *   Collect and preprocess reference corpora (e.g., Brown Corpus).
2.  **Stylistic Profiling:**
    *   Extract lexical features (TTR, MTLD, etc.).
    *   Analyze syntactic features (sentence length, POS distributions).
    *   Investigate discourse features.
    *   Compare LLM outputs against reference corpora.
3.  **Bias and Toxicity Evaluation:**
    *   Utilize benchmarks like StereoSet and HolisticBias.
    *   Score outputs using tools like the Perspective API.
    *   Analyze results for demographic biases.
4.  **LLM-Assisted Annotation:**
    *   Pilot an annotation task (e.g., speech acts, politeness) using LLMs to pre-annotate.
    *   Evaluate the LLM's annotation performance.

For more details, refer to `docs/Plan.md`.

## Contributing

Contributions are welcome! Please read the project guidelines in `docs/rules.mdc` before contributing.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. 