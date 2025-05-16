# LLM Stylistic Profiling & Bias Audit: A Data-Centric Corpus Linguistics Showcase

This project aims to design and execute an end-to-end workflow for evaluating Large Language Model (LLM) outputs using corpus linguistic methods. It combines traditional corpus analysis techniques with modern LLM evaluation approaches to profile style, diversity, consistency, and audit bias and toxicity.

## Project Goals

The primary goals of this project are to:
1.  Collect and curate an evaluation corpus of LLM outputs across multiple registers and demographic prompts.
2.  Quantitatively profile the style, diversity, and consistency of these outputs using corpus-linguistic methods (e.g., lexical diversity, syntactic complexity).
3.  Audit bias and toxicity in LLM outputs using open fairness benchmarks and corpus statistics.
4.  Demonstrate an LLM-in-the-loop annotation pipeline to accelerate manual corpus work.
5.  Produce interactive visualizations and a written report suitable for a portfolio.

By completing this project, you will illustrate expertise in:
*   Interpretability & explainability of LLM outputs via linguistic features.
*   Evaluation & validation (style drift, bias, toxicity).
*   Data curation / domain adaptation principles.
*   LLM-augmented corpus workflows (spaCy-LLM, Label Studio AI-assist).

## Learning Outcomes
Upon completion of this project, you will be able to:
*   Apply classic corpus methods to modern LLM evaluation.
*   Build reproducible Python pipelines (datasets → analysis → visualisation).
*   Use open-source LLMs/APIs programmatically.
*   Communicate findings in clear, portfolio-ready artefacts (notebooks + comprehensive README).

## Project Structure

The project is organized as follows:

```
.
├── data/
│   ├── raw/          # Original, unmodified datasets (e.g., downloaded corpora)
│   └── processed/    # Cleaned, transformed, or sampled datasets for analysis
├── docs/             # Supporting project documentation (e.g. original Plan.md)
├── notebooks/        # Jupyter notebooks for experimentation and analysis
│   └── corpus_analysis_demo.ipynb # Demo notebook for core functionalities
├── src/
│   ├── analysis/     # Python scripts for analysis (e.g., llm_analysis.py)
│   ├── utils/        # Utility functions (e.g., corpus_utils.py)
│   └── visualization/# Python scripts for generating visualizations (e.g., visualization.py)
├── requirements.txt  # Project dependencies
├── LICENSE           # Project's MIT License
└── README.md         # This file: overview, setup, planning, and documentation
```
*(The `docs` directory may contain supplementary documents not merged into this README).*

## Datasets and Tools

This project utilizes several types of datasets and tools:

| Purpose                                   | Dataset / Tool                                                                 | Link / Source                                                                 | Status                                   |
| :---------------------------------------- | :----------------------------------------------------------------------------- | :---------------------------------------------------------------------------- | :--------------------------------------- |
| **Reference Corpora**                     |                                                                                |                                                                               |                                          |
| Stylistic Baselines                       | Brown Corpus                                                                   | NLTK (`nltk.download('brown')`)                                               | Integrated                               |
| Stylistic Baselines                       | COCA Sample                                                                    | Kaggle                                                                        | Planned (Store in `data/raw/coca_sample/`) |
| Stylistic Baselines                       | Reddit Comments Corpus                                                         | Kaggle                                                                        | Planned (Store in `data/raw/reddit_comments/`) |
| **Bias Evaluation**                       |                                                                                |                                                                               |                                          |
|                                           | StereoSet                                                                      | Hugging Face Datasets                                                         | Planned (Store in `data/raw/stereoset/`)   |
|                                           | HolisticBias                                                                   | Hugging Face Datasets                                                         | Planned (Store in `data/raw/holisticbias/`) |
|                                           | RealToxicityPrompts                                                            | Hugging Face Datasets                                                         | Planned (Store in `data/raw/realtoxicityprompts/`) |
| **Toxicity Scoring**                      |                                                                                |                                                                               |                                          |
|                                           | Perspective API (Jigsaw)                                                       | [Perspective API](https://developers.perspectiveapi.com)                      | Planned                                  |
| **MDA Feature Extraction**                |                                                                                |                                                                               |                                          |
|                                           | MAT 1.3 (Java) OR Biber feature list via spaCy / custom scripts                | [MAT Tagger](https://ucrel-web.lancs.ac.uk/mat/)                              | Optional / Planned                       |
| **LLM Models**                            |                                                                                |                                                                               |                                          |
|                                           | Llama-2-7B-Chat (HF), OpenAI GPT-3.5/4                                         | Hugging Face / OpenAI                                                         | For Comparison                           |
| **Annotation Samples**                    |                                                                                |                                                                               |                                          |
|                                           | Switchboard Dialog Act Corpus                                                  | UC Berkeley                                                                   | Planned (Store in `data/raw/switchboard/`) |
|                                           | Reddit Politeness Corpus                                                         | Kaggle                                                                        | Planned (Store in `data/raw/reddit_politeness/`) |
| **LLM Outputs (Generated by this project)** |                                                                                |                                                                               |                                          |
|                                           | Outputs from LLMs based on designed prompts (CSV/JSON)                         | Generated in project                                                          | To be generated (Store in `data/raw/llm_outputs/`) |

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
*(This is a subset, see `requirements.txt` for all packages. Ensure `perspective-api-client==1.5.0` is installed if using Perspective API.)*

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
    # If perspective-api-client==1.5.0 was problematic, try installing it separately
    # or ensure your Python/pip versions are compatible.
    ```
    *Note:* You may need Java 1.8+ on PATH to run MAT 1.3 if you choose to use it.

4.  **Download NLTK Data (Brown Corpus, Punkt Tokenizer):**
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
    *   (Optional) If you prefer to have NLTK data within the project's `data/raw/nltk_data` directory, you can manually set the `NLTK_DATA` environment variable to point to `./data/raw/nltk_data` before running Python scripts, or configure NLTK within Python to look at this path. You would then need to manually copy the `corpora/brown` and other downloaded resources from the default NLTK path to `./data/raw/nltk_data/corpora/`.

5.  **Download spaCy Models:**
    A small English model from spaCy is useful for various NLP tasks.
    ```bash
    python -m spacy download en_core_web_sm
    ```

6.  **Download Other Datasets:**
    *   For datasets like COCA Sample, Reddit Comments, StereoSet, etc., download them from their respective sources (Kaggle, Hugging Face).
    *   Place the raw data files into the appropriate subdirectories within `data/raw/` as specified in the "Datasets and Tools" section table (e.g., `data/raw/coca_sample/`, `data/raw/stereoset/`).

7.  **Run the Demo Notebook:**
    *   Start Jupyter Lab:
        ```bash
        jupyter lab
        ```
    *   Navigate to the `notebooks/` directory and open `corpus_analysis_demo.ipynb`.
    *   Execute the cells in the notebook to see a demonstration of the corpus analysis utilities and visualization functions.

## Project Timeline & Milestones (Approx. 8–10 weeks)

| Week     | Milestone                                                                 | Key Deliverables                      |
| :------- | :------------------------------------------------------------------------ | :------------------------------------ |
| 1        | Project setup & literature refresh                                        | README outline; environment created   |
| 2        | Prompt design & LLM output collection (≥4 registers, ≥3 demo. templates)  | `data/raw/llm_outputs.csv` (initial)  |
| 3        | Reference corpus download & cleaning (e.g., Brown)                        | `data/processed/reference_corpus/`    |
| 4        | Style feature extraction (lexical, syntactic, discourse)                    | Notebook #1 (Styling), feature CSVs   |
| 5        | Multidimensional/register analysis & visualisation                        | Notebook #2 (MDA), dimension plots    |
| 6        | Bias & toxicity audit using benchmarks & Perspective API                  | Notebook #3 (Bias), bias metrics table|
| 7        | LLM-assisted annotation pilot (speech acts / politeness)                    | Notebook #4 (Annotation), label eval. |
| 8        | Synthesis report, dashboard, portfolio polish                             | `report.md`, interactive HTML figs    |
| 9-10 (Stretch) | Domain adaptation: fine-tune small model on curated corpus              | Fine-tuned model card                 |

## Detailed Workflow

### Phase 0 – Setup
1.  Fork / create a GitHub repo: `llm-corpus-audit`.
2.  Attach a Project board / issues mirroring milestones.
3.  Install packages (see `requirements.txt`); run `python -m spacy download en_core_web_sm`; download NLTK data (`brown`, `punkt`).

### Phase 1 – Data Collection & Curation
1.  **Prompt engineering:** Design ~20 prompt templates across registers (academic abstract, news lead, Reddit post, email, etc.) and demographic templates ("The <profession> is…", "A <nationality> person…").
2.  **Generation:** Use `transformers` pipeline (Llama-2) and/or OpenAI API to sample ~5 completions per prompt → ~200 outputs.
3.  Save to `data/raw/llm_outputs/` (e.g., as CSV) with metadata: `prompt_id`, `prompt_text`, `register`, `demographic_var`, `model_name`, `output_text`, `generation_date`.
4.  **Reference corpora:** Download Brown Corpus via NLTK. For other corpora (COCA, Reddit), download and place in `data/raw/`. Sample equal-size subsets matching registers if needed, saving processed versions to `data/processed/reference_corpus/`.

### Phase 2 – Stylistic Profiling
1.  **Feature extraction and analysis modules** (`src/utils/corpus_utils.py`, `src/analysis/llm_analysis.py`):
    *   Lexical: type-token ratio (lexicalrichness), hapax legomena ratio, average word frequency.
    *   Syntactic: mean sentence length, parse tree depth (spaCy), POS tag distribution.
    *   Discourse: pronoun rate, connectives list, stance markers.
    *   Comparative analysis of LLM outputs against reference corpus.
2.  Optionally run **MAT 1.3** on outputs & reference corpora; parse results.
3.  **Analysis and Visualization notebook** (`notebooks/corpus_analysis_demo.ipynb` and subsequent notebooks): Use `src/visualization/visualization.py` for plotting. Conduct PCA / clustering to visualise how LLM outputs align with human registers.

### Phase 3 – Bias & Toxicity Evaluation
1.  Load StereoSet & HolisticBias datasets from `data/raw/` (after downloading from Hugging Face). Use model to complete each context; compute score deltas.
2.  Run generated outputs (from `data/raw/llm_outputs/`) through **Perspective API** for toxicity / insult / profanity scores.
3.  Aggregate by demographic group; test for significant differences (chi-square, t-test).
4.  Visualise with `seaborn` (boxplots, heatmaps) via `src/visualization/visualization.py`.

### Phase 4 – LLM-Assisted Annotation
1.  Choose small sample (e.g., 200 Reddit comments from `data/raw/reddit_comments/` or your own forum sample).
2.  Use **spaCy-LLM** or OpenAI API to label each with speech-act tags (question, request, apology, etc.).
3.  Manually code a subset (e.g., 50 examples); measure precision/recall of LLM labels.
4.  Iterate prompt to improve; document gains.
5.  Export final annotated dataset to `data/processed/annotated_data/` + evaluation metrics.

### Phase 5 – Reporting & Portfolio Assets
1.  Compile findings into `report.md` (executive summary, methodology, key charts).
2.  Convert notebooks to HTML; host via GitHub Pages (optional).
3.  Create short demo video / GIF showing interactive dashboards (Plotly) and store in `media/`.
4.  Write LinkedIn post summarising insights and linking to repo.

### Stretch Goal – Domain Adaptation (Optional)
*   Curate a small **finance corpus** (~5 MB) and store in `data/raw/finance_corpus/`.
*   Continue-pretrain a 125 M parameter GPT-Neo on it using `accelerate`.
*   Evaluate before/after on finance Q&A prompts.
*   Publish model card to HF Hub.

## Evaluation Rubric for Your Own Learning

| Skill                         | Evidence in Project                                                |
| :---------------------------- | :----------------------------------------------------------------- |
| Corpus design & cleaning      | Data pipeline scripts/notebooks, documented sampling decisions     |
| Feature engineering & MDA     | `src/utils/corpus_utils.py`, PCA plots with interpretation         |
| Statistical evaluation        | Bias significance tests, diversity metrics, comparative analysis   |
| LLM integration / prompting   | Prompt templates, spaCy-LLM config, annotation accuracy table      |
| Visualization & storytelling  | Interactive plots (Plotly), `report.md`                            |
| Reproducible code             | `requirements.txt`, this README, well-commented scripts/notebooks  |

## Contributing

Contributions are welcome! Please adhere to good coding practices, document your changes, and ensure your contributions align with the project goals. (Further contribution guidelines can be developed if needed).

## Final Deliverables Checklist
- [ ] GitHub repo with MIT license & this detailed README.
- [ ] Jupyter notebooks (e.g., `notebooks/corpus_analysis_demo.ipynb`, plus others for data processing, style analysis, bias audit, annotation).
- [ ] `data/` folder populated with raw & processed data (or scripts to download/generate).
- [ ] `src/` folder with modular Python functions:
    - [ ] `src/utils/corpus_utils.py` (corpus loading, feature extraction)
    - [ ] `src/analysis/llm_analysis.py` (comparative LLM analysis)
    - [ ] `src/visualization/visualization.py` (plotting functions)
- [ ] `report.md` (≈2-3 k words) with key findings & next steps.
- [ ] Short demo video / GIF (in `media/`).
- [ ] (Optional) Fine-tuned model + model card.

## References & Further Reading
*   Biber, D. (1988). *Dimensions of Register Variation*.
*   Guo et al. (2024). "Benchmarking Linguistic Diversity of LLMs."
*   Gallegos et al. (2023). "Bias and Fairness in LLMs."
*   spaCy-LLM documentation: https://spacy.io/universe/project/spacy-llm
*   Multidimensional Analysis Tagger: https://ucrel-web.lancs.ac.uk/mat/
*   Stanford HELM benchmark: https://crfm.stanford.edu/helm/latest/
*   Jigsaw Perspective API: https://perspectiveapi.com

## Next Steps After Completion
1.  Present findings at a local NLP meetup or record a talk.
2.  Pitch the project in job applications: "*I built a full corpus-based audit of an LLM's style, bias, and consistency, integrating modern AI tools with linguistic theory.*"
3.  Extend project to new domains (healthcare, legal) to further showcase adaptability.

> *Start by opening a GitHub repo and creating the milestones board – everything else will flow from there!* 