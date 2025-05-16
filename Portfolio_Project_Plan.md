# Portfolio Project Plan: Corpus Linguistics Meets LLMs

## 1. Project Title
**"LLM Stylistic Profiling & Bias Audit: A Data-Centric Corpus Linguistics Showcase"**

## 2. Project Goal
Design and execute an end-to-end workflow that:
1. Collects and curates an evaluation corpus of LLM outputs across multiple registers and demographic prompts.
2. Quantitatively profiles style, diversity, and consistency of those outputs using corpus-linguistic methods (MDA-inspired feature analysis, lexical diversity, syntactic complexity, etc.).
3. Audits bias and toxicity with open fairness benchmarks (StereoSet, HolisticBias, RealToxicityPrompts) and corpus statistics.
4. Demonstrates an **LLM-in-the-loop annotation pipeline** (e.g., speech-act or politeness tagging) to accelerate manual corpus work.
5. Produces interactive visualisations & a written report suitable for a public GitHub portfolio and job applications.

By completing the project you will illustrate expertise in:
• Interpretability & explainability of LLM outputs via linguistic features.
• Evaluation & validation (style drift, bias, toxicity).
• Data curation / domain adaptation principles.
• LLM-augmented corpus workflows (spaCy-LLM, Label Studio AI-assist).

---

## 3. Learning Outcomes
• Apply classic corpus methods to modern LLM evaluation.
• Build reproducible Python pipelines (datasets → analysis → visualisation).
• Use open-source LLMs/APIs programmatically.
• Communicate findings in clear, portfolio-ready artefacts (notebooks + blog-style README).

---

## 4. Public Data & Resources
| Purpose | Dataset / Tool | Link |
|---------|----------------|------|
| Reference corpora for stylistic baselines | Brown Corpus (NLTK), COCA Sample, Reddit Comments Corpus | NLTK, Kaggle |
| Bias evaluation | StereoSet, HolisticBias, RealToxicityPrompts | Hugging Face Datasets |
| Toxicity scoring | Perspective API (Jigsaw) | https://developers.perspectiveapi.com |
| MDA feature extraction | Multidimensional Analysis Tagger (MAT) 1.3 (Java) OR Biber feature list via spaCy / custom scripts | https://ucrel-web.lancs.ac.uk/mat/ |
| LLM model | Llama-2-7B-Chat (HF), OpenAI GPT-3.5/4 for comparison | Hugging Face / OpenAI |
| Annotation sample | Switchboard Dialog Act, Reddit Politeness Corpus, or your own forum sample | UC Berkeley / Kaggle |

---

## 5. Required Python Packages
Create a fresh environment (Python ≥3.10) and install:
```bash
pip install pandas numpy scipy nltk spacy spacy-llm lexicalrichness textstat
pip install matplotlib seaborn wordcloud scattertext plotly
pip install transformers accelerate sentencepiece datasets evaluate huggingface_hub
pip install perspective-api-client==1.5.0
pip install jupyterlab notebook
pip install label-studio==1.11.0
```
(Optional) If using OpenAI:
```bash
pip install openai tiktoken
```
*Note:* You may need Java 1.8+ on PATH to run MAT 1.3.

---

## 6. Project Timeline & Milestones (8–10 weeks, ~5–8 h/week)
| Week | Milestone | Key Deliverables |
|------|-----------|------------------|
| 1 | Project setup & literature refresh | README outline; environment created |
| 2 | Prompt design & LLM output collection (≥4 registers, ≥3 demographic templates) | `data/llm_outputs.csv` |
| 3 | Reference corpus download & cleaning | `data/reference/` |
| 4 | Style feature extraction (lexical, syntactic, discourse) | Notebook #1, feature CSVs |
| 5 | Multidimensional/register analysis & visualisation | Notebook #2, dimension plots |
| 6 | Bias & toxicity audit using benchmarks & Perspective API | Notebook #3, bias metrics table |
| 7 | LLM-assisted annotation pilot (speech acts / politeness) | Notebook #4, label evaluation |
| 8 | Synthesis report, dashboard, portfolio polish | `report.md`, interactive HTML figs |
| 9-10 (stretch) | Domain adaptation: fine-tune small model on curated corpus | Fine-tuned model card |

---

## 7. Detailed Workflow
### Phase 0 – Setup
1. Fork / create a GitHub repo: `llm-corpus-audit`.
2. Attach a Project board / issues mirroring milestones.
3. Install packages; run `python -m spacy download en_core_web_sm`; download NLTK data.

### Phase 1 – Data Collection & Curation
1. **Prompt engineering:** Design ~20 prompt templates across registers (academic abstract, news lead, Reddit post, email, etc.) and demographic templates ("The <profession> is…", "A <nationality> person…").
2. **Generation:** Use `transformers` pipeline (Llama-2) and/or OpenAI API to sample 5 completions per prompt → ~200 outputs.
3. Save to CSV with metadata: prompt_type, register, demographic_var, model_name, date.
4. **Reference corpora:** Download Brown Corpus via NLTK; sample equal-size subsets matching registers. Clean & save.

### Phase 2 – Stylistic Profiling
1. **Feature extraction scripts** (`scripts/features.py`):
   • Lexical: type-token ratio (lexicalrichness), hapax legomena ratio, average word frequency.
   • Syntactic: mean sentence length, parse tree depth (spaCy), POS tag distribution.
   • Discourse: pronoun rate, connectives list, stance markers.
2. Optionally run **MAT 1.3** on outputs & reference corpora; parse results.
3. **Analysis notebook**: PCA / clustering to visualise how LLM outputs align with human registers.

### Phase 3 – Bias & Toxicity Evaluation
1. Load StereoSet & HolisticBias datasets (Hugging Face). Use model to complete each context; compute score deltas.
2. Run generated outputs through **Perspective API** for toxicity / insult / profanity scores.
3. Aggregate by demographic group; test for significant differences (chi-square, t-test).
4. Visualise with seaborn (boxplots, heatmaps).

### Phase 4 – LLM-Assisted Annotation
1. Choose small sample (e.g., 200 Reddit comments).
2. Use **spaCy-LLM** or OpenAI API to label each with speech-act tags (question, request, apology, etc.).
3. Manually code 50 examples; measure precision/recall of LLM labels.
4. Iterate prompt to improve; document gains.
5. Export final annotated dataset + evaluation metrics.

### Phase 5 – Reporting & Portfolio Assets
1. Compile findings into `report.md` (executive summary, methodology, key charts).
2. Convert notebooks to HTML; host via GitHub Pages.
3. Create short demo video / GIF showing interactive dashboards (Plotly).
4. Write LinkedIn post summarising insights and linking to repo.

### Stretch Goal – Domain Adaptation (Optional)
• Curate a small **finance corpus** (~5 MB) and continue-pretrain a 125 M parameter GPT-Neo on it using `accelerate`.
• Evaluate before/after on finance Q&A prompts.
• Publish model card to HF Hub.

---

## 8. Evaluation Rubric for Your Own Learning
| Skill | Evidence in Project |
|-------|--------------------|
| Corpus design & cleaning | Data pipeline scripts, documented sampling decisions |
| Feature engineering & MDA | Feature repo, PCA plots with interpretation |
| Statistical evaluation | Bias significance tests, diversity metrics |
| LLM integration / prompting | Prompt templates, spaCy-LLM config, annotation accuracy table |
| Visualization & storytelling | Interactive plots, report.md |
| Reproducible code | `requirements.txt`, README, Makefile or CLI scripts |

---

## 9. Final Deliverables Checklist
- [ ] GitHub repo with MIT license & detailed README.
- [ ] 4+ Jupyter notebooks (data, style analysis, bias audit, annotation).
- [ ] `data/` folder (raw & processed, or scripts to download).
- [ ] `scripts/` with modular functions (feature extraction, analysis).
- [ ] `report.md` (≈2-3 k words) with key findings & next steps.
- [ ] Short demo video / GIF (in `media/`).
- [ ] (Optional) Fine-tuned model + model card.

---

## 10. References & Further Reading
• Biber, D. (1988). *Dimensions of Register Variation*.
• Guo et al. (2024). "Benchmarking Linguistic Diversity of LLMs."
• Gallegos et al. (2023). "Bias and Fairness in LLMs."
• spaCy-LLM documentation: https://spacy.io/universe/project/spacy-llm
• Multidimensional Analysis Tagger: https://ucrel-web.lancs.ac.uk/mat/
• Stanford HELM benchmark: https://crfm.stanford.edu/helm/latest/
• Jigsaw Perspective API: https://perspectiveapi.com

---

### Next Steps After Completion
1. Present findings at a local NLP meetup or record a talk.
2. Pitch the project in job applications: "*I built a full corpus-based audit of an LLM's style, bias, and consistency, integrating modern AI tools with linguistic theory.*"
3. Extend project to new domains (healthcare, legal) to further showcase adaptability.

> *Start by opening a GitHub repo and creating the milestones board – everything else will flow from there!* 