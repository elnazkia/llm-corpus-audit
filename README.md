# LLM Stylistic Profiling & Bias Audit .

A data-centric corpus linguistics project that demonstrates the application of corpus linguistic methods to evaluate and analyze Large Language Models (LLMs).

## Project Overview

This project combines traditional corpus linguistics techniques with modern LLM analysis to:
- Profile LLM outputs across different registers and styles
- Evaluate bias and toxicity using established benchmarks
- Demonstrate LLM-assisted corpus annotation
- Provide insights into model behavior through linguistic features

## Project Structure

```
llm-corpus-audit/
├── data/
│   ├── raw/          # Original data files
│   └── processed/    # Cleaned and processed data
├── scripts/          # Python scripts for data processing and analysis
├── notebooks/        # Jupyter notebooks for analysis and visualization
├── media/           # Images, videos, and other media files
└── requirements.txt  # Python package dependencies
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/elnazkia/llm-corpus-audit.git
cd llm-corpus-audit
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('brown')"
```

## Project Timeline

- Week 1: Project setup & literature review
- Week 2: Data collection & prompt engineering
- Week 3: Reference corpus preparation
- Week 4: Feature extraction & analysis
- Week 5: Style profiling & visualization
- Week 6: Bias & toxicity evaluation
- Week 7: LLM-assisted annotation
- Week 8: Documentation & portfolio preparation

## Contributing

This is a portfolio project. Feel free to fork and adapt for your own learning purposes.

## License

MIT License - see LICENSE file for details 