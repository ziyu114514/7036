# NLP-Based Stock Strategy from Research Reports

## Project Overview

This project builds a **complete quantitative research pipeline** that extracts investment signals from **sell-side research reports** using NLP techniques. It automates the entire workflow from **data acquisition → text cleaning → NLP modeling → factor construction → portfolio construct → backtesting**, enabling systematic evaluation of how research report sentiment, tone, and content relate to stock performance.

The goal is to transform unstructured financial research reports into **actionable alpha factors** for quantitative investment strategies.

## Key Features

- 📥 **Automated Data Collection**  
  Downloads research reports (PDF/HTML) from Eastmoney with metadata, page filtering, and retry logic.

- 🧹 **High-Precision Text Cleaning Pipeline**  
  Multi-stage blank-aware, table-aware, and LLM-based semantic filtering.

- 🧠 **NLP Modeling for Investment Signals**  
  Extracts sentiment, forward-looking tone, risk disclosures, and topic distributions.

- 📈 **Alpha Factor Construction**  
  Converts NLP outputs into quantitative factors (sentiment factor, novelty factor, topic factor, etc.).

- 🔄 **Backtesting Framework**  
  Evaluates factor performance using standard quant metrics (IC, IR, long-short returns).

- 🧪 **Reproducible Research Workflow**  
  All intermediate data is saved for debugging, auditing, and future analysis.

## System Architecture
```
┌──────────────────────────┐
│  Project   Work   Flow   │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 1.Data Acquisition       │
│ - API crawling           │
│ - Page filtering         │
│ - Metadata extraction    │
│ - Trade data acquisition │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 2.Chinese Text Cleaning  │
│ - PDF OCR layout analysis│
│ - Line-wise Text detect  │
| - Blank-aware filtering  │
│ - Table-aware filtering  │
│ - RE filtering           │
│ - LLM semantic filtering │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 3.Ch-En Text Translation │
│ - Sentence level  merge  │
│ - Paragraph level merge  │
| - LLM Ch to En translate │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 4.English Text Cleaning  │
│ - RE filtering           │
│ - Paragraph level merge  │
| - LLM semantic filtering │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 5.Data Transformation    │
│ - (Optional)             │
│ - Word level tokenization│
| - Quarterly data reform  │
| - Monthly data reform    │
| - Handle missing data    │
│ - Assign weight by depth │
│ report or not            │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 6.Data Merge             │
│ - Chronologically reform │
│ trading data             │
│ - Match corresponding    │
│ text data                │
| - Handle unmatched data  │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 7.Data Analysis          │
│ (With pretrained model)  │
│ - Sentiment analysis     │
│ - Topic modeling (LDA)   │
│ - Forward-looking tone   │
│ - Risk disclosure scoring│
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 8.Data Analysis          │
│(Fine tune pretrain model)│
│ - Tagging label using    │
│ report conclusion eg.    │
│`Buy` or `Sell`           │
│ - Tagging label using    │
│ trading data eg. `Return`│
│ - Fine tune model based  │
│ on labels                │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 9.Data Analysis          │
│(Train model from scratch)│
│ - Small semantic model   │
│ - LLM such as Transformer│
│ LSTM, RNN, etc.          │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 10.Factor Construction   │
│ - Sentiment factor       │
│ - Novelty factor         │
│ - Topic factor           │
│ - Report intensity factor│
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 11.Portfolio Construction│
│ - Factor Weighted        │
│ sole Long and Long-short │
│ Portfolio                │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 12.Backtesting Framework │
│ - IC/IR evaluation       │
│ - Benchmark Compare      │
│ - Factor Combination     │
└─────────────┬────────────┘
              ▼
┌──────────────────────────┐
│ 13.Result Visualization  │
│ - Project Report         │
│ - Diagram Making         │
│ - Interactive Webpage    │
└──────────────────────────┘
```
## Usage
- 1. **Download Research Reports**
```bash
python download_reports.py
```
- 2. **Clean and Structure Text**
```bash
python clean_text.py
```
- 3. **Run NLP Models**
```bash
python run_nlp.py
```
- 4. **Generate Factors**
```bash
python generate_factors.py
```
- 5. **Backtest Strategy**
```bash
python backtest.py
```
## File Structure
```
project_root/
├── reports_pdf/
│   ├── StockName1/
│   │   ├── DepthReport/
│   │   └── Report.pdf
│   └── StockName2/
├── clean_txt/
│   ├── StockName1/
│   │   ├── DepthReport/
│   │   └── Report.txt
│   └── StockName2/
├── nlp_outputs/
│   ├── sentiment/
│   ├── topics/
│   └── risk_scores/
├── factors/
│   ├── sentiment_factor.csv
│   ├── topic_factor.csv
│   └── novelty_factor.csv
├── backtest_results/
│   ├── ic_analysis.png
│   ├── long_short_curve.png
│   └── summary.csv
└── config.json
```
## NLP Modeling Details
- 1. **Sentiment Analysis**
Fine-tuned financial sentiment classifier
Outputs: [-1, 0, 1] or continuous score
- 2. **Topic Modeling (LDA / BERT)**
Identifies themes such as:
Company fundamentals
Industry outlook
Risk warnings
Policy impact
- 3. **Forward-Looking Tone**
Measures how much the report discusses future expectations.
- 4. **Risk Disclosure Scoring**
Counts and weights risk-related sentences.

## Factor Construction
- `Sentiment Factor`:	Average sentiment score of the report
- `Novelty Factor` :	Measures new information vs. previous reports
- `Topic Factor` :	Topic distribution mapped to returns
- `Report Intensity Factor`: Number of reports / length / density

## Changelog
- v1.0.0: Basic pipeline (download → pdf text extract and trading data acquire → data merge → NLP(key word match) → long-short portfolio → back test)
- v1.1.0: Tested Chinese to English translation basic feature on several files, pipeline not implement yet.
- v1.2.0: Rewrite pdf to txt pipeline using OCR, implement layout analysis, blank-aware, table-aware and RE text cleaning for Chinese version report. Formed pdf to txt convert pipeline.
- v1.3.0: Implement Embedding NLP model for high-level txt noise clean. Tested on several files, pipeline not implement yet.

### What's next?
Implement Chinese text NLP cleaning, Chinese to English txt
