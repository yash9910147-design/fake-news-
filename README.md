# Fake News Detection Project

<img src="https://socialify.git.ci/kapilsinghnegi/Fake-News-Detection/image?description=1&font=Source%20Code%20Pro&forks=1&issues=1&language=1&name=1&owner=1&pattern=Charlie%20Brown&pulls=1&stargazers=1&theme=Dark" alt="Fake-News-Detection" width="1280" height="320" />


This project trains a classical NLP model to classify article text as **Fake** or **Real**. It combines word-level and character-level TF-IDF features and compares three supervised classifiers.

1. **Logistic Regression**
2. **Linear SVM**
3. **Multinomial Naive Bayes**

## Project Overview

The project demonstrates text preprocessing, TF-IDF feature extraction, model selection, held-out evaluation, and a Gradio interface. It is an educational classifier, not a source-verification or fact-checking system.

## Dataset

We have used a labelled dataset containing news articles along with their corresponding labels (true or false). The dataset is divided into two classes:
- True: Genuine news articles
- False: Fake or fabricated news articles

The workflow normalizes text, removes empty and exact duplicate cleaned articles, and combines title and body text. Each pipeline unions word TF-IDF (unigrams and bigrams) with character TF-IDF (3-5 character sequences within word boundaries). It runs two separate benchmarks: a reproducible stratified random split (64%/16%/20%) and a chronological split that keeps each publication date together. For each benchmark, validation macro-F1 selects among Logistic Regression, Linear SVM, and Multinomial Naive Bayes; the selected model is refit on that benchmark's training plus validation articles, then evaluated on its untouched test split.

The evaluation reports accuracy, macro-F1, ROC-AUC, a confusion matrix, and per-class precision/recall/F1. The chronological test covers the latest portion of the dated data; articles without a parseable publication date are excluded from that benchmark and counted in the report. The model is saved as `models/fake_news_tfidf.joblib`; `models/fake_news_tfidf.metadata.json` records both benchmarks, date ranges, class counts, dataset fingerprint, feature settings, training time, and library versions.

Short inputs are handled by a separate headline-only Logistic Regression model trained from the dataset's `title` column and saved as `models/fake_news_headline.joblib`. Inputs up to 280 cleaned characters use this headline model; longer article text uses the original full-article model. Model settings and a decision threshold are selected on validation splits, then evaluated on untouched random and chronological test splits; the chronological benchmark is the more conservative estimate. The benchmark and threshold are recorded in `models/fake_news_headline.metadata.json`. Neither model is a fact-check or proven accurate on current news.

## System Requirements 

Hardware :
1. 4GB RAM
2. i3 Processor
3. 500MB free space

Software :
1. Anaconda
2. Python

## Dependencies

Before running the code, make sure you have the following libraries and packages installed:

- Python 3
- Scikit-learn and Joblib for training

Install the runtime dependencies using pip:

```bash
pip install -r requirements.txt
```

For model training, install the additional pandas dependency:

```bash
pip install -r requirements-training.txt
```

Training commands save a compressed portable inference artifact alongside the scikit-learn model. The API runtime uses the portable artifact and does not require pandas, NumPy, or scikit-learn.

The Gradio demo has an additional optional dependency. Install `requirements-gradio.txt` only when you want to run the legacy Gradio interface.

## Usage

1. Clone this repository to your local machine:

```bash
git clone https://github.com/kapilsinghnegi/Fake-News-Detection.git
```

2. Navigate to the project directory:

```bash
cd fake-news-detection
```

3. Install training dependencies, then train and compare the TF-IDF text classifiers:

```bash
pip install -r requirements-training.txt
python fake_news_detector.py
```

The script reads `Datasets/datasets/Fake.csv` and `Datasets/datasets/True.csv`, evaluates both random and chronological benchmarks, and saves the random-split model plus a JSON report containing both evaluations.

4. Train the separate headline classifier (needed for short headline inputs):

```bash
python fake_news_detector.py --train-headline-model
```

This creates `models/fake_news_headline.joblib` and its portable inference artifact without overwriting the full-article model. Keep the `.inference.json.gz` artifacts and metadata files when running the API.

5. Classify a headline or article:

```bash
python fake_news_detector.py --predict "Paste the news article text here"
```

Short inputs use the headline model and longer inputs use the full-article model. The displayed decision score is not a calibrated probability. A prediction is a learned text-pattern classification, not a fact-check or a guarantee that a story is true.

## Tests

Install development dependencies and run the focused test suite:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Gradio Demo

Install the app dependencies and launch the web interface:

```bash
pip install -r requirements-gradio.txt
python app.py
```

Running `python app.py` opens the app in your browser at `http://127.0.0.1:7860`. The interface includes **Analyze News**, **Training & Test Data**, and **Model Report** tabs. NLP feature extraction and model selection stay in the backend. Keep the terminal process running while you use it; press `Ctrl+C` to stop it. Train first with `python fake_news_detector.py` if the saved model or report is missing.

For a temporary public link you can share with your professor, start the app with sharing enabled:

```bash
GRADIO_SHARE=1 python app.py
```

The share URL is public and temporary. Use only non-sensitive sample text. The **Training & Test Data** tab reads `Datasets/datasets/Fake.csv` and `Datasets/datasets/True.csv`; include both files when deploying to a persistent Hugging Face Space, along with `app.py`, `fake_news_detector.py`, `requirements.txt`, `models/fake_news_tfidf.joblib`, and `models/fake_news_tfidf.metadata.json`.

## Vercel Deployment

The Vercel build serves the Vite frontend and FastAPI backend from one project. Analysis history uses MongoDB when `MONGODB_URI` is configured; Vercel does not use the local SQLite database because its filesystem is ephemeral.

1. Create a MongoDB Atlas database and a database user. Allow connections from your deployment environment in Atlas Network Access.
2. Link this repository in Vercel or run `vercel` from the project root.
3. In Vercel Project Settings → Environment Variables, set:
   - `MONGODB_URI`: the Atlas connection URI (keep this secret; do not commit it).
   - `MONGODB_DATABASE`: `truthlens` (optional; this is the default).
   - `NEWSAPI_KEY` and `FACTCHECK_API_KEY` (optional provider keys).
4. Redeploy after setting environment variables.

The production health endpoint is `/api/v1/health`; the API documentation is `/docs`. Without `MONGODB_URI`, Vercel API calls deliberately return a configuration error rather than writing history to ephemeral SQLite storage.

## Results

The **Model Report** tab reports the full-article benchmark. On the headline model's current dated dataset split (test dates begin 2017-10-17; 7,817 test headlines), validation-selected Logistic Regression with a non-zero decision threshold reached 98.16% chronological test accuracy and 95.34% macro-F1; Fake recall was 86.35% and Real recall was 99.74%. The random headline test reached 96.07% accuracy and 96.05% macro-F1. The original full-article model reports 99.69% random-split accuracy and 99.85% chronological-split accuracy. Every score is limited to this dataset and is not an estimate of current real-world fact-check accuracy.

## Limitations

The dataset may encode publisher, topic, writing-style, and time-period artifacts. The chronological test is still not source-independent, and its latest articles are from 2018; neither benchmark establishes performance on current news or new publishers. A separate, recent dataset with different sources is needed to assess that. Predictions are not evidence that an article is true or false; use independent sources and human review.
---

## Project Screenshots

#### Not a Fake News
![Not a Fake News](https://github.com/kapilsinghnegi/Fake-News-Detection/assets/118688453/3d079c46-118a-4c53-a515-43b9146001c5)

#### Fake News
![Fake News](https://github.com/kapilsinghnegi/Fake-News-Detection/assets/118688453/2f5262f7-801d-4293-824c-13c29fb97fed)
