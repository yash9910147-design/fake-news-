import argparse
import gzip
import hashlib
import html
import json
import math
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Datasets" / "datasets"
MODEL_PATH = ROOT / "models" / "fake_news_tfidf.joblib"
HEADLINE_MODEL_PATH = ROOT / "models" / "fake_news_headline.joblib"
MODEL_INFERENCE_PATH = ROOT / "models" / "fake_news_tfidf.inference.json.gz"
HEADLINE_INFERENCE_PATH = ROOT / "models" / "fake_news_headline.inference.json.gz"
LABELS = {0: "Fake", 1: "Real"}
RANDOM_STATE = 42
HEADLINE_MAX_LENGTH = 280
TFIDF_SETTINGS = {
    "word_ngrams": {
        "ngram_range": [1, 2],
        "max_features": 100_000,
        "min_df": 2,
        "stop_words": "english",
        "sublinear_tf": True,
    },
    "character_ngrams": {
        "analyzer": "char_wb",
        "ngram_range": [3, 5],
        "max_features": 50_000,
        "min_df": 2,
        "sublinear_tf": True,
    },
}


def clean_text(value):
    text = html.unescape(str(value)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(data_dir=DATA_DIR):
    import pandas as pd

    fake = pd.read_csv(data_dir / "Fake.csv")
    real = pd.read_csv(data_dir / "True.csv")

    for name, frame in (("Fake.csv", fake), ("True.csv", real)):
        if "text" not in frame.columns:
            raise ValueError(f"{name} must contain a 'text' column")

    fake["label"] = 0
    real["label"] = 1
    data = pd.concat([fake, real], ignore_index=True)
    data["title"] = data["title"].fillna("") if "title" in data else ""
    data["text"] = data["text"].fillna("")
    data["news"] = (data["title"] + " " + data["text"]).map(clean_text)
    data["published_at"] = (
        pd.to_datetime(data["date"], format="mixed", errors="coerce", utc=True)
        if "date" in data
        else pd.NaT
    )
    data = data.loc[
        data["news"].str.len() > 0, ["news", "label", "published_at"]
    ]
    data = data.drop_duplicates(subset="news").sample(frac=1, random_state=42)

    if data["label"].nunique() != 2:
        raise ValueError("Both Fake.csv and True.csv must contain usable articles")
    return data.reset_index(drop=True)


def build_pipeline(model):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline

    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_ngrams",
                            TfidfVectorizer(
                                ngram_range=tuple(
                                    TFIDF_SETTINGS["word_ngrams"]["ngram_range"]
                                ),
                                max_features=TFIDF_SETTINGS["word_ngrams"]["max_features"],
                                min_df=TFIDF_SETTINGS["word_ngrams"]["min_df"],
                                stop_words=TFIDF_SETTINGS["word_ngrams"]["stop_words"],
                                sublinear_tf=TFIDF_SETTINGS["word_ngrams"]["sublinear_tf"],
                            ),
                        ),
                        (
                            "character_ngrams",
                            TfidfVectorizer(
                                analyzer=TFIDF_SETTINGS["character_ngrams"]["analyzer"],
                                ngram_range=tuple(
                                    TFIDF_SETTINGS["character_ngrams"]["ngram_range"]
                                ),
                                max_features=TFIDF_SETTINGS["character_ngrams"]["max_features"],
                                min_df=TFIDF_SETTINGS["character_ngrams"]["min_df"],
                                sublinear_tf=TFIDF_SETTINGS["character_ngrams"]["sublinear_tf"],
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", model),
        ]
    )


def split_dataset(data):
    from sklearn.model_selection import train_test_split

    train_validation, test = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_validation["label"],
    )
    return {
        "Training": train,
        "Validation": validation,
        "Test": test,
        "Training + validation": train_validation,
    }


def temporal_split_dataset(data):
    import pandas as pd

    undated_rows = int(data["published_at"].isna().sum())
    dated_data = data.dropna(subset=["published_at"]).copy()
    unique_dates = pd.Index(dated_data["published_at"].unique()).sort_values()
    if len(unique_dates) < 5:
        raise ValueError("At least five unique publication dates are required")

    validation_start = dated_data["published_at"].quantile(0.6)
    test_start = dated_data["published_at"].quantile(0.8)
    training = dated_data.loc[dated_data["published_at"] < validation_start]
    validation = dated_data.loc[
        (dated_data["published_at"] >= validation_start)
        & (dated_data["published_at"] < test_start)
    ]
    test = dated_data.loc[dated_data["published_at"] >= test_start]

    for split_name, frame in (
        ("training", training),
        ("validation", validation),
        ("test", test),
    ):
        if frame.empty or frame["label"].nunique() != 2:
            raise ValueError(
                f"The chronological {split_name} split must contain both labels"
            )

    splits = {
        "Training": training,
        "Validation": validation,
        "Test": test,
        "Training + validation": pd.concat([training, validation]).sort_index(),
    }
    boundaries = {
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "undated_rows": undated_rows,
    }
    return splits, boundaries


@lru_cache(maxsize=2)
def load_data_splits(data_dir=DATA_DIR):
    return split_dataset(load_dataset(data_dir))


def fit_and_evaluate(splits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    train = splits["Training"]
    validation = splits["Validation"]
    train_validation = splits["Training + validation"]
    test = splits["Test"]

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE, dual="auto"),
        "Multinomial Naive Bayes": MultinomialNB(),
    }
    fitted = {}
    for name, classifier in candidates.items():
        pipeline = build_pipeline(classifier)
        pipeline.fit(train["news"], train["label"])
        validation_predictions = pipeline.predict(validation["news"])
        validation_macro_f1 = f1_score(
            validation["label"],
            validation_predictions,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
        fitted[name] = (pipeline, validation_macro_f1)
        print(f"{name}: validation macro-F1 = {validation_macro_f1:.4f}")

    best_name, (selected_pipeline, _) = max(
        fitted.items(), key=lambda item: item[1][1]
    )
    best_pipeline = build_pipeline(selected_pipeline.named_steps["classifier"])
    best_pipeline.fit(train_validation["news"], train_validation["label"])
    test_predictions = best_pipeline.predict(test["news"])
    test_classifier = best_pipeline.named_steps["classifier"]
    if hasattr(test_classifier, "decision_function"):
        test_scores = best_pipeline.decision_function(test["news"])
    else:
        real_column = list(best_pipeline.classes_).index(1)
        test_scores = best_pipeline.predict_proba(test["news"])[:, real_column]
    test_macro_f1 = f1_score(
        test["label"], test_predictions, labels=[0, 1], average="macro", zero_division=0
    )
    test_roc_auc = (
        roc_auc_score(test["label"], test_scores)
        if test["label"].nunique() == 2
        else None
    )
    test_report = classification_report(
        test["label"],
        test_predictions,
        labels=[0, 1],
        target_names=[LABELS[0], LABELS[1]],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(test["label"], test_predictions),
        "macro_f1": test_macro_f1,
        "roc_auc": test_roc_auc,
        "confusion_matrix": confusion_matrix(
            test["label"], test_predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": test_report,
        "test_rows": len(test),
    }
    return {
        "model": best_name,
        "validation_macro_f1": fitted[best_name][1],
        "pipeline": best_pipeline,
        "test_metrics": metrics,
    }


def get_date_ranges(splits):
    return {
        name: {
            "start": frame["published_at"].min().isoformat(),
            "end": frame["published_at"].max().isoformat(),
        }
        for name, frame in splits.items()
        if name != "Training + validation"
    }


def train_and_evaluate(data_dir=DATA_DIR, model_path=MODEL_PATH):
    import pandas as pd
    import joblib
    import sklearn
    from sklearn.metrics import classification_report

    splits = load_data_splits(data_dir)
    random_results = fit_and_evaluate(splits)
    best_pipeline = random_results["pipeline"]
    random_metrics = random_results["test_metrics"]

    print(f"\nRandom split selected model: {random_results['model']}")
    print(f"Validation macro-F1: {random_results['validation_macro_f1']:.4f}")
    print(f"Test accuracy: {random_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {random_metrics['macro_f1']:.4f}")
    print(f"Test ROC-AUC: {random_metrics['roc_auc']:.4f}")
    print("Test confusion matrix (rows=true, columns=predicted):")
    print(random_metrics["confusion_matrix"])
    print("Test classification report:")
    print(
        classification_report(
            splits["Test"]["label"],
            best_pipeline.predict(splits["Test"]["news"]),
            labels=[0, 1],
            target_names=[LABELS[0], LABELS[1]],
            zero_division=0,
        )
    )
    temporal_results = None
    temporal_splits = None
    temporal_boundaries = None
    temporal_error = None
    try:
        data = pd.concat(
            [splits["Training + validation"], splits["Test"]]
        ).sort_index()
        temporal_splits, temporal_boundaries = temporal_split_dataset(data)
        temporal_results = fit_and_evaluate(temporal_splits)
        temporal_metrics = temporal_results["test_metrics"]
        print("\nChronological benchmark")
        print(f"Selected model: {temporal_results['model']}")
        print(f"Validation macro-F1: {temporal_results['validation_macro_f1']:.4f}")
        print(f"Test accuracy: {temporal_metrics['accuracy']:.4f}")
        print(f"Test macro-F1: {temporal_metrics['macro_f1']:.4f}")
        print(f"Test ROC-AUC: {temporal_metrics['roc_auc']:.4f}")
        print(f"Date cutoffs: {temporal_boundaries}")
    except ValueError as error:
        temporal_error = str(error)
        print(f"Chronological benchmark unavailable: {temporal_error}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path, compress=3)
    export_inference_model(best_pipeline, MODEL_INFERENCE_PATH)
    print(f"Saved model to: {model_path}")

    all_data = pd.concat(
        [splits["Training + validation"], splits["Test"]]
    ).sort_index()
    dataset_hash = hashlib.sha256(
        pd.util.hash_pandas_object(all_data, index=False).values.tobytes()
    ).hexdigest()
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": random_results["model"],
        "selection_metric": "validation_macro_f1",
        "validation_macro_f1": random_results["validation_macro_f1"],
        "test_metrics": random_metrics,
        "temporal_benchmark": {
            "available": temporal_results is not None,
            "error": temporal_error,
            "model": temporal_results["model"] if temporal_results else None,
            "validation_macro_f1": (
                temporal_results["validation_macro_f1"] if temporal_results else None
            ),
            "test_metrics": (
                temporal_results["test_metrics"] if temporal_results else None
            ),
            "date_boundaries": temporal_boundaries,
            "date_ranges": get_date_ranges(temporal_splits) if temporal_splits else None,
            "split_rows": (
                {name: len(frame) for name, frame in temporal_splits.items()}
                if temporal_splits
                else None
            ),
            "split_class_counts": (
                {
                    name: {
                        LABELS[int(label)]: int(count)
                        for label, count in frame["label"].value_counts().items()
                    }
                    for name, frame in temporal_splits.items()
                    if name != "Training + validation"
                }
                if temporal_splits
                else None
            ),
        },
        "dataset": {
            "rows": len(all_data),
            "sha256": dataset_hash,
            "split_rows": {
                name: len(frame)
                for name, frame in splits.items()
            },
            "labels": {str(label): name for label, name in LABELS.items()},
        },
        "features": {"tfidf": TFIDF_SETTINGS},
        "random_state": RANDOM_STATE,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "Predictions reflect patterns in this dataset and are not fact-checks.",
            "The dataset may contain source, topic, and time-period artifacts.",
            "Decision scores are not calibrated probabilities.",
            "The chronological benchmark is a time holdout, not a source-independent evaluation.",
        ],
    }
    metadata_path = model_path.with_name(f"{model_path.stem}.metadata.json")
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(f"Saved evaluation metadata to: {metadata_path}")
    return best_pipeline


def _headline_classifier_candidates():
    return [
        ("Logistic Regression", c, class_weight)
        for c in (0.3, 1.0, 3.0, 10.0)
        for class_weight in (None, "balanced")
    ]


def _headline_metrics(labels, scores, threshold):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    predictions = (np.asarray(scores) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, labels=[0, 1], average="macro", zero_division=0
        ),
        "roc_auc": roc_auc_score(labels, scores),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=[LABELS[0], LABELS[1]],
            output_dict=True,
            zero_division=0,
        ),
        "test_rows": len(labels),
    }


def _headline_threshold_metrics(labels, scores, threshold):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    predictions = (np.asarray(scores) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, labels=[0, 1], average="macro", zero_division=0
        ),
    }


def _select_headline_model(training, validation, test):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    candidates = []
    thresholds = None
    for model_name, c, class_weight in _headline_classifier_candidates():
        classifier = LogisticRegression(
            C=c,
            class_weight=class_weight,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        pipeline = build_pipeline(classifier)
        pipeline.fit(training["news"], training["label"])
        validation_scores = pipeline.decision_function(validation["news"])
        if thresholds is None:
            thresholds = np.unique(
                np.quantile(validation_scores, np.linspace(0, 1, 401))
            )

        best_threshold = None
        best_validation_metrics = None
        for threshold in thresholds:
            metrics = _headline_threshold_metrics(
                validation["label"], validation_scores, float(threshold)
            )
            if best_validation_metrics is None or (
                metrics["accuracy"],
                metrics["macro_f1"],
            ) > (
                best_validation_metrics["accuracy"],
                best_validation_metrics["macro_f1"],
            ):
                best_threshold = float(threshold)
                best_validation_metrics = metrics

        best_validation_metrics = _headline_metrics(
            validation["label"], validation_scores, best_threshold
        )
        candidates.append(
            {
                "model_name": model_name,
                "c": c,
                "class_weight": class_weight,
                "threshold": best_threshold,
                "validation_metrics": best_validation_metrics,
            }
        )

    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["validation_metrics"]["accuracy"],
            candidate["validation_metrics"]["macro_f1"],
        ),
    )
    final_pipeline = build_pipeline(
        LogisticRegression(
            C=selected["c"],
            class_weight=selected["class_weight"],
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    )
    fit_rows = pd.concat([training, validation]).sort_index()
    final_pipeline.fit(fit_rows["news"], fit_rows["label"])
    test_scores = final_pipeline.decision_function(test["news"])
    test_metrics = _headline_metrics(
        test["label"], test_scores, selected["threshold"]
    )
    selected["test_metrics"] = test_metrics
    selected["fit_rows"] = len(fit_rows)
    return final_pipeline, selected


def train_headline_model(data_dir=DATA_DIR, model_path=HEADLINE_MODEL_PATH):
    import pandas as pd
    import joblib
    import sklearn
    from sklearn.model_selection import train_test_split

    frames = []
    for filename, label in (("Fake.csv", 0), ("True.csv", 1)):
        frame = pd.read_csv(data_dir / filename)
        if "title" not in frame.columns:
            raise ValueError(f"{filename} must contain a 'title' column")
        frame = pd.DataFrame(
            {
                "news": frame["title"].fillna("").map(clean_text),
                "label": label,
                "published_at": (
                    pd.to_datetime(
                        frame["date"], format="mixed", errors="coerce", utc=True
                    )
                    if "date" in frame
                    else pd.NaT
                ),
            }
        )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.loc[data["news"].str.len() > 0]
    conflicting_titles = (
        data.groupby("news")["label"].nunique().loc[lambda labels: labels > 1].index
    )
    data = data.loc[~data["news"].isin(conflicting_titles)]
    data = data.drop_duplicates(subset="news").reset_index(drop=True)
    if data["label"].nunique() != 2:
        raise ValueError("Both Fake.csv and True.csv must contain usable headlines")

    random_train_validation, random_test = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    random_train, random_validation = train_test_split(
        random_train_validation,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=random_train_validation["label"],
    )
    _, random_selection = _select_headline_model(
        random_train, random_validation, random_test
    )

    dated = data.dropna(subset=["published_at"])
    validation_start = dated["published_at"].quantile(0.6)
    test_start = dated["published_at"].quantile(0.8)
    chronological_train = dated.loc[dated["published_at"] < validation_start]
    chronological_validation = dated.loc[
        (dated["published_at"] >= validation_start)
        & (dated["published_at"] < test_start)
    ]
    chronological_test = dated.loc[dated["published_at"] >= test_start]
    for split_name, split in (
        ("training", chronological_train),
        ("validation", chronological_validation),
        ("test", chronological_test),
    ):
        if split.empty or split["label"].nunique() != 2:
            raise ValueError(
                f"The chronological headline {split_name} split must contain both labels"
            )

    final_model, chronological_selection = _select_headline_model(
        chronological_train,
        chronological_validation,
        chronological_test,
    )
    temporal_boundaries = {
        "training_end": chronological_train["published_at"].max().isoformat(),
        "validation_start": chronological_validation["published_at"].min().isoformat(),
        "validation_end": chronological_validation["published_at"].max().isoformat(),
        "test_start": chronological_test["published_at"].min().isoformat(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path, compress=3)
    export_inference_model(final_model, HEADLINE_INFERENCE_PATH, chronological_selection["threshold"])
    metadata = {
        "model": "Logistic Regression headline model",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": chronological_selection["fit_rows"],
        "dataset_rows": len(data),
        "selected_parameters": {
            "model": chronological_selection["model_name"],
            "C": chronological_selection["c"],
            "class_weight": chronological_selection["class_weight"],
            "decision_threshold": chronological_selection["threshold"],
            "selection_metric": "chronological validation accuracy, then macro-F1",
        },
        "decision_threshold": chronological_selection["threshold"],
        "random_holdout_metrics": random_selection["test_metrics"],
        "random_selection": {
            "C": random_selection["c"],
            "class_weight": random_selection["class_weight"],
            "decision_threshold": random_selection["threshold"],
            "validation_metrics": random_selection["validation_metrics"],
        },
        "chronological_validation_metrics": chronological_selection[
            "validation_metrics"
        ],
        "chronological_holdout_metrics": chronological_selection["test_metrics"],
        "chronological_boundaries": temporal_boundaries,
        "labels": {str(label): name for label, name in LABELS.items()},
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "The headline model is trained on historical headlines from the project dataset.",
            "The threshold and model settings are selected on validation data; test data is not used for tuning.",
            "The chronological holdout is the more conservative estimate and does not prove performance on current news.",
            "Predictions are text-pattern classifications, not factual verification.",
            "Decision scores are not calibrated probabilities.",
        ],
    }
    metadata_path = model_path.with_name(f"{model_path.stem}.metadata.json")
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    print(
        f"Headline random holdout accuracy: "
        f"{random_selection['test_metrics']['accuracy']:.4f}"
    )
    print(
        f"Headline random holdout macro-F1: "
        f"{random_selection['test_metrics']['macro_f1']:.4f}"
    )
    print(
        f"Headline chronological holdout accuracy: "
        f"{chronological_selection['test_metrics']['accuracy']:.4f}"
    )
    print(
        f"Headline chronological holdout macro-F1: "
        f"{chronological_selection['test_metrics']['macro_f1']:.4f}"
    )
    print(
        f"Selected C={chronological_selection['c']}, "
        f"class_weight={chronological_selection['class_weight']}, "
        f"decision_threshold={chronological_selection['threshold']:.4f}"
    )
    print(f"Saved headline model to: {model_path}")
    return final_model


@lru_cache(maxsize=4)
def load_model(model_path=MODEL_PATH):
    if Path(model_path).resolve() == MODEL_PATH.resolve():
        return load_inference_model(MODEL_INFERENCE_PATH)
    if Path(model_path).suffixes[-2:] == [".json", ".gz"]:
        return load_inference_model(Path(model_path))
    import joblib

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run this script without --predict first."
        )
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_headline_model():
    return load_inference_model(HEADLINE_INFERENCE_PATH)


@lru_cache(maxsize=3)
def load_inference_model(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Inference model not found at {model_path}. Train and export the model first."
        )
    with gzip.open(model_path, "rt", encoding="utf-8") as model_file:
        model = json.load(model_file)
    for transformer in model["transformers"]:
        transformer["vocabulary"] = {
            term: index for index, term in enumerate(transformer.pop("terms"))
        }
    return model


def export_inference_model(pipeline, target_path, decision_threshold=0.0):
    import numpy as np

    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        classifier_kind = "linear"
        coefficient_rows = np.asarray(classifier.coef_, dtype=float)
        intercepts = np.asarray(classifier.intercept_, dtype=float).tolist()
    elif hasattr(classifier, "feature_log_prob_"):
        classifier_kind = "multinomial_nb"
        coefficient_rows = np.asarray(classifier.feature_log_prob_, dtype=float)
        intercepts = np.asarray(classifier.class_log_prior_, dtype=float).tolist()
    else:
        raise ValueError(f"Cannot export {type(classifier).__name__} for lightweight inference")

    transformers = []
    offset = 0
    for name, vectorizer in pipeline.named_steps["features"].transformer_list:
        terms = [None] * len(vectorizer.vocabulary_)
        for term, index in vectorizer.vocabulary_.items():
            terms[index] = term
        width = len(terms)
        transformers.append(
            {
                "name": name,
                "analyzer": vectorizer.analyzer,
                "ngram_range": list(vectorizer.ngram_range),
                "stop_words": sorted(vectorizer.get_stop_words() or ()),
                "terms": terms,
                "idf": np.asarray(vectorizer.idf_, dtype=float).tolist(),
                "coefficients": coefficient_rows[:, offset : offset + width].tolist(),
            }
        )
        offset += width

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target_path, "wt", encoding="utf-8", compresslevel=9) as model_file:
        json.dump(
            {
                "format_version": 1,
                "classifier": classifier_kind,
                "classes": np.asarray(classifier.classes_).tolist(),
                "intercepts": intercepts,
                "decision_threshold": float(decision_threshold),
                "transformers": transformers,
            },
            model_file,
            separators=(",", ":"),
        )


def _vectorize_transformer(text, transformer):
    vocabulary = transformer["vocabulary"]
    if transformer["analyzer"] == "word":
        stop_words = set(transformer["stop_words"])
        tokens = [
            token
            for token in re.findall(r"(?u)\b\w\w+\b", text.lower())
            if token not in stop_words
        ]
        min_n, max_n = transformer["ngram_range"]
        counts = Counter(tokens if min_n <= 1 else ())
        if min_n <= 2 <= max_n:
            counts.update(
                f"{left} {right}"
                for left, right in zip(tokens, tokens[1:])
            )
    else:
        counts = Counter()
        min_n, max_n = transformer["ngram_range"]
        for token in text.split():
            padded = f" {token} "
            for size in range(min_n, max_n + 1):
                counts.update(
                    padded[index : index + size]
                    for index in range(len(padded) - size + 1)
                )

    weighted = {}
    squared_norm = 0.0
    for term, count in counts.items():
        index = vocabulary.get(term)
        if index is None:
            continue
        value = (1.0 + math.log(count)) * transformer["idf"][index]
        weighted[index] = value
        squared_norm += value * value
    if squared_norm:
        norm = math.sqrt(squared_norm)
        for index in weighted:
            weighted[index] /= norm
    return weighted


def _portable_scores(model, text):
    if model["format_version"] != 1:
        raise ValueError(f"Unsupported inference model format: {model['format_version']}")
    if model["classifier"] == "linear":
        score = model["intercepts"][0]
        terms = []
        for transformer in model["transformers"]:
            weighted = _vectorize_transformer(text, transformer)
            coefficients = transformer["coefficients"][0]
            terms.extend(coefficients[index] * value for index, value in weighted.items())
        return math.fsum([score, *terms])

    class_scores = list(model["intercepts"])
    for transformer in model["transformers"]:
        weighted = _vectorize_transformer(text, transformer)
        for class_index, coefficients in enumerate(transformer["coefficients"]):
            class_scores[class_index] += math.fsum(
                coefficients[index] * value for index, value in weighted.items()
            )
    return class_scores


def load_model_metadata(model_path=MODEL_PATH):
    metadata_path = model_path.with_name(f"{model_path.stem}.metadata.json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def predict_news(text, model_path=MODEL_PATH):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Enter some news text to classify")

    is_default_model = Path(model_path).resolve() == MODEL_PATH.resolve()
    is_headline = is_default_model and len(cleaned) <= HEADLINE_MAX_LENGTH
    model = load_headline_model() if is_headline else load_model(model_path)
    decision_score = None
    decision_threshold = None
    probability_score = None
    if isinstance(model, dict):
        model_score = _portable_scores(model, cleaned)
        if model["classifier"] == "linear":
            decision_score = float(model_score)
        else:
            maximum = max(model_score)
            exponentials = [math.exp(score - maximum) for score in model_score]
            probabilities = [
                probability / sum(exponentials) for probability in exponentials
            ]
            class_index = probabilities.index(max(probabilities))
            prediction = int(model["classes"][class_index])
            probability_score = probabilities[class_index]
        if is_headline and decision_score is not None:
            decision_threshold = load_model_metadata(HEADLINE_MODEL_PATH).get(
                "decision_threshold", model["decision_threshold"]
            )
            prediction = int(decision_score >= decision_threshold)
        elif decision_score is not None:
            prediction = int(decision_score > 0)
        elif model["classifier"] != "multinomial_nb":
            raise ValueError(f"Unsupported inference classifier: {model['classifier']}")
    else:
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "decision_function"):
            decision_score = float(model.decision_function([cleaned])[0])
        if is_headline and decision_score is not None:
            decision_threshold = load_model_metadata(HEADLINE_MODEL_PATH).get(
                "decision_threshold", 0.0
            )
            prediction = int(decision_score >= decision_threshold)
        else:
            prediction = int(model.predict([cleaned])[0])

    result = {
        "label": LABELS[prediction],
        "model_name": "Logistic Regression headline model" if is_headline else "Linear SVM article model",
        "model_version": (
            HEADLINE_MODEL_PATH.stem if is_headline else Path(model_path).stem
        ),
        "input_mode": "headline" if is_headline else "article",
        "confidence": None,
        "processed_text": cleaned,
    }
    if decision_threshold is not None:
        result["decision_threshold"] = decision_threshold
    if is_headline and decision_score is not None:
        result["score"] = decision_score
        result["score_type"] = "decision score; not a probability"
    elif not isinstance(model, dict) and hasattr(classifier, "predict_proba"):
        result["score"] = float(model.predict_proba([cleaned])[0][prediction])
        result["score_type"] = "model probability; not a fact-check"
    elif probability_score is not None:
        result["score"] = probability_score
        result["score_type"] = "model probability; not a fact-check"
    elif decision_score is not None:
        result["score"] = decision_score
        result["score_type"] = "decision score; not a probability"
    return result


def main():
    parser = argparse.ArgumentParser(description="Train or use the fake-news text model")
    parser.add_argument("--predict", help="Classify a news article with the saved model")
    parser.add_argument(
        "--train-headline-model",
        action="store_true",
        help="Train the separate short-headline classifier",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    if args.predict:
        print(predict_news(args.predict, model_path=args.model_path))
    elif args.train_headline_model:
        train_headline_model(data_dir=args.data_dir)
    else:
        train_and_evaluate(data_dir=args.data_dir, model_path=args.model_path)


if __name__ == "__main__":
    main()