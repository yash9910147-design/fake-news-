import os

import gradio as gr

from fake_news_detector import (
    LABELS,
    MODEL_PATH,
    load_data_splits,
    load_model,
    load_model_metadata,
    predict_news,
)


if not MODEL_PATH.is_file():
    raise RuntimeError(
        "Trained model not found. Run `python fake_news_detector.py` before launching the app."
    )

load_model()
MODEL_METADATA = load_model_metadata()
TEST_METRICS = MODEL_METADATA.get("test_metrics", {})
TEMPORAL_BENCHMARK = MODEL_METADATA.get("temporal_benchmark", {})
TEMPORAL_METRICS = TEMPORAL_BENCHMARK.get("test_metrics") or {}


def classification_rows(metrics):
    report = metrics.get("classification_report", {})
    return [
        [
            label,
            report[label]["precision"],
            report[label]["recall"],
            report[label]["f1-score"],
            report[label]["support"],
        ]
        for label in ("Fake", "Real", "macro avg", "weighted avg")
        if label in report
    ]


def confusion_rows(metrics):
    matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    return [
        ["Fake", matrix[0][0], matrix[0][1]],
        ["Real", matrix[1][0], matrix[1][1]],
    ]


CLASS_REPORT_ROWS = classification_rows(TEST_METRICS)
TEMPORAL_REPORT_ROWS = classification_rows(TEMPORAL_METRICS)


def classify_article(article):
    try:
        prediction = predict_news(article)
    except ValueError as error:
        return f'<div class="result empty">{error}</div>'

    label = prediction["label"]
    result_class = "fake" if label == "Fake" else "real"
    score = prediction.get("score")
    score_line = ""
    if score is not None:
        score_line = (
            f'<p class="score">Decision score: {score:.3f} '
            "<span>(not a probability)</span></p>"
        )

    return (
        f'<div class="result {result_class}">'
        '<p class="eyebrow">MODEL CLASSIFICATION</p>'
        f'<h2>Prediction: {label} News</h2>'
        f"{score_line}"
        "<p class=\"caveat\">This model recognizes patterns in its training data; "
        "it does not verify claims or sources.</p>"
        "</div>"
    )


def update_dataset_view(split_name, label_filter, row_limit):
    frame = load_data_splits()[split_name]
    if label_filter != "All":
        expected_label = 0 if label_filter == "Fake" else 1
        frame = frame.loc[frame["label"] == expected_label]

    sample = frame.head(int(row_limit))
    table = []
    choices = []
    for row_id, row in sample.iterrows():
        label = LABELS[int(row["label"])]
        article = row["news"]
        excerpt = article[:220] + ("..." if len(article) > 220 else "")
        table.append([int(row_id), label, excerpt])
        choices.append((f"#{row_id} · {label} · {article[:85]}", int(row_id)))

    selected_id = choices[0][1] if choices else None
    dropdown = gr.update(choices=choices, value=selected_id)
    counts = frame["label"].value_counts().reindex([0, 1], fill_value=0)
    status = (
        f"{len(frame):,} {split_name.lower()} articles match this filter. "
        f"Fake: {counts[0]:,} · Real: {counts[1]:,}. "
        f"Showing {len(sample):,}."
    )
    return table, dropdown, status


def analyze_dataset_example(split_name, row_id):
    if row_id is None:
        return '<div class="result empty">Choose an article from the dataset first.</div>'

    record = load_data_splits()[split_name].loc[int(row_id)]
    known_label = LABELS[int(record["label"])]
    prediction = predict_news(record["news"])
    predicted_label = prediction["label"]
    result_class = "fake" if predicted_label == "Fake" else "real"
    comparison = "Matches dataset label" if predicted_label == known_label else "Differs from dataset label"
    score = prediction.get("score")
    score_line = f'<p class="score">Decision score: {score:.3f} <span>(not a probability)</span></p>' if score is not None else ""

    return (
        f'<div class="result {result_class}">'
        '<p class="eyebrow">HELD-OUT DATA CHECK</p>'
        f'<h2>Model predicts: {predicted_label}</h2>'
        f'<p>Dataset label: <strong>{known_label}</strong></p>'
        f'<p>{comparison}</p>'
        f'{score_line}'
        '</div>'
    )


def load_dataset_example(split_name, row_id):
    if row_id is None:
        return "", "Choose an article from the dataset first."
    record = load_data_splits()[split_name].loc[int(row_id)]
    return record["news"], f"Loaded row #{int(row_id)}. Switch to Analyze News to classify it."


CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto !important; }
#hero { padding: 30px 0 22px; border-bottom: 1px solid #dce5e2; }
#hero h1 { font-family: Georgia, serif; font-size: 38px; margin: 6px 0 8px; color: #162d2b; }
#hero p { max-width: 720px; color: #536764; font-size: 15px; }
.eyebrow { color: #bc542f; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
.result { min-height: 250px; padding: 25px; border: 1px solid #dce5e2; border-left: 5px solid #4a8a78; border-radius: 6px; background: #fff; }
.result.fake { border-left-color: #c25738; }
.result.real { border-left-color: #39836d; }
.result.empty { color: #536764; }
.result h2 { margin: 10px 0; color: #162d2b; font-family: Georgia, serif; font-size: 30px; }
.score { color: #304542; font-variant-numeric: tabular-nums; }
.score span, .caveat { color: #64736f; font-size: 13px; }
#footer-note { border-top: 1px solid #dce5e2; margin-top: 16px; padding-top: 14px; color: #64736f; font-size: 13px; }
"""


with gr.Blocks(
    title="Fake News Classifier | NLP",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="orange", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.Markdown(
        """
<div id="hero">
<p class="eyebrow">NLP · TF-IDF · MACHINE LEARNING</p>
<h1>Fake News Classifier</h1>
<p>Classify news articles as Fake or Real using NLP, TF-IDF, and machine learning.</p>
</div>
        """
    )
    with gr.Tabs():
        with gr.Tab("Analyze News"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=6):
                    article_input = gr.Textbox(
                        label="Article text",
                        placeholder="Paste a news headline and article text...",
                        lines=13,
                        max_lines=20,
                    )
                    with gr.Row():
                        analyze_button = gr.Button("Analyze article", variant="primary")
                        clear_button = gr.ClearButton(article_input, value="Clear")
                with gr.Column(scale=5):
                    result_output = gr.HTML(
                        '<div class="result empty">'
                        '<p class="eyebrow">READY FOR INPUT</p>'
                        "<p>Your model classification will appear here.</p></div>"
                    )

            analyze_button.click(
                fn=classify_article,
                inputs=article_input,
                outputs=result_output,
            )
            article_input.submit(
                fn=classify_article,
                inputs=article_input,
                outputs=result_output,
            )
            clear_button.click(lambda: "", outputs=result_output)
            gr.Examples(
                examples=[
                    [
                        "WASHINGTON (Reuters) - A Republican senator said the committee would review the new tax bill and discuss the budget next week. The administration is expected to release details after the hearing."
                    ],
                    [
                        "BREAKING: SHOCKING secret cure exposed as officials panic and media refuses to report the truth."
                    ],
                ],
                inputs=article_input,
                label="Try an example",
            )

        with gr.Tab("Training & Test Data"):
            gr.Markdown(
                "Browse the same stratified splits used by the model. Select a labeled row to compare its prediction with the dataset label."
            )
            with gr.Row():
                split_selector = gr.Dropdown(
                    choices=["Training", "Validation", "Test"],
                    value="Test",
                    label="Dataset split",
                )
                label_selector = gr.Dropdown(
                    choices=["All", "Fake", "Real"],
                    value="All",
                    label="Known label",
                )
                row_limit = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=10,
                    step=5,
                    label="Rows to display",
                )
            refresh_button = gr.Button("Refresh dataset", variant="secondary")
            dataset_status = gr.Markdown("Loading dataset preview...")
            dataset_table = gr.Dataframe(
                headers=["Row ID", "Known label", "Article preview"],
                datatype=["number", "str", "str"],
                interactive=False,
                label="Labeled article samples",
            )
            sample_selector = gr.Dropdown(label="Selected article", choices=[])
            with gr.Row():
                compare_button = gr.Button("Compare with model", variant="primary")
                load_button = gr.Button("Load into analyzer")
            dataset_result = gr.HTML(
                '<div class="result empty">Select a row and compare it with the model.</div>'
            )
            load_status = gr.Markdown()
            gr.Markdown(
                "The split is stratified and reproducible: 64% training, 16% validation, and 20% test. "
                "Use the test split for an independent evaluation; training examples were used to fit the model."
            )

            demo.load(
                fn=update_dataset_view,
                inputs=[split_selector, label_selector, row_limit],
                outputs=[dataset_table, sample_selector, dataset_status],
            )
            refresh_button.click(
                fn=update_dataset_view,
                inputs=[split_selector, label_selector, row_limit],
                outputs=[dataset_table, sample_selector, dataset_status],
            )
            compare_button.click(
                fn=analyze_dataset_example,
                inputs=[split_selector, sample_selector],
                outputs=dataset_result,
            )
            load_button.click(
                fn=load_dataset_example,
                inputs=[split_selector, sample_selector],
                outputs=[article_input, load_status],
            )

        with gr.Tab("Model Report"):
            if MODEL_METADATA:
                gr.Markdown(
                    "Compare a stratified random holdout with a chronological holdout. "
                    "The time-based result is a stronger check against distribution shift, "
                    "but it is not source-independent."
                )
                gr.Markdown("### Random stratified holdout")
                gr.Markdown(
                    f"**Selected model:** {MODEL_METADATA['model']}  \n"
                    f"**Selection rule:** highest validation macro-F1 "
                    f"({MODEL_METADATA['validation_macro_f1']:.4f})"
                )
                with gr.Row():
                    gr.Number(
                        value=TEST_METRICS["accuracy"],
                        label="Test accuracy",
                        precision=4,
                        interactive=False,
                    )
                    gr.Number(
                        value=TEST_METRICS["macro_f1"],
                        label="Test macro-F1",
                        precision=4,
                        interactive=False,
                    )
                    gr.Number(
                        value=TEST_METRICS["roc_auc"],
                        label="Test ROC-AUC",
                        precision=4,
                        interactive=False,
                    )
                gr.Dataframe(
                    value=CLASS_REPORT_ROWS,
                    headers=["Class", "Precision", "Recall", "F1", "Support"],
                    datatype=["str", "number", "number", "number", "number"],
                    interactive=False,
                    label="Held-out test classification report",
                )
                gr.Dataframe(
                    value=confusion_rows(TEST_METRICS),
                    headers=["Actual \\ Predicted", "Fake", "Real"],
                    datatype=["str", "number", "number"],
                    interactive=False,
                    label="Confusion matrix",
                )
                gr.Markdown("### Chronological holdout")
                if TEMPORAL_BENCHMARK.get("available"):
                    date_ranges = TEMPORAL_BENCHMARK["date_ranges"]
                    temporal_rows = TEMPORAL_BENCHMARK["split_class_counts"]
                    gr.Markdown(
                        f"**Selected model:** {TEMPORAL_BENCHMARK['model']}  \n"
                        f"**Validation macro-F1:** "
                        f"{TEMPORAL_BENCHMARK['validation_macro_f1']:.4f}  \n"
                        f"**Test period:** {date_ranges['Test']['start'][:10]} to "
                        f"{date_ranges['Test']['end'][:10]}  \n"
                        f"**Undated rows excluded:** "
                        f"{TEMPORAL_BENCHMARK['date_boundaries']['undated_rows']}"
                    )
                    with gr.Row():
                        gr.Number(
                            value=TEMPORAL_METRICS["accuracy"],
                            label="Chronological test accuracy",
                            precision=4,
                            interactive=False,
                        )
                        gr.Number(
                            value=TEMPORAL_METRICS["macro_f1"],
                            label="Chronological test macro-F1",
                            precision=4,
                            interactive=False,
                        )
                        gr.Number(
                            value=TEMPORAL_METRICS["roc_auc"],
                            label="Chronological test ROC-AUC",
                            precision=4,
                            interactive=False,
                        )
                    gr.Dataframe(
                        value=[
                            [
                                split,
                                counts.get("Fake", 0),
                                counts.get("Real", 0),
                            ]
                            for split, counts in temporal_rows.items()
                        ],
                        headers=["Time split", "Fake articles", "Real articles"],
                        datatype=["str", "number", "number"],
                        interactive=False,
                        label="Class balance by time split",
                    )
                    gr.Dataframe(
                        value=TEMPORAL_REPORT_ROWS,
                        headers=["Class", "Precision", "Recall", "F1", "Support"],
                        datatype=["str", "number", "number", "number", "number"],
                        interactive=False,
                        label="Chronological test classification report",
                    )
                    gr.Dataframe(
                        value=confusion_rows(TEMPORAL_METRICS),
                        headers=["Actual \\ Predicted", "Fake", "Real"],
                        datatype=["str", "number", "number"],
                        interactive=False,
                        label="Chronological confusion matrix",
                    )
                else:
                    gr.Markdown(
                        "Chronological benchmark unavailable: "
                        f"{TEMPORAL_BENCHMARK.get('error', 'no date data')}"
                    )
                with gr.Accordion("Reproducibility and limitations", open=False):
                    gr.JSON(
                        value={
                            "trained_at_utc": MODEL_METADATA["trained_at_utc"],
                            "dataset": MODEL_METADATA["dataset"],
                            "temporal_benchmark": TEMPORAL_BENCHMARK,
                            "features": MODEL_METADATA["features"],
                            "random_state": MODEL_METADATA["random_state"],
                            "versions": MODEL_METADATA["versions"],
                            "limitations": MODEL_METADATA["limitations"],
                        },
                        label="Run metadata",
                    )
            else:
                gr.Markdown(
                    "No saved evaluation report is available yet. Run "
                    "`python fake_news_detector.py` to train and evaluate the model."
                )

    gr.Markdown(
        "**Important:** This is an educational text classifier, not a fact-checking system. "
        "Its output is a learned pattern, not proof that an article is true or false.",
        elem_id="footer-note",
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("PORT", "7860")),
        inbrowser=True,
        share=os.environ.get("GRADIO_SHARE", "").lower() in {"1", "true", "yes"},
    )