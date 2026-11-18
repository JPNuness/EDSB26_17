"""Compact Gradio 5 application. Use the separate .venv-gradio interpreter."""

import argparse
from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import gradio as gr

from director_dashboard import (
    DEFAULT_METRICS,
    DEMOGRAPHIC_OPTIONS,
    METRICS,
    bulk_chart,
    demographics_chart,
    early_readmission_chart,
    planning_rows,
    planning_table_html,
    summary_html,
)
from history_data import replace_history

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_HISTORY = PROJECT_ROOT / "data" / "processed" / "test_sample.csv"
CSS = """
.gradio-container {max-width: 1240px !important; margin: auto;}
#app-heading {margin-bottom: 0;}
#app-heading h1 {font-size: 1.45rem; margin: 0;}
.source-summary {font-size: .9rem; padding: 5px 0;}
.chart-note {font-size: .86rem; min-height: 32px;}
.compact-note p {margin: 2px 0; font-size: .88rem;}
#scenario-note {border-left: 3px solid #c88c15; padding-left: 10px;}
.planning-table {overflow-x: auto;}
.planning-table table {width: 100%; border-collapse: collapse; font-size: .88rem;}
.planning-table td, .planning-table th {text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border-color-primary);}
"""


def refresh_history(
    path, current, metrics, reference, demographic=DEMOGRAPHIC_OPTIONS[0]
):
    """Session-only replacement. Failed imports cannot erase a valid snapshot."""
    # Validate the scenario first, so an invalid date cannot partially replace data.
    planning_rows(current, reference)
    history, message = replace_history(path, current)
    chart, chart_status = bulk_chart(history, metrics)
    early_chart, early_status = early_readmission_chart(history)
    demographic_plot, demographic_status = demographics_chart(history, demographic)
    return (
        history,
        summary_html(history),
        message,
        chart,
        chart_status,
        early_chart,
        early_status,
        planning_table_html(history, reference),
        demographic_plot,
        demographic_status,
    )


def change_reference(history, reference):
    try:
        return planning_table_html(history, reference), ""
    except (ValueError, TypeError):
        return gr.skip(), "Enter a valid demo reference date in YYYY-MM-DD format."


def request_forecast(history):
    if history is None:
        return "Load a CSV first. No forecast has been calculated."
    return (
        "Forecasting is not connected yet. No patient-level predictions have "
        "been calculated."
    )


def create_app(default_history=DEFAULT_HISTORY):
    reference_date = datetime.now(ZoneInfo("Europe/Lisbon")).date().isoformat()
    with gr.Blocks(
        title="Diabetes readmission · Director",
        css=CSS,
        analytics_enabled=False,
        theme=gr.themes.Soft(
            primary_hue="teal", font=["Segoe UI", "Arial", "sans-serif"]
        ),
        fill_width=True,
    ) as demo:
        gr.Markdown("# Diabetes hospital readmission", elem_id="app-heading")
        history = gr.State(None)
        with gr.Tabs():
            with gr.Tab("Director dashboard"):
                gr.Markdown(
                    "**DEMO TIME ASSUMPTION:** the entire loaded batch represents the previous 30 days. The CSV has no event dates.",
                    elem_id="scenario-note",
                    elem_classes="compact-note",
                )
                summary = gr.HTML(summary_html(None))
                with gr.Accordion("Load CSV / scenario date", open=False):
                    with gr.Row():
                        upload = gr.File(
                            label="Replacement CSV · same 50 columns",
                            file_types=[".csv"],
                            type="filepath",
                            height=80,
                        )
                        with gr.Column(min_width=210, scale=0):
                            reference = gr.Textbox(
                                value=reference_date,
                                label="Demo reference date A · YYYY-MM-DD",
                            )
                            reload = gr.Button("Load test_sample.csv", size="sm")
                    import_status = gr.Markdown("", elem_classes="compact-note")
                    reference_status = gr.Markdown("", elem_classes="compact-note")
                    gr.Markdown(
                        "Imports replace this session's batch. The source file is read only; "
                        "the demo date changes window labels only. The processed test sample "
                        "uses its readmitted_30d target.",
                        elem_classes="compact-note",
                    )
                metrics = gr.CheckboxGroup(
                    list(METRICS), value=DEFAULT_METRICS, label="Batch measures"
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=400):
                        gr.Markdown("### Loaded batch · assumed previous 30 days")
                        activity = gr.Plot()
                        activity_status = gr.Markdown(
                            "Awaiting CSV.", elem_classes="chart-note"
                        )
                    with gr.Column(scale=1, min_width=320):
                        gr.Markdown("### Early readmissions · sample and forecast")
                        early_plot = gr.Plot()
                        early_status = gr.Markdown(
                            "Awaiting CSV.", elem_classes="chart-note"
                        )
                        estimate = gr.Button(
                            "Estimate patients to be readmitted in the next 30 days",
                            variant="primary",
                        )
                        estimate_status = gr.Markdown("", elem_classes="compact-note")
                gr.Markdown("### Early-readmission demographics · sample vs forecast")
                demographic = gr.Dropdown(
                    choices=DEMOGRAPHIC_OPTIONS,
                    value=DEMOGRAPHIC_OPTIONS[0],
                    label="Break down by",
                )
                demographic_plot = gr.Plot()
                demographic_status = gr.Markdown(
                    "Load the demo sample to view its early-readmission demographics.",
                    elem_classes="chart-note",
                )
                planning = gr.HTML(planning_table_html(None, reference_date))
                with gr.Accordion("Definitions", open=False):
                    gr.Markdown(
                        "**Early** means a recorded return **<30 days** from discharge. "
                        "The raw '<30', '>30' and 'NO' labels are preserved. "
                        "Exact elapsed days, first-ever intake and calendar trends cannot be recovered from this CSV. "
                        "The same-cohort outlook will need one reconciled discharge anchor per patient and a supported timing method. "
                        "No future patient inflow, beds, staff-hours, treatment effect or model probability is inferred here."
                    )
            with gr.Tab("Clinician dashboard"):
                pass
            with gr.Tab("Operational worklist"):
                pass
        outputs = [
            history,
            summary,
            import_status,
            activity,
            activity_status,
            early_plot,
            early_status,
            planning,
            demographic_plot,
            demographic_status,
        ]
        metrics.change(
            bulk_chart,
            inputs=[history, metrics],
            outputs=[activity, activity_status],
            api_name=False,
        )
        reference.change(
            change_reference,
            inputs=[history, reference],
            outputs=[planning, reference_status],
            api_name=False,
        )
        demographic.change(
            demographics_chart,
            inputs=[history, demographic],
            outputs=[demographic_plot, demographic_status],
            api_name=False,
        )
        estimate.click(
            request_forecast,
            inputs=history,
            outputs=estimate_status,
            api_name=False,
        )
        upload.upload(
            refresh_history,
            inputs=[upload, history, metrics, reference, demographic],
            outputs=outputs,
            api_name=False,
        )
        upload.clear(
            lambda: "Upload cleared; the current loaded batch is retained.",
            outputs=import_status,
            api_name=False,
        )
        reload.click(
            lambda current, selected, ref, dimension: refresh_history(
                default_history, current, selected, ref, dimension
            ),
            inputs=[history, metrics, reference, demographic],
            outputs=outputs,
            api_name=False,
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()
    create_app().launch(
        server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        show_api=False,
        ssr_mode=False,
        max_file_size="64mb",
    )


if __name__ == "__main__":
    main()
