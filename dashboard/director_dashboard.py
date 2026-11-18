"""Bulk CSV summaries and explicitly assumed 30-day demo windows.

Source outcomes remain observed labels. No forecast is calculated from them.
"""

from datetime import date, timedelta
from html import escape

import pandas as pd
import plotly.graph_objects as go

METRICS = {
    "All encounters": "encounters",
    "Patients (distinct IDs)": "patients",
    "Later (>30)": ">30",
    "No recorded return (NO)": "NO",
}
DEFAULT_METRICS = [
    "All encounters",
    "Patients (distinct IDs)",
    "No recorded return (NO)",
]
DEMOGRAPHIC_OPTIONS = [
    "Admission type",
    "Age band",
    "Race",
    "Diagnosis",
    "Lab tests",
    "Medication",
    "Prior visits",
    "Dosage changes",
]
ACTUAL_BLUE = "#2a78d6"
FORECAST_RED = "#eb6834"

_DEMOGRAPHIC_COLUMNS = {
    "Admission type": "demo_admission_type",
    "Age band": "demo_age_band",
    "Race": "demo_race",
    "Diagnosis": "demo_diagnosis",
    "Prior visits": "demo_prior_visits",
    "Dosage changes": "demo_dosage_changes",
}


def _bar_figure(frame, title, series_name="Sample", height=300, horizontal=False):
    labels = frame["measure"].astype(str).tolist()
    counts = frame["count"].astype(int).tolist()
    if horizontal:
        trace = go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            name=series_name,
            marker_color=ACTUAL_BLUE,
            hovertemplate="%{y}<br>Count: %{x:,}<extra></extra>",
        )
    else:
        trace = go.Bar(
            x=labels,
            y=counts,
            name=series_name,
            marker_color=ACTUAL_BLUE,
            hovertemplate="%{x}<br>Count: %{y:,}<extra></extra>",
        )
    figure = go.Figure(
        trace
    )
    figure.update_layout(
        title=title,
        template="plotly_dark",
        height=max(height, len(frame) * 34 + 90) if horizontal else height,
        margin={"l": 190 if horizontal else 45, "r": 20, "t": 55, "b": 45 if horizontal else 90},
        showlegend=False,
    )
    if horizontal:
        figure.update_yaxes(
            tickmode="array",
            tickvals=labels,
            ticktext=[
                f"{label}<br>(n = {count:,})" for label, count in zip(labels, counts)
            ],
            automargin=True,
            autorange="reversed",
        )
        figure.update_xaxes(rangemode="tozero", title_text="Count")
    else:
        figure.update_xaxes(
            tickmode="array",
            tickvals=labels,
            ticktext=[
                f"{label}<br>(n = {count:,})" for label, count in zip(labels, counts)
            ],
            automargin=True,
        )
        figure.update_yaxes(rangemode="tozero", title_text="Count")
    return figure


def _add_forecast_legend(figure):
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Forecasted · model pending",
            marker={"color": FORECAST_RED, "symbol": "square", "size": 10},
            showlegend=True,
            hoverinfo="skip",
        )
    )
    figure.update_layout(showlegend=True, barmode="group")
    return figure


def planning_windows(reference):
    anchor = date.fromisoformat(reference)
    if anchor.isoformat() != reference:
        raise ValueError("Use YYYY-MM-DD for the demo reference date.")
    return {
        "input": ((anchor - timedelta(days=29)).isoformat(), anchor.isoformat()),
        "outlook": (
            (anchor + timedelta(days=1)).isoformat(),
            (anchor + timedelta(days=30)).isoformat(),
        ),
    }


def summary_html(history):
    if history is None:
        return '<div class="source-summary">No CSV loaded.</div>'
    s = history.summary
    if s.get("processed"):
        patient_text = (
            f" · {s['patients']:,} unique patients"
            if s["patients"] is not None
            else " · Patient identifiers are not present"
        )
        return (
            '<div class="source-summary">'
            f"<strong>{escape(history.source)}</strong> · "
            f'{s["encounters"]:,} sample rows · '
            f'30-day readmission target: <strong>{s["<30"]:,}</strong> positive / '
            f'<strong>{s["NO"]:,}</strong> negative'
            f"{patient_text}"
            "</div>"
        )
    return (
        '<div class="source-summary">'
        f"<strong>{escape(history.source)}</strong> · "
        f'{s["encounters"]:,} encounters · {s["patients"]:,} patients · '
        f'Source labels: &lt;30 <strong>{s["<30"]:,}</strong> / '
        f'&gt;30 <strong>{s[">30"]:,}</strong> / NO <strong>{s["NO"]:,}</strong>'
        "</div>"
    )


def bulk_chart(history, metrics):
    if not set(metrics) <= set(METRICS):
        raise ValueError("Choose the listed bulk measures.")
    if history is None:
        return None, "Load a CSV to view the batch."
    if not metrics:
        return None, "Select at least one measure."
    counts = history.summary
    if counts.get("processed"):
        processed_keys = {
            "All encounters": "encounters",
            "No recorded return (NO)": "NO",
        }
        if counts["patients"] is not None:
            processed_keys["Patients (distinct IDs)"] = "patients"
        unsupported = [
            label
            for label in metrics
            if label not in processed_keys
        ]
        supported = [
            label for label in metrics if label in processed_keys
        ]
        if unsupported:
            status = (
                "The processed sample provides a binary <30-day target; "
                "a later >30 measure is unavailable."
            )
        else:
            status = (
                "The processed sample reports encounter, patient and binary "
                "30-day target counts."
            )
        if not supported:
            return None, status
        frame = pd.DataFrame(
            [
                {
                    "measure": label,
                    "count": counts[processed_keys[label]],
                }
                for label in METRICS
                if label in supported
            ]
        )
        return _bar_figure(frame, "Loaded batch · assumed previous 30 days"), status
    frame = pd.DataFrame(
        [
            {
                "measure": label,
                "count": counts[METRICS[label]],
            }
            for label in METRICS
            if label in metrics
        ]
    )
    return (
        _bar_figure(frame, "Loaded batch · assumed previous 30 days"),
        "Outcome bars count encounters; Patients counts distinct IDs. Overlapping measures must not be added together.",
    )


def early_readmission_chart(history):
    if history is None:
        return None, "Load a CSV to view recorded early readmissions."
    summary = history.summary
    count = summary["<30"]
    frame = pd.DataFrame(
        [{"measure": "Early readmissions (<30 days)", "count": count}]
    )
    figure = _bar_figure(frame, "Early readmissions · sample", "Sample")
    _add_forecast_legend(figure)
    note = (
        "Blue is the recorded 30-day target in the sample. The red model estimate "
        "is a placeholder; no forecast has been calculated."
    )
    return figure, note


def _early_rows(history):
    if history.kind == "processed":
        return history.rows[history.rows["readmitted_30d"] == "1"]
    return history.rows[history.rows["readmitted"] == "<30"]


def demographics_chart(history, demographic):
    if history is None:
        return None, "Load a CSV to view sample demographics."
    if demographic not in DEMOGRAPHIC_OPTIONS:
        raise ValueError("Choose a listed demographic measure.")
    early = _early_rows(history)
    if demographic == "Medication":
        columns = sorted(
            column
            for column in early.columns
            if column.startswith("demo_medication__")
        )
        if not columns:
            return None, "Medication detail is not present in this CSV."
        frame = pd.DataFrame(
            [
                {
                    "measure": column.removeprefix("demo_medication__"),
                    "count": int((early[column] == "1").sum()),
                }
                for column in columns
            ]
        )
        note = (
            "Counts are early-readmission encounters with each medication; "
            "medication groups overlap."
        )
    elif demographic == "Lab tests":
        columns = ["demo_max_glu_serum", "demo_A1Cresult"]
        if not set(columns) <= set(early.columns):
            return None, "Lab-test detail is not present in this CSV."
        values = []
        for column, display_name in (
            ("demo_max_glu_serum", "Max glucose"),
            ("demo_A1Cresult", "A1C"),
        ):
            counts = early[column].replace("", "Unknown").value_counts()
            values.extend(
                {
                    "measure": f"{display_name}: {label}",
                    "count": int(count),
                }
                for label, count in counts.items()
            )
        frame = pd.DataFrame(values)
        note = "Counts are early-readmission encounters by recorded lab-test result."
    else:
        column = _DEMOGRAPHIC_COLUMNS[demographic]
        if column not in early.columns:
            return None, f"{demographic} detail is not present in this CSV."
        values = early[column].replace("", "Unknown")
        if demographic in {"Prior visits", "Dosage changes"}:
            counts = pd.to_numeric(values).value_counts().sort_index()
        else:
            counts = values.value_counts().sort_index()
        frame = pd.DataFrame(
            [
                {"measure": str(label), "count": int(count)}
                for label, count in counts.items()
            ]
        )
        note = "Counts are sample encounters with an early readmission."
    if frame.empty:
        return None, "No early-readmission sample rows are available."
    figure = _bar_figure(
        frame,
        f"Early-readmission sample · {demographic}",
        "Early-readmission sample",
        height=380,
        horizontal=len(frame) > 8,
    )
    _add_forecast_legend(figure)
    return figure, f"{note} Forecast comparison will be added in red when estimates are available."


def forecast_chart(history):
    return None, (
        "Awaiting model and event-timing outputs for this cohort. Recorded outcome labels are not a next-30-days forecast."
        if history is not None
        else "Awaiting CSV and supported model outputs."
    )


def planning_rows(history, reference):
    windows = planning_windows(reference)
    previous = " → ".join(windows["input"])
    next_period = " → ".join(windows["outlook"])
    if history is None:
        count = "Awaiting CSV"
    elif history.summary.get("processed"):
        count = f'{history.summary["encounters"]:,} sample rows'
    else:
        count = f'{history.summary["encounters"]:,} encounters / {history.summary["patients"]:,} patients'
    return pd.DataFrame(
        [
            ["Input batch · assumed 30 days", previous, count],
            [
                "Next 30 days · same cohort",
                next_period,
                "Forecast unavailable — model/timing pending",
            ],
        ],
        columns=["Planning proposal", "Inclusive window", "Volume / readiness"],
    )


def planning_table_html(history, reference):
    table = planning_rows(history, reference)
    header = "".join(f"<th>{escape(column)}</th>" for column in table.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in table.itertuples(index=False, name=None)
    )
    return f'<div class="planning-table"><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'
