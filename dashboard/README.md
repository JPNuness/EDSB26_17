# EDSB26 - Group 17

## Current prototype: compact Gradio Director dashboard

Updated 2026-11-07. This scope supersedes the earlier three-view follow-up demo.

- Director: read the supplied CSV, inspect interactive bulk-count bars and a
  two-row planning proposal. The next-30-days forecast panel awaits supported
  model/timing outputs.
- Clinician dashboard and operational worklist: empty until their scope/model
  inputs are agreed.
- Follow-up, questionnaires, bookings and weighted capacity are deferred. Their
  prior design remains in DEMO_SPEC.md as a possible later addition.
- Explicit demo assumption: the entire uploaded batch represents the previous
  30 days. The source CSV has no dates. No calendar history, first-ever intake,
  exact elapsed-day counts or forecast is inferred from IDs or row order.

## Run locally in VS Code

Keep the team's existing Python and `.venv` unchanged. Gradio 5.50.0 is installed
in the existing `EDSB26_17_dashboard/.venv-gradio` environment. Select that
interpreter for UI work or run this dashboard with its absolute interpreter path:

~~~powershell
Set-Location "C:\Workstation\EDSA\Data BootCamp\EDSB26_17\dashboard"
& "C:\Workstation\EDSA\Data BootCamp\EDSB26_17_dashboard\.venv-gradio\Scripts\python.exe" .\app.py
~~~

Open [the local dashboard](http://127.0.0.1:7860/). Stop with Ctrl+C.
Use --port 7862 if the port is occupied. An older tab at port 8765 is a separate
old process; stop its terminal and open the new address. demo_app.py now forwards
to this same Gradio app and must also use the Gradio interpreter.

For live editing, launch with the Gradio CLI instead of `python app.py`:

~~~powershell
Set-Location "C:\Workstation\EDSA\Data BootCamp\EDSB26_17\dashboard"
& "C:\Workstation\EDSA\Data BootCamp\EDSB26_17_dashboard\.venv-gradio\Scripts\python.exe" -m gradio app.py --demo-name demo
~~~

Leave the server running, edit and save a Python file, then refresh the browser
after Gradio detects the change. Restart only when changing dependencies or
when a reload fails.

Plotly 6.5.0 is installed in the separate dashboard environment and listed in
requirements-gradio.txt. The project's shared `.venv` and `requirements.txt`
are unchanged. The downloaded Space copies in hf-space/ and bootcamp/ were not
changed or published. Their README metadata currently names a different Gradio
version; align the selected Space runtime with the local app before a later
deployment.

## Load and inspect CSV data

The app starts with no CSV loaded. Expand **Load CSV / scenario date**, then:

- Click **Load test_sample.csv** to load
  `C:\Workstation\EDSA\Data BootCamp\EDSB26_17\data\processed\test_sample.csv`,
  or
- Upload another CSV in the original 50-column format or processed model-ready
  format.

Each valid import replaces the current browser session's batch; repeated imports
do not append or double the totals. Invalid imports retain and label the last
valid batch. The generated sample includes encounter and patient IDs plus
demographic fields used by the dashboard.

The reader validates headers, row lengths, whole-number fields, encounter
identity and source outcome labels. It preserves cells as strings, including
literal None, ?, age bands and raw diagnosis strings. Existing source/audit
scripts remain unchanged. Changing a CSV or the demo reference date does not
write to the source file or generate individual event dates.

The left chart shows batch volume and does not include early readmissions. Its
tick labels show `(n = ...)` on a separate line below each measure. The right
chart shows the sample's known early readmissions in blue; the red forecast
series is a placeholder until a model is connected. The estimate button
currently reports that no predictions have been made.

Below the charts, select Admission type, Age band, Race, Diagnosis, Lab tests,
Medication, Prior visits or Dosage changes to inspect the sample encounters
labelled as early readmissions. Category counts also show `(n = ...)` beneath
each label. The red forecast comparison remains empty until model estimates are
available. Some uploaded CSVs may not have the extra demographic fields; the
chart reports when a selected breakdown is unavailable.

The demo reference A defaults to the Lisbon calendar date. The assumed input
window is A−29 through A, and the outlook is A+1 through A+30, inclusive.
These dates label the scenario only; they are not verified source coverage.
No future patient inflow, beds, staff-hours, treatment effect or model
probability is inferred from historical outcome labels.

## Verification and next step

~~~powershell
.\.venv-gradio\Scripts\python.exe -m unittest -v test_director_dashboard test_demo
~~~

The current checks cover CSV preservation, replacement/rejection, actual bulk
counts, bar selections, scenario dates, compact layout and empty tabs. test_demo
retains the deferred follow-up rules' regression checks; those features are not
in the current UI. [Revision evidence](DIRECTOR_REVISION.md) distinguishes the
current checks from the [earlier prototype verification](STEP8_VERIFICATION.md).

[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md), [PROJECT_TASKS.md](PROJECT_TASKS.md),
[DEMO_SPEC.md](DEMO_SPEC.md) and [MODEL_HANDOVER.md](MODEL_HANDOVER.md) record
decisions and handover requirements. DEMO_GUIDE.md is an archived walkthrough of
the earlier follow-up prototype; it is not the current run guide.

Next: review the bulk bar layout together. Genuine forecasts need documented
model/probability and event-timing support for the same input cohort; expected
returns do not directly supply beds, staff-hours or evidence of prevention.
