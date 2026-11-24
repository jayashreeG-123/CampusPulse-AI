# CampusPulse AI

A Flask student-support analytics demonstration using a synthetic-data ML screening model, an optional local/cloud LLM, SQLite persistence, teacher intervention tracking, historical progress comparisons, and human-reviewed support recommendations.

## Features
- CSV upload with column, numeric-range, blank-value and duplicate-ID validation.
- SQLite-backed datasets and student snapshots, scoped to the signed-in account.
- Student progress comparison between the latest and previous uploads.
- Experimental risk indicators, class summary and AI explanations.
- Support recommendations saved for teacher/admin review; approval or rejection is recorded.
- Existing intervention records and user accounts are preserved during additive database initialization.
- CSV and PDF reports.
- Admin and teacher roles, password hashing, session cookie settings, CSRF token validation, 10 MB upload limit and safe uploaded filenames.

## Setup (Windows)
1. Open this folder in VS Code.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\\Scripts\\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env`. Set `SECRET_KEY` to a long random value. Keep `.env` private and never upload it.
5. Ensure Ollama is running and the model is installed:
   ```powershell
   ollama pull qwen2.5:3b
   ```
6. Start the application:
   ```powershell
   python app.py
   ```
7. Open http://127.0.0.1:5000. On first launch, create the administrator account.

## CSV format
Required columns:
`student_id,name,attendance,internal_marks,assignment_completion,previous_result`

- attendance: 0–100
- internal_marks: 0–30
- assignment_completion: 0–100
- previous_result: 0–100
- Student IDs must be unique within each uploaded CSV.

## New pages
- `/progress` — compares each student's latest saved snapshot with their previous upload.
- `/recommendations` — lets a signed-in teacher/admin review, approve or reject generated support suggestions.

Progress comparisons are based on successive uploads for the same student ID under the same account. They do not establish that an intervention caused a change.

## Data and privacy
- SQLite database: `data/interventions.db`.
- Uploaded CSV copies: `data/uploads/`.
- Dataset snapshots and recommendation review states are saved in SQLite.
- Each account sees its own uploaded datasets and student progress. Intervention visibility is restricted to admins and the assigned/creating teacher.
- Back up `data/interventions.db` and `data/uploads/` together before moving or updating the project.
- This is a learning/demo project. Do not expose it publicly or use identifiable student data without institutional authorization, privacy review, access controls and a retention policy.

## Model limitations
The included estimators are trained on generated synthetic examples. Scores, categories, anomaly flags and clusters are experimental indicators, not validated probabilities, diagnoses or conclusions about a student. LLM output may be wrong. Faculty must review all context and approve any support. Do not use this app for admissions, grading, discipline, eligibility or automated decisions.
