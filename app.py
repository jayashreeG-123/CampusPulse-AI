
from pathlib import Path
import uuid
import secrets
from werkzeug.utils import secure_filename
from flask import g, abort
import sqlite3
import os
import webbrowser
import csv
import json
from io import StringIO, BytesIO
from datetime import date, datetime
from functools import wraps

import pandas as pd

from werkzeug.security import generate_password_hash, check_password_hash

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    session,
    send_file
)

from campuspulse.config import Config
from campuspulse.agent import CampusPulseAgent
from campuspulse.llm import LLMClient


# --------------------------------------------------
# APPLICATION SETUP
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "interventions.db"

DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config.from_object(Config)
app.config.update(
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
    PERMANENT_SESSION_LIFETIME=__import__("datetime").timedelta(hours=8),
)

llm_client = LLMClient(Config)
agent = CampusPulseAgent(llm_client)

REQUIRED_COLUMNS = [
    "student_id",
    "name",
    "attendance",
    "internal_marks",
    "assignment_completion",
    "previous_result"
]

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

def init_intervention_db():

    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            assigned_to TEXT NOT NULL,
            support_type TEXT NOT NULL,
            description TEXT NOT NULL,
            due_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending',
            created_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS intervention_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intervention_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            followup_note TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (intervention_id)
                REFERENCES interventions(id)
        )
    """)

    conn.commit()
    conn.close()


def init_user_db():

    conn = get_db_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
                CHECK(role IN ('admin', 'teacher')),
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_intervention_db()
init_user_db()


def init_persistence_db():
    """Additive schema upgrades; existing intervention/user data is preserved."""
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT,
            created_at TEXT NOT NULL,
            is_sample INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(owner_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS student_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            owner_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            attendance REAL NOT NULL,
            internal_marks REAL NOT NULL,
            assignment_completion REAL NOT NULL,
            previous_result REAL NOT NULL,
            screening_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            result_json TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES datasets(id),
            FOREIGN KEY(owner_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_owner_student
        ON student_snapshots(owner_id, student_id, captured_at)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_active_dataset (
            user_id INTEGER PRIMARY KEY,
            dataset_id INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS support_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id INTEGER NOT NULL,
            dataset_id INTEGER NOT NULL,
            student_id TEXT NOT NULL,
            student_name TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending'
                CHECK(status IN ('Pending','Approved','Rejected')),
            reviewed_by TEXT,
            created_at TEXT NOT NULL,
            reviewed_at TEXT
        )
    """)
    # Safe migration for existing intervention records.
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(interventions)")}
    if "created_by_id" not in columns:
        conn.execute("ALTER TABLE interventions ADD COLUMN created_by_id INTEGER")
    for metric in ("attendance", "internal_marks", "assignment_completion", "screening_score"):
        if f"baseline_{metric}" not in columns:
            conn.execute(f"ALTER TABLE interventions ADD COLUMN baseline_{metric} REAL")
        if f"outcome_{metric}" not in columns:
            conn.execute(f"ALTER TABLE interventions ADD COLUMN outcome_{metric} REAL")
    if "outcome_captured_at" not in columns:
        conn.execute("ALTER TABLE interventions ADD COLUMN outcome_captured_at TEXT")
    conn.commit()
    conn.close()


init_persistence_db()


def current_student_results():
    return getattr(g, "current_results", [])


def persist_analysis(df, results, original_filename, stored_filename=None, is_sample=False):
    owner_id = session["user_id"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cur = conn.execute(
        """INSERT INTO datasets(owner_id, original_filename, stored_filename, created_at, is_sample)
           VALUES (?, ?, ?, ?, ?)""",
        (owner_id, original_filename, stored_filename, now, int(is_sample))
    )
    dataset_id = cur.lastrowid
    for student in results:
        conn.execute(
            """INSERT INTO student_snapshots(
                dataset_id, owner_id, student_id, student_name, attendance,
                internal_marks, assignment_completion, previous_result,
                screening_score, risk_level, result_json, captured_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                dataset_id, owner_id, str(student["student_id"]), str(student["name"]),
                float(student["attendance"]), float(student["internal_marks"]),
                float(student["assignment_completion"]), float(student["previous_result"]),
                float(student["score"]), student["level"],
                json.dumps(student), now
            )
        )
        conn.execute(
            """INSERT INTO support_recommendations(
                owner_id, dataset_id, student_id, student_name,
                recommendation, status, created_at
            ) VALUES (?, ?, ?, ?, ?, 'Pending', ?)""",
            (owner_id, dataset_id, str(student["student_id"]), str(student["name"]),
             " ".join(student.get("plan", [])), now)
        )
    conn.execute(
        """INSERT INTO user_active_dataset(user_id, dataset_id) VALUES (?, ?)
           ON CONFLICT(user_id) DO UPDATE SET dataset_id=excluded.dataset_id""",
        (owner_id, dataset_id)
    )
    conn.commit()
    conn.close()
    return dataset_id


def load_active_analysis(owner_id):
    conn = get_db_connection()
    row = conn.execute(
        """SELECT d.id, d.original_filename, s.result_json
           FROM user_active_dataset a
           JOIN datasets d ON d.id=a.dataset_id
           LEFT JOIN student_snapshots s ON s.dataset_id=d.id
           WHERE a.user_id=? ORDER BY s.id""", (owner_id,)
    ).fetchall()
    conn.close()
    if not row:
        return [], None, None
    results = [json.loads(r["result_json"]) for r in row if r["result_json"]]
    return results, row[0]["id"], row[0]["original_filename"]


@app.before_request
def protect_and_load_user_data():
    if request.method == "POST":
        token = session.get("_csrf_token")
        submitted = request.form.get("_csrf_token") or request.headers.get("X-CSRFToken")
        if not token or not submitted or not secrets.compare_digest(token, submitted):
            abort(400, description="Security token missing or invalid. Refresh the page and try again.")
    if session.get("user_id"):
        g.current_results, g.active_dataset_id, g.active_filename = load_active_analysis(session["user_id"])
    else:
        g.current_results, g.active_dataset_id, g.active_filename = [], None, None


@app.context_processor
def inject_security_context():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_urlsafe(32)
    return {"csrf_token": session["_csrf_token"]}




# --------------------------------------------------
# AUTHENTICATION HELPERS
# --------------------------------------------------

def user_count():

    conn = get_db_connection()

    count = conn.execute(
        "SELECT COUNT(*) AS total FROM users"
    ).fetchone()["total"]

    conn.close()

    return count


def login_required(view):

    @wraps(view)
    def wrapped_view(*args, **kwargs):

        if not session.get("user_id"):

            flash(
                "Please log in to continue.",
                "error"
            )

            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped_view


def roles_required(*allowed_roles):

    def decorator(view):

        @wraps(view)
        def wrapped_view(*args, **kwargs):

            if not session.get("user_id"):

                flash(
                    "Please log in to continue.",
                    "error"
                )

                return redirect(url_for("login"))

            if session.get("role") not in allowed_roles:

                flash(
                    "You do not have permission to access that page.",
                    "error"
                )

                return redirect(url_for("portal"))

            return view(*args, **kwargs)

        return wrapped_view

    return decorator


# --------------------------------------------------
# CSV FUNCTIONS
# --------------------------------------------------

def load_sample():

    return pd.read_csv(
        DATA_DIR / "sample_students.csv", dtype={"student_id": str, "name": str}
    )


def validate_dataframe(df):

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    if df.empty:

        raise ValueError(
            "The CSV contains no student rows."
        )

    for col in REQUIRED_COLUMNS[2:]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if df[col].isna().any():

            raise ValueError(
                f"Column {col} must contain numeric values in every row."
            )

    limits = {
        "attendance": (0, 100),
        "internal_marks": (0, 30),
        "assignment_completion": (0, 100),
        "previous_result": (0, 100)
    }

    for col, (low, high) in limits.items():

        if not df[col].between(low, high).all():

            raise ValueError(
                f"{col} values must be between {low} and {high}."
            )

    if df["student_id"].isna().any() or df["name"].isna().any():
        raise ValueError("Student ID and name are required for every row.")
    df["student_id"] = df["student_id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    if (df["student_id"] == "").any() or (df["name"] == "").any():
        raise ValueError("Student ID and name cannot be blank.")
    if df["student_id"].duplicated().any():
        duplicates = df.loc[df["student_id"].duplicated(), "student_id"].tolist()
        raise ValueError("Duplicate student IDs found: " + ", ".join(duplicates[:5]))
    return df


def analyse(df, uploaded, original_filename="sample_students.csv", stored_filename=None, is_sample=False):
    clean_df = validate_dataframe(df.copy())
    results = agent.analyse_class(clean_df)
    dataset_id = persist_analysis(clean_df, results, original_filename, stored_filename, is_sample)
    g.current_results = results
    g.active_dataset_id = dataset_id
    return render_template(
        "index.html", results=results, count=len(results), uploaded=uploaded,
        trace=agent.last_trace, llm_status=llm_client.status()
    )


# FIRST-TIME ADMIN SETUP
# --------------------------------------------------

@app.route("/setup", methods=["GET", "POST"])
def setup():

    if user_count() > 0:

        return redirect(
            url_for("login")
        )

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if len(username) < 3:

            flash(
                "Username must contain at least 3 characters.",
                "error"
            )

        elif len(password) < 8:

            flash(
                "Password must contain at least 8 characters.",
                "error"
            )

        elif password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

        else:

            conn = get_db_connection()

            try:

                conn.execute("""
                    INSERT INTO users
                    (username, password_hash, role, created_at)
                    VALUES (?, ?, 'admin', ?)
                """, (
                    username,
                    generate_password_hash(password),
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ))

                conn.commit()

                flash(
                    "Administrator account created. Please log in.",
                    "success"
                )

                return redirect(
                    url_for("login")
                )

            except sqlite3.IntegrityError:

                flash(
                    "That username is already in use.",
                    "error"
                )

            finally:

                conn.close()

    return render_template(
        "setup.html"
    )


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if user_count() == 0:

        return redirect(
            url_for("setup")
        )

    if session.get("user_id"):

        return redirect(
            url_for("portal")
        )

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        conn = get_db_connection()

        user = conn.execute("""
            SELECT *
            FROM users
            WHERE username = ?
        """, (username,)).fetchone()

        conn.close()

        if user and check_password_hash(
            user["password_hash"],
            password
        ):

            session.clear()

            session.permanent = True
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            session["chat_key"] = str(uuid.uuid4())

            flash(
                f"Welcome, {user['username']}!",
                "success"
            )

            return redirect(
                url_for("portal")
            )

        flash(
            "Invalid username or password.",
            "error"
        )

    return render_template(
        "login.html"
    )


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout", methods=["POST"])
@login_required
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# --------------------------------------------------
# ADMIN USER MANAGEMENT
# --------------------------------------------------

@app.route("/users", methods=["GET", "POST"])
@roles_required("admin")
def users():

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        role = request.form.get(
            "role", "teacher"
        ).strip().lower()

        if len(username) < 3:

            flash(
                "Username must contain at least 3 characters.",
                "error"
            )

        elif len(password) < 8:

            flash(
                "Password must contain at least 8 characters.",
                "error"
            )

        elif role not in ("admin", "teacher"):

            flash(
                "Choose a valid role.",
                "error"
            )

        else:

            conn = get_db_connection()

            try:

                conn.execute("""
                    INSERT INTO users
                    (username, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    username,
                    generate_password_hash(password),
                    role,
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ))

                conn.commit()

                flash(
                    f"{role.title()} account created successfully.",
                    "success"
                )

            except sqlite3.IntegrityError:

                flash(
                    "That username already exists.",
                    "error"
                )

            finally:

                conn.close()

    conn = get_db_connection()

    users_list = conn.execute("""
        SELECT id, username, role, created_at
        FROM users
        ORDER BY id
    """).fetchall()

    conn.close()

    return render_template(
        "users.html",
        users=users_list
    )


# --------------------------------------------------
# CAMPUSPULSE PORTAL
# --------------------------------------------------

@app.route("/")
def home():

    return redirect(
        url_for("portal")
    )


@app.route("/portal")
@login_required
def portal():

    return render_template(
        "portal.html"
    )


# --------------------------------------------------
# STUDENT ANALYTICS DASHBOARD
# --------------------------------------------------

@app.route("/dashboard", methods=["GET", "POST"])
@roles_required("admin", "teacher")
def index():
    if request.method == "POST":
        uploaded = request.files.get("student_file")
        if not uploaded or not uploaded.filename:
            flash("Please choose a CSV file.", "error")
            return redirect(url_for("index"))
        safe_name = secure_filename(uploaded.filename)
        if not safe_name.lower().endswith(".csv"):
            flash("Only CSV files are allowed.", "error")
            return redirect(url_for("index"))
        stored_name = f"{session['user_id']}_{uuid.uuid4().hex}.csv"
        stored_path = UPLOAD_DIR / stored_name
        try:
            uploaded.save(stored_path)
            if stored_path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("The CSV file exceeds the 10 MB upload limit.")
            frame = pd.read_csv(stored_path, dtype={"student_id": str, "name": str})
            return analyse(frame, True, safe_name, stored_name, False)
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            flash(f"Could not process file: {exc}", "error")
            return redirect(url_for("index"))

    if current_student_results():
        return render_template(
            "index.html", results=current_student_results(),
            count=len(current_student_results()), uploaded=False,
            trace=["Restored your most recently saved dataset."],
            llm_status=llm_client.status()
        )
    try:
        return analyse(load_sample(), False, "sample_students.csv", None, True)
    except Exception as exc:
        return f"Application setup error: {exc}", 500


# STUDENT REPORT DOWNLOADS
# --------------------------------------------------

def get_report_rows():

    if not current_student_results():

        return []

    rows = []

    for student in current_student_results():

        model_scores = student.get("model_scores") or {}
        reasons = student.get("reasons") or []
        plan = student.get("plan") or []

        rows.append({

            "Student ID": student.get("student_id", ""),
            "Student Name": student.get("name", ""),
            "Attendance (%)": student.get("attendance", ""),
            "Internal Marks (/30)": student.get("internal_marks", ""),
            "Assignment Completion (%)": student.get("assignment_completion", ""),
            "Previous Result (%)": student.get("previous_result", ""),
            "Experimental Screening Score (/100)": student.get("score", ""),
            "Risk Indicator": student.get("level", ""),
            "Random Forest Score (%)": model_scores.get("Random Forest", ""),
            "Gradient Boosting Score (%)": model_scores.get("Gradient Boosting", ""),
            "Pattern Group": student.get("segment", ""),
            "Unusual Pattern Flag": (
                "Review" if student.get("anomaly")
                else "Not flagged"
            ),
            "Observed Factors": "; ".join(
                str(item) for item in reasons
            ),
            "Suggested Support Plan": "; ".join(
                str(item) for item in plan
            )

        })

    return rows


# --------------------------------------------------
# CSV REPORT DOWNLOAD
# --------------------------------------------------

@app.get("/reports/students.csv")
@roles_required("admin", "teacher")
def download_students_csv():

    rows = get_report_rows()

    if not rows:

        flash(
            "Please analyse a student CSV before downloading a report.",
            "error"
        )

        return redirect(
            url_for("index")
        )

    output = StringIO()

    writer = csv.DictWriter(
        output,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)

    csv_bytes = BytesIO(
        output.getvalue().encode("utf-8-sig")
    )

    csv_bytes.seek(0)

    filename = (
        f"campuspulse_student_report_"
        f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    return send_file(
        csv_bytes,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


# --------------------------------------------------
# PDF REPORT DOWNLOAD
# --------------------------------------------------

@app.get("/reports/students.pdf")
@roles_required("admin", "teacher")
def download_students_pdf():

    rows = get_report_rows()

    if not rows:

        flash(
            "Please analyse a student CSV before downloading a report.",
            "error"
        )

        return redirect(
            url_for("index")
        )

    try:

        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle
        )
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )
        from xml.sax.saxutils import escape

    except ImportError:

        return (
            "PDF export needs ReportLab. "
            "Run: pip install reportlab",
            500
        )

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CampusPulseTitle",
        parent=styles["Title"],
        alignment=TA_LEFT,
        fontSize=19,
        leading=23,
        textColor=colors.HexColor("#17324d"),
        spaceAfter=5
    )

    small_style = ParagraphStyle(
        "CampusPulseSmall",
        parent=styles["BodyText"],
        fontSize=7,
        leading=9,
        alignment=TA_LEFT
    )

    cell_style = ParagraphStyle(
        "CampusPulseCell",
        parent=styles["BodyText"],
        fontSize=6.5,
        leading=8,
        alignment=TA_LEFT,
        wordWrap="CJK"
    )

    header_style = ParagraphStyle(
        "CampusPulseHeader",
        parent=cell_style,
        textColor=colors.white,
        fontName="Helvetica-Bold"
    )

    story = [

        Paragraph(
            "CampusPulse AI — Student Analytics Report",
            title_style
        ),

        Paragraph(
            f"Generated: {datetime.now():%d %B %Y, %I:%M %p} "
            f"&nbsp;&nbsp;|&nbsp;&nbsp; Students analysed: {len(rows)}",
            styles["BodyText"]
        ),

        Spacer(1, 7 * mm)

    ]

    columns = [

        "Student ID",
        "Student Name",
        "Attendance (%)",
        "Internal Marks (/30)",
        "Assignment Completion (%)",
        "Previous Result (%)",
        "Experimental Screening Score (/100)",
        "Risk Indicator"

    ]

    table_data = [[

        Paragraph(
            escape(column),
            header_style
        )

        for column in columns

    ]]

    for row in rows:

        table_data.append([

            Paragraph(
                escape(str(row.get(column, ""))),
                cell_style
            )

            for column in columns

        ])

    available_width = (
        landscape(A4)[0] - 24 * mm
    )

    widths = [

        available_width * 0.09,
        available_width * 0.14,
        available_width * 0.10,
        available_width * 0.11,
        available_width * 0.13,
        available_width * 0.10,
        available_width * 0.16,
        available_width * 0.17

    ]

    table = Table(
        table_data,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(TableStyle([

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#17324d")
        ),

        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.white
        ),

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.35,
            colors.HexColor("#d7e0e8")
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "TOP"
        ),

        (
            "LEFTPADDING",
            (0, 0),
            (-1, -1),
            4
        ),

        (
            "RIGHTPADDING",
            (0, 0),
            (-1, -1),
            4
        ),

        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            5
        ),

        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            5
        ),

        (
            "ROWBACKGROUNDS",
            (0, 1),
            (-1, -1),
            [
                colors.white,
                colors.HexColor("#f3f7fa")
            ]
        )

    ]))

    story.append(table)

    story.append(
        Spacer(1, 6 * mm)
    )

    story.append(
        Paragraph(
            "<b>Important note:</b> The included models are trained on "
            "synthetic demonstration data, not validated institutional data. "
            "Screening scores and categories are experimental indicators, "
            "not diagnoses or decisions. Faculty must verify the context and "
            "approve any support action. Do not use this report for "
            "disciplinary action, admissions, grading, or automated interventions.",
            small_style
        )
    )

    document.build(story)

    pdf_buffer.seek(0)

    filename = (
        f"campuspulse_student_report_"
        f"{datetime.now():%Y%m%d_%H%M%S}.pdf"
    )

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )


# --------------------------------------------------
# AI STATUS
# --------------------------------------------------

@app.get("/api/status")
@roles_required("admin", "teacher")
def api_status():

    return jsonify(
        llm_status=llm_client.status(),
        students=len(current_student_results())
    )


# --------------------------------------------------
# AI CHAT
# --------------------------------------------------

@app.post("/api/chat")
@roles_required("admin", "teacher")
def api_chat():

    payload = request.get_json(
        silent=True
    ) or {}

    question = str(
        payload.get("message", "")
    ).strip()

    if not question:

        return jsonify(
            error="Please enter a message."
        ), 400

    if not current_student_results():

        return jsonify(
            error="Analyse a student CSV before using the assistant."
        ), 400

    if "chat_key" not in session:

        session["chat_key"] = str(uuid.uuid4())

    answer, provider = agent.chat(
        question,
        current_student_results(),
        session["chat_key"]
    )

    return jsonify(
        answer=answer,
        provider=provider,
        trace=agent.last_trace[-2:]
    )


# --------------------------------------------------
# TEACHER INTERVENTION DASHBOARD
# --------------------------------------------------

@app.route("/interventions")
@roles_required("admin", "teacher")
def interventions():

    conn = get_db_connection()

    if session.get("role") == "admin":
        records = conn.execute("""
            SELECT * FROM interventions
            ORDER BY CASE status WHEN 'Pending' THEN 1 WHEN 'In Progress' THEN 2
            WHEN 'Completed' THEN 3 ELSE 4 END, due_date ASC
        """).fetchall()
    else:
        records = conn.execute("""
            SELECT * FROM interventions
            WHERE assigned_to = ? OR created_by_id = ?
            ORDER BY CASE status WHEN 'Pending' THEN 1 WHEN 'In Progress' THEN 2
            WHEN 'Completed' THEN 3 ELSE 4 END, due_date ASC
        """, (session.get("username"), session.get("user_id"))).fetchall()

    if session.get("role") == "admin":
        updates = conn.execute("""
            SELECT intervention_updates.*, interventions.student_name
            FROM intervention_updates
            JOIN interventions ON intervention_updates.intervention_id = interventions.id
            ORDER BY intervention_updates.updated_at DESC
        """).fetchall()
    else:
        updates = conn.execute("""
            SELECT intervention_updates.*, interventions.student_name
            FROM intervention_updates
            JOIN interventions ON intervention_updates.intervention_id = interventions.id
            WHERE interventions.assigned_to = ? OR interventions.created_by_id = ?
            ORDER BY intervention_updates.updated_at DESC
        """, (session.get("username"), session.get("user_id"))).fetchall()

    conn.close()

    pending_count = sum(
        1 for item in records
        if item["status"] == "Pending"
    )

    progress_count = sum(
        1 for item in records
        if item["status"] == "In Progress"
    )

    completed_count = sum(
        1 for item in records
        if item["status"] == "Completed"
    )

    return render_template(
        "interventions.html",
        interventions=records,
        updates=updates,
        students=current_student_results(),
        pending_count=pending_count,
        progress_count=progress_count,
        completed_count=completed_count,
        now_date=date.today().isoformat()
    )


# --------------------------------------------------
# CREATE NEW INTERVENTION
# --------------------------------------------------

@app.post("/interventions/create")
@roles_required("admin", "teacher")
def create_intervention():

    student_id = request.form.get(
        "student_id", ""
    ).strip()

    assigned_to = request.form.get(
        "assigned_to", ""
    ).strip()

    support_type = request.form.get(
        "support_type", ""
    ).strip()

    description = request.form.get(
        "description", ""
    ).strip()

    due_date = request.form.get(
        "due_date", ""
    ).strip()

    if session.get("role") == "teacher":
        assigned_to = session.get("username", "")
    if not all([
        student_id,
        assigned_to,
        support_type,
        description,
        due_date
    ]):

        flash(
            "Please complete all intervention fields.",
            "error"
        )

        return redirect(
            url_for("interventions")
        )

    student = next(
        (
            item for item in current_student_results()
            if str(item.get("student_id")) == student_id
        ),
        None
    )

    if student is None:

        flash(
            "Student not found. Upload and analyse the student CSV first.",
            "error"
        )

        return redirect(
            url_for("interventions")
        )

    try:

        datetime.strptime(
            due_date,
            "%Y-%m-%d"
        )

    except ValueError:

        flash(
            "Please enter a valid due date.",
            "error"
        )

        return redirect(
            url_for("interventions")
        )

    conn = get_db_connection()

    conn.execute("""
        INSERT INTO interventions (
            student_id,
            student_name,
            assigned_to,
            support_type,
            description,
            due_date,
            status,
            created_at,
            created_by_id,
            baseline_attendance,
            baseline_internal_marks,
            baseline_assignment_completion,
            baseline_screening_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        student_id,
        student.get("name", ""),
        assigned_to,
        support_type,
        description,
        due_date,
        "Pending",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        session["user_id"],
        student.get("attendance"),
        student.get("internal_marks"),
        student.get("assignment_completion"),
        student.get("score")
    ))

    conn.commit()
    conn.close()

    flash(
        f"Intervention assigned to {student.get('name', 'student')} successfully.",
        "success"
    )

    return redirect(
        url_for("interventions")
    )


# --------------------------------------------------
# UPDATE INTERVENTION AND ADD FOLLOW-UP
# --------------------------------------------------

@app.post("/interventions/<int:intervention_id>/update")
@roles_required("admin", "teacher")
def update_intervention(intervention_id):

    status = request.form.get(
        "status", ""
    ).strip()

    followup_note = request.form.get(
        "followup_note", ""
    ).strip()

    allowed_statuses = [
        "Pending",
        "In Progress",
        "Completed"
    ]

    if status not in allowed_statuses:

        flash(
            "Invalid intervention status.",
            "error"
        )

        return redirect(
            url_for("interventions")
        )

    if not followup_note:

        flash(
            "Please enter a follow-up remark.",
            "error"
        )

        return redirect(
            url_for("interventions")
        )

    conn = get_db_connection()

    intervention = conn.execute("""
        SELECT id, assigned_to, created_by_id
        FROM interventions
        WHERE id = ?
    """, (intervention_id,)).fetchone()

    if intervention is not None and session.get("role") != "admin":
        if intervention["assigned_to"] != session.get("username") and intervention["created_by_id"] != session.get("user_id"):
            conn.close()
            abort(403)

    if intervention is None:

        conn.close()

        flash(
            "Intervention record not found.",
            "error"
        )

        return redirect(
            url_for("interventions")
        )

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn.execute("""
        UPDATE interventions
        SET status = ?
        WHERE id = ?
    """, (
        status,
        intervention_id
    ))

    conn.execute("""
        INSERT INTO intervention_updates (
            intervention_id,
            status,
            followup_note,
            updated_at
        )
        VALUES (?, ?, ?, ?)
    """, (
        intervention_id,
        status,
        followup_note,
        now
    ))

    if status == "Completed":
        # Capture the latest analysed values when the teacher marks the plan complete.
        target_row = conn.execute(
            "SELECT student_id FROM interventions WHERE id=?", (intervention_id,)
        ).fetchone()
        outcome_student = next(
            (item for item in current_student_results()
             if target_row and str(item.get("student_id")) == str(target_row["student_id"])),
            None
        )
        if outcome_student:
            conn.execute("""
                UPDATE interventions SET
                    outcome_attendance=?, outcome_internal_marks=?,
                    outcome_assignment_completion=?, outcome_screening_score=?,
                    outcome_captured_at=?
                WHERE id=?
            """, (
                outcome_student.get("attendance"), outcome_student.get("internal_marks"),
                outcome_student.get("assignment_completion"), outcome_student.get("score"),
                now, intervention_id
            ))

    conn.commit()
    conn.close()

    flash(
        "Intervention progress updated successfully.",
        "success"
    )

    return redirect(
        url_for("interventions")
    )


# --------------------------------------------------


@app.get("/progress")
@roles_required("admin", "teacher")
def progress():
    conn = get_db_connection()
    rows = conn.execute(
        """SELECT student_id, student_name, attendance, internal_marks,
                  assignment_completion, previous_result, screening_score,
                  risk_level, captured_at
           FROM student_snapshots WHERE owner_id=?
           ORDER BY student_id, captured_at DESC, id DESC""",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    grouped = {}
    for row in rows:
        grouped.setdefault(row["student_id"], []).append(dict(row))
    comparisons = []
    for sid, history in grouped.items():
        latest = history[0]
        previous = history[1] if len(history) > 1 else None
        comparisons.append({
            "student_id": sid, "name": latest["student_name"],
            "latest": latest, "previous": previous,
            "attendance_change": round(latest["attendance"] - previous["attendance"], 1) if previous else None,
            "marks_change": round(latest["internal_marks"] - previous["internal_marks"], 1) if previous else None,
            "assignment_change": round(latest["assignment_completion"] - previous["assignment_completion"], 1) if previous else None,
            "history_count": len(history)
        })
    conn = get_db_connection()
    if session.get("role") == "admin":
        intervention_comparisons = conn.execute("""
            SELECT student_name, student_id, support_type, status,
                   baseline_attendance, outcome_attendance,
                   baseline_internal_marks, outcome_internal_marks,
                   baseline_assignment_completion, outcome_assignment_completion,
                   baseline_screening_score, outcome_screening_score,
                   outcome_captured_at
            FROM interventions WHERE outcome_captured_at IS NOT NULL
            ORDER BY outcome_captured_at DESC
        """).fetchall()
    else:
        intervention_comparisons = conn.execute("""
            SELECT student_name, student_id, support_type, status,
                   baseline_attendance, outcome_attendance,
                   baseline_internal_marks, outcome_internal_marks,
                   baseline_assignment_completion, outcome_assignment_completion,
                   baseline_screening_score, outcome_screening_score,
                   outcome_captured_at
            FROM interventions
            WHERE outcome_captured_at IS NOT NULL
              AND (assigned_to=? OR created_by_id=?)
            ORDER BY outcome_captured_at DESC
        """, (session.get("username"), session.get("user_id"))).fetchall()
    conn.close()
    return render_template("progress.html", comparisons=comparisons,
                           intervention_comparisons=intervention_comparisons)


@app.get("/recommendations")
@roles_required("admin", "teacher")
def recommendations():
    conn = get_db_connection()
    rows = conn.execute(
        """SELECT * FROM support_recommendations
           WHERE owner_id=? ORDER BY created_at DESC""",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return render_template("recommendations.html", recommendations=rows)


@app.post("/recommendations/<int:recommendation_id>/review")
@roles_required("admin", "teacher")
def review_recommendation(recommendation_id):
    decision = request.form.get("decision", "")
    if decision not in ("Approved", "Rejected"):
        abort(400)
    conn = get_db_connection()
    row = conn.execute(
        "SELECT id FROM support_recommendations WHERE id=? AND owner_id=?",
        (recommendation_id, session["user_id"])
    ).fetchone()
    if row is None:
        conn.close()
        abort(404)
    conn.execute(
        """UPDATE support_recommendations
           SET status=?, reviewed_by=?, reviewed_at=?
           WHERE id=? AND owner_id=?""",
        (decision, session.get("username"),
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         recommendation_id, session["user_id"])
    )
    conn.commit()
    conn.close()
    flash("Recommendation " + decision.lower() + ".", "success")
    return redirect(url_for("recommendations"))


# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":

        webbrowser.open_new(
            "http://127.0.0.1:5000/"
        )

    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1"
    )
