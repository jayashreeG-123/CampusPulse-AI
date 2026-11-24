
def investigate_student(row):
    reasons = []

    if float(row["attendance"]) < 75:
        reasons.append(
            f"Attendance is {row['attendance']}%, below the 75% review threshold."
        )

    if float(row["internal_marks"]) < 18:
        reasons.append(
            f"Internal marks are {row['internal_marks']}/30; subject support may help."
        )

    if float(row["assignment_completion"]) < 70:
        reasons.append(
            f"Assignment completion is {row['assignment_completion']}%."
        )

    if float(row["previous_result"]) < 50:
        reasons.append(
            f"Previous result is {row['previous_result']}%."
        )

    return reasons or [
        "No major concern was detected by the current screening rules."
    ]


def create_action_plan(row, reasons):
    tasks = []

    if float(row["attendance"]) < 75:
        tasks.append(
            "A mentor can discuss attendance barriers and agree on a realistic attendance goal."
        )

    if float(row["internal_marks"]) < 18:
        tasks.append(
            "Offer subject-specific doubt-clearing or additional academic support."
        )

    if float(row["assignment_completion"]) < 70:
        tasks.append(
            "Prepare a pending-assignment checklist with achievable submission dates."
        )

    if float(row["previous_result"]) < 50:
        tasks.append(
            "Review subject-wise performance and identify topics for revision."
        )

    if not tasks:
        tasks.append(
            "Continue regular mentoring and review progress at the next check-in."
        )

    tasks.append("Review progress with the student after 14 days.")

    return tasks


def class_summary(results):
    total = len(results)

    counts = {
        level: sum(r["level"] == level for r in results)
        for level in ("High", "Medium", "Low")
    }

    return {
        "total": total,
        "counts": counts,
        "average_score": round(
            sum(r["score"] for r in results) / total, 1
        ) if total else 0,
        "anomalies": sum(bool(r["anomaly"]) for r in results)
    }


def find_student(results, query):
    q = str(query).strip().lower()

    matches = []

    for student in results:
        name = student["name"].lower()
        student_id = student["student_id"].lower()

        if name in q or student_id in q:
            matches.append(student)

    return matches


def filter_students_by_risk(results, level):
    """Return actual student records belonging to a risk category."""
    level = level.capitalize()

    return [
        student for student in results
        if student["level"] == level
    ]


def compare_risk_groups(results):
    """Calculate actual statistics for all three risk groups."""

    comparison = {}

    for level in ("High", "Medium", "Low"):

        students = filter_students_by_risk(results, level)

        total = len(students)

        comparison[level] = {
            "total": total,
            "average_screening_score": round(
                sum(s["score"] for s in students) / total, 1
            ) if total else 0,
            "average_attendance": round(
                sum(s["attendance"] for s in students) / total, 1
            ) if total else 0,
            "average_internal_marks": round(
                sum(s["internal_marks"] for s in students) / total, 1
            ) if total else 0,
            "average_assignment_completion": round(
                sum(s["assignment_completion"] for s in students) / total, 1
            ) if total else 0
        }

    return comparison
