
import json

from .model import CampusModelSuite
from .tools import (
    investigate_student,
    create_action_plan,
    class_summary,
    find_student,
    filter_students_by_risk,
    compare_risk_groups
)
from .memory import ConversationMemory


SYSTEM_PROMPT = """
You are CampusPulse AI, an assistant for supportive academic review.

Use only the student records and calculated statistics supplied to you.

Important rules:
1. Never invent student names, IDs, scores or statistics.
2. Use the actual screening category supplied by the application.
3. Explain risk indicators using attendance, internal marks,
   assignment completion and previous result.
4. Do not claim that a screening indicator proves a student has a problem.
5. Treat model scores as experimental screening indicators, not diagnoses.
6. Recommend supportive human review, never punishment.
7. Do not invent demographic information or personal circumstances.
8. If information is missing, clearly say so.
9. Faculty members must review and approve support actions.
10. Keep responses clear and practical.
"""


class CampusPulseAgent:

    def __init__(self, llm_client=None):
        self.models = CampusModelSuite()
        self.llm = llm_client
        self.memory = ConversationMemory()
        self.last_trace = []

    def analyse_class(self, df):

        self.last_trace = [
            "1. Intake agent: received and validated the student dataset."
        ]

        results = []

        for _, row in df.iterrows():

            prediction = self.models.predict(row)

            reasons = investigate_student(row)

            plan = create_action_plan(row, reasons)

            self.last_trace.append(
                f"2. Prediction tools: evaluated {row['student_id']}."
            )

            self.last_trace.append(
                f"3. Investigation tool: checked student indicators for {row['student_id']}."
            )

            self.last_trace.append(
                f"4. Pattern tools: anomaly detection and clustering evaluated {row['student_id']}."
            )

            results.append({

                "student_id": str(row["student_id"]),
                "name": str(row["name"]),

                # Actual student input data
                "attendance": float(row["attendance"]),
                "internal_marks": float(row["internal_marks"]),
                "assignment_completion": float(row["assignment_completion"]),
                "previous_result": float(row["previous_result"]),

                # Model prediction
                "score": prediction["score"],
                "level": prediction["level"],
                "model_scores": prediction["model_scores"],
                "anomaly": prediction["anomaly"],
                "segment": prediction["segment"],

                # Investigation and support
                "reasons": reasons,
                "plan": plan
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        self.last_trace.append(
            "5. Planning agent: prepared suggested support actions; faculty approval is required."
        )

        return results

    def chat(self, question, results, session_key="demo"):

        q = question.strip()
        lower_q = q.lower()

        if not q:
            return "Please enter a question.", "tools"

        # 1. Identify category-comparison requests
        comparison_words = (
            "compare",
            "comparison",
            "difference between",
            "versus",
            " vs "
        )

        if any(word in lower_q for word in comparison_words):

            data = compare_risk_groups(results)

            self.last_trace.append(
                "Coordinator agent: selected risk-group comparison tool."
            )

            return self._format_comparison(data), "tools"

        # 2. Identify risk-category questions
        risk_levels = {
            "high": "High",
            "medium": "Medium",
            "low": "Low"
        }

        requested_level = None

        for keyword, level in risk_levels.items():

            if keyword in lower_q and any(
                word in lower_q
                for word in (
                    "risk",
                    "student",
                    "students",
                    "category",
                    "category",
                    "identify",
                    "list",
                    "show",
                    "who",
                    "which"
                )
            ):
                requested_level = level
                break

        if requested_level:

            students = filter_students_by_risk(
                results,
                requested_level
            )

            self.last_trace.append(
                f"Coordinator agent: selected {requested_level}-risk filtering tool."
            )

            return self._format_student_list(
                students,
                requested_level
            ), "tools"

        # 3. Identify class-summary questions
        if any(
            word in lower_q
            for word in (
                "summary",
                "summarize",
                "class overview",
                "overall",
                "how many",
                "class performance"
            )
        ):

            data = class_summary(results)

            self.last_trace.append(
                "Coordinator agent: selected class summary tool."
            )

            return self._fallback_answer(
                q, results, data, "class_summary"
            ), "tools"

        # 4. Identify individual student questions
        matches = find_student(results, q)

        if matches:

            tool_name = "find_student"
            tool_data = matches[:5]

            self.last_trace.append(
                "Coordinator agent: selected student investigation tool."
            )

        else:

            return (
                "I couldn't identify the student from your question. "
                "Please include the student's name or ID. "
                "You can also ask me to list High, Medium or Low-risk students.",
                "tools"
            )

        # 5. Send actual student data to Ollama
        if not self.llm:

            return self._fallback_answer(
                q, results, tool_data, tool_name
            ), "tools"

        messages = [
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + "\nAuthoritative student records:\n"
                    + json.dumps(tool_data, indent=2)
                )
            }
        ]

        messages.extend(
            self.memory.get(session_key)
        )

        messages.append({
            "role": "user",
            "content": q
        })

        try:

            answer, provider = self.llm.generate(messages)

            self.memory.add(session_key, "user", q)
            self.memory.add(session_key, "assistant", answer)

            self.last_trace.append(
                f"LLM response agent: generated an explanation using {provider}."
            )

            return answer, provider

        except Exception as exc:

            self.last_trace.append(
                "LLM unavailable: used deterministic tool-based response instead."
            )

            return (
                self._fallback_answer(
                    q, results, tool_data, tool_name
                )
                + f"\n\n(LLM unavailable: {exc})"
            ), "tools"

    @staticmethod
    def _format_student_list(students, level):

        if not students:
            return f"There are no students in the {level}-risk category."

        lines = [
            f"{level}-Risk Students: {len(students)} student(s)\n"
        ]

        for student in students:

            lines.append(
                f"• {student['name']} "
                f"(ID: {student['student_id']})\n"
                f"  Screening score: {student['score']}/100\n"
                f"  Attendance: {student['attendance']}%\n"
                f"  Internal marks: {student['internal_marks']}/30\n"
                f"  Assignment completion: {student['assignment_completion']}%\n"
                f"  Previous result: {student['previous_result']}%"
            )

        lines.append(
            "\nThese are experimental screening indicators. "
            "Teachers should review each student's circumstances "
            "before deciding on supportive action."
        )

        return "\n\n".join(lines)

    @staticmethod
    def _format_comparison(data):

        lines = [
            "Actual comparison of student risk categories:\n"
        ]

        for level in ("High", "Medium", "Low"):

            group = data[level]

            lines.append(
                f"{level} Risk:\n"
                f"• Students: {group['total']}\n"
                f"• Average screening score: {group['average_screening_score']}/100\n"
                f"• Average attendance: {group['average_attendance']}%\n"
                f"• Average internal marks: {group['average_internal_marks']}/30\n"
                f"• Average assignment completion: "
                f"{group['average_assignment_completion']}%"
            )

        lines.append(
            "\nAll figures are calculated from the analysed student records. "
            "Risk categories are experimental screening indicators, "
            "not diagnoses."
        )

        return "\n\n".join(lines)

    @staticmethod
    def _fallback_answer(question, results, data, tool_name):

        if tool_name == "find_student" and data:

            lines = []

            for student in data:

                lines.append(
                    f"{student['name']} ({student['student_id']}): "
                    f"{student['level']} screening indicator, "
                    f"{student['score']}/100.\n"
                    f"Attendance: {student['attendance']}%.\n"
                    f"Internal marks: {student['internal_marks']}/30.\n"
                    f"Assignment completion: "
                    f"{student['assignment_completion']}%.\n"
                    f"Previous result: {student['previous_result']}%.\n"
                    f"Factors: {' '.join(student['reasons'])}\n"
                    f"Suggested support: {' '.join(student['plan'])}"
                )

            return "\n\n".join(lines)

        summary = data

        return (
            f"Class overview: {summary['total']} students analysed.\n"
            f"High risk: {summary['counts']['High']}\n"
            f"Medium risk: {summary['counts']['Medium']}\n"
            f"Low risk: {summary['counts']['Low']}\n"
            f"Average screening score: {summary['average_score']}/100.\n"
            f"Unusual patterns flagged for review: {summary['anomalies']}.\n\n"
            "These are experimental indicators based on synthetic training data. "
            "Faculty should verify context and decide whether supportive "
            "follow-up is appropriate."
        )
