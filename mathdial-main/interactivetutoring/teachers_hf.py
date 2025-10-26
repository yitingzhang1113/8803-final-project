from typing import Optional, List, Literal

from .history import History
from .roles import Roles
from .hf_backend import HFTextGenerator


TEACHER_BASE = """You are a helpful math tutor.
Follow the MathDial tutoring style and respond in English only.
Requirements:
- Start the Teacher message with exactly one pedagogical tag: (generic) or (focus) or (telling).
- Be concise; ask guiding questions before revealing results.
- Do not state the final numeric answer unless the student has derived it or it is strictly necessary to correct a misconception.
- Keep the reply to 1-3 short sentences.

Math problem: {problem}
Correct solution (for teacher reference):
{ground_truth}
{student_profile_block}{teacher_confusion_block}
The following is a conversation between a Teacher and a Student. Stay on topic and avoid revealing the final answer too early unless necessary.
"""


class HFFinetunedTeacher(object):
    def __init__(
        self,
        model_name: str = "eth-nlped/MathDial-SFT-Qwen2.5-1.5B-Instruct",
        hf_token: str = None,
        fallback_model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
    ):
        self.persona = Roles.TEACHER
        self.name = "HF Teacher"
        self.generator = HFTextGenerator(
            model=model_name,
            hf_token=hf_token,
            fallback_model=fallback_model,
            fallback_models=fallback_models,
        )

    def reset(self):
        pass

    def response(
        self,
        history: History,
        question: str,
        ground_truth_solution: str,
        student_profile: Optional[str] = None,
        teacher_confusion: Optional[str] = None,
        prompt_mode: Literal["guided", "dataset"] = "guided",
    ):
        # Build prompt according to selected mode
        if prompt_mode == "dataset":
            # No system preamble; use conversation-only prompt for exact MathDial style
            convo = []
            for m in history.messages:
                role_prefix = "Teacher" if m.persona == Roles.TEACHER else "Student"
                convo.append(f"{role_prefix}: {m.text}")
            convo_text = "\n".join(convo)
            prompt = f"{convo_text}\nTeacher: "
        else:
            # guided (default): include problem/ground truth and optional metadata
            student_profile_block = (
                f"\nStudent profile: {student_profile}\n" if student_profile else ""
            )
            teacher_confusion_block = (
                f"Teacher-annotated confusion: {teacher_confusion}\n" if teacher_confusion else ""
            )
            system_prompt = TEACHER_BASE.format(
                problem=question,
                ground_truth=ground_truth_solution or "(not provided)",
                student_profile_block=student_profile_block,
                teacher_confusion_block=teacher_confusion_block,
            )

            convo = []
            for m in history.messages:
                role_prefix = "Teacher" if m.persona == Roles.TEACHER else "Student"
                convo.append(f"{role_prefix}: {m.text}")
            convo_text = "\n".join(convo)

            prompt = (
                f"{system_prompt}\n\n"
                f"{convo_text}\n"
                f"Teacher: "
            )
        # Be conservative with stop sequences; stopping on "Teacher:" may prematurely truncate output on some backends.
        generated = self.generator.generate(
            prompt,
            stop=["\nStudent:"],
            max_new_tokens=256,
            temperature=0.2,
        )
        return generated.strip()
