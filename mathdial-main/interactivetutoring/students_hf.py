from typing import Optional, List
from .history import History
from .roles import Roles
from .hf_backend import HFTextGenerator


STUDENT_NAME = "Kayla"
STUDENT_PERSONA = STUDENT_NAME + " is a 7th grade student. She has problem with understanding of what steps or procedures are required to solve a problem."

STUDENT_PROMPT = f"""
Student Persona: {STUDENT_PERSONA}

Context: {STUDENT_NAME} thinks their answer is correct. Only when the teacher provides several good reasoning questions, {STUDENT_NAME} understands the problem and corrects the solution. {STUDENT_NAME} can use calculator and thus makes no calculation errors.
Respond in English only, as the Student.
Keep replies concise (1-3 short sentences) and avoid adding the words "Teacher:" or "Student:".
"""


class HFStudent(object):
    def __init__(
        self,
        model_name: str = "eth-nlped/MathDial-SFT-Qwen2.5-1.5B-Instruct",
        hf_token: Optional[str] = None,
        fallback_model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
    ):
        self.persona = Roles.STUDENT
        self.name = STUDENT_NAME
        self.generator = HFTextGenerator(
            model=model_name,
            hf_token=hf_token,
            fallback_model=fallback_model,
            fallback_models=fallback_models,
        )

    def reset(self):
        pass

    def response(self, history: History, question: str, incorrect_solution: str):
        # Turn the dialog history into role-prefixed lines
        convo = []
        for m in history.messages:
            role_prefix = "Teacher" if m.persona == Roles.TEACHER else "Student"
            convo.append(f"{role_prefix}: {m.text}")
        convo_text = "\n".join(convo)

        prompt = (
            f"{STUDENT_PROMPT}\n\n"
            f"Math problem: {question}\n\n"
            f"Student incorrect solution: {incorrect_solution}\n\n"
            f"Dialog so far:\n{convo_text}\n"
            f"Student: "
        )
        generated = self.generator.generate(
            prompt,
            stop=["\nTeacher:", "\nStudent:"],
            max_new_tokens=200,
            temperature=0.5,
        )
        utterance = generated.replace("Student:", "").replace(self.name + ":", "").replace("<EOM>", "").strip()
        return utterance
