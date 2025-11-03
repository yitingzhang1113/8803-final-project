"""
Dependency Extractor for MathDial Conversations
Extracts student steps and builds cross-turn dependency graphs.
"""

import json
import re
import os
from typing import List, Dict, Optional, Tuple


# Operation dictionary (compiled as regex patterns)
# Expanded to include natural language patterns
OPS = {
    "substitute": r"\b(substitut|plug\s+in|let\s+\w+\s*=|replace\s+\w+\s+with|substituted)\b",
    "simplify": r"\b(simplif|reduce|combine\s+(like\s+)?terms|collect\s+terms|cancel)\b",
    "expand": r"\b(expand|distribute)\b",
    "factor": r"\b(factor|take\s+out|common\s+factor)\b",
    "isolate": r"\b(isolate|move\s+.*?\s+to\s+the\s+other\s+side|bring\s+.*?\s+over)\b",
    "differentiate": r"\b(differentiate|derivative)\b",
    "integrate": r"\b(integrate|antiderivative|integral)\b",
    "solve": r"\b(solve\s+for|solve|solved)\b",
    "check": r"\b(check|verify|compare)\b",
    "calculate": r"\b(calculate|calculated|compute|computed|figure\s+out|figured\s+out)\b",
    "divide": r"\b(divided|divide|dividing)\b",
    "multiply": r"\b(multiplied|multiply|multiplying)\b",
    "add": r"\b(added|add|adding)\b",
    "subtract": r"\b(subtracted|subtract|subtracting)\b",
}

# Explicit reference pattern (step/turn/line numbers)
EXPLICIT_REF = re.compile(r"\b(step|turn|line)\s*(\d+)\b", re.I)

# Pattern for previous/substitution references
PREVIOUS_REF = re.compile(r"\b(previous|after\s+substitut|based\s+on)\b", re.I)

# Variable substitution pattern (e.g., "x=3", "after substituting x=...")
VAR_SUB_PATTERN = re.compile(r"\b(\w+)\s*=\s*([\d\.]+)\b")


def tag_operation(text: str) -> Tuple[str, float]:
    """
    Tag operation type from text using keyword matching.
    Returns (operation, match_coverage) where coverage is proportion of pattern matched.
    """
    text_lower = text.lower()
    matches = []
    
    for op, pattern in OPS.items():
        match = re.search(pattern, text_lower, re.I)
        if match:
            # Calculate coverage: how much of the pattern matched relative to text length
            coverage = len(match.group()) / max(len(text), 1)
            matches.append((op, coverage))
    
    if not matches:
        return "other", 0.0
    
    # Return the operation with highest coverage (most specific match)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0]


def find_explicit_reference(text: str) -> Optional[int]:
    """
    Find explicit step/turn/line reference in text.
    Returns the referenced step number if found, None otherwise.
    """
    match = EXPLICIT_REF.search(text)
    if match:
        return int(match.group(2))
    return None


def extract_variable_substitutions(text: str) -> List[Tuple[str, str]]:
    """
    Extract variable assignments from text (e.g., "x=3", "let y=5").
    Returns list of (variable, value) tuples.
    """
    matches = VAR_SUB_PATTERN.findall(text)
    return matches


def find_concept_alignment(current_text: str, history: List[Dict]) -> Optional[int]:
    """
    Find dependency based on concept alignment (variable usage).
    If current turn uses x=3, find the last step that produced x=3.
    """
    current_vars = extract_variable_substitutions(current_text)
    if not current_vars:
        return None
    
    # Search backwards through history for matching variable assignments
    for i in range(len(history) - 1, -1, -1):
        hist_vars = extract_variable_substitutions(history[i]["text"])
        for var, val in current_vars:
            for hist_var, hist_val in hist_vars:
                if var.lower() == hist_var.lower() and val == hist_val:
                    return history[i]["turn_id"]
    
    return None


def check_operation_compatibility(op: str, prev_ops: List[Dict]) -> Optional[int]:
    """
    Check if current operation is compatible with previous operations.
    e.g., simplify usually depends on recent expand/substitute/isolate.
    """
    if op == "simplify":
        # Look for expand, substitute, or isolate in recent history
        compatible_ops = ["expand", "substitute", "isolate"]
        for i in range(len(prev_ops) - 1, -1, -1):
            if prev_ops[i]["operation"] in compatible_ops:
                return prev_ops[i]["turn_id"]
    
    elif op == "factor":
        # Factor might depend on expand
        for i in range(len(prev_ops) - 1, -1, -1):
            if prev_ops[i]["operation"] == "expand":
                return prev_ops[i]["turn_id"]
    
    elif op == "solve":
        # Solve might depend on isolate or simplify
        for i in range(len(prev_ops) - 1, -1, -1):
            if prev_ops[i]["operation"] in ["isolate", "simplify"]:
                return prev_ops[i]["turn_id"]
    
    return None


def calculate_confidence(
    op: str,
    has_explicit: bool,
    has_concept_match: bool,
    has_operation_compat: bool,
    match_coverage: float,
    is_root: bool
) -> float:
    """
    Calculate confidence score (0-1) based on various signals.
    """
    if is_root:
        # Root steps have moderate confidence
        return 0.5 if op != "other" else 0.3
    
    conf = 0.0
    
    # Explicit reference is strongest signal
    if has_explicit:
        conf += 0.6
    elif has_concept_match:
        conf += 0.5
    elif has_operation_compat:
        conf += 0.4
    else:
        # Default proximity heuristic
        conf += 0.3
    
    # Operation match quality
    if op != "other":
        conf += 0.2
        # Add bonus for good match coverage
        conf += min(match_coverage * 0.2, 0.2)
    else:
        conf += 0.1
    
    # Cap at 1.0
    return min(conf, 1.0)


def parse_conversation(conversation_str: str) -> List[Dict]:
    """
    Parse conversation string into list of turns with role and text.
    Format: "Teacher: (type)text|EOM|Student: text|EOM|..."
    """
    turns = []
    parts = conversation_str.split("|EOM|")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Match role and text
        # Generic matcher: anything ending with ":" is treated as a speaker label.
        # Normalize all non-Teacher speakers to Student.
        match = re.match(r"^([A-Za-z]+):\s*(\([^)]+\))?(.+)$", part)
        if match:
            role = match.group(1)
            # Normalize role names (student names -> Student)
            if role != "Teacher":
                role = "Student"
            text = match.group(3).strip()
            turns.append({"role": role, "text": text})
        else:
            # Fallback: if no role prefix, assume it's a continuation
            # In practice, this shouldn't happen with MathDial format
            if turns:
                turns[-1]["text"] += " " + part
    
    return turns


def extract_dependencies(conversation: List[Dict]) -> List[Dict]:
    """
    Extract dependencies from conversation.
    Only processes student turns.
    """
    student_turns = []
    dependencies = []
    
    # First pass: collect all student turns with their indices
    for i, turn in enumerate(conversation):
        if turn["role"].lower() == "student":
            op, match_coverage = tag_operation(turn["text"])
            student_turns.append({
                "turn_id": len(student_turns),  # Student turn index (0-based)
                "original_index": i,  # Original conversation index
                "text": turn["text"],
                "operation": op,
                "match_coverage": match_coverage
            })
    
    # Second pass: determine dependencies
    last_valid_student_id = None
    
    for idx, student_turn in enumerate(student_turns):
        turn_id = student_turn["turn_id"]
        text = student_turn["text"]
        op = student_turn["operation"]
        match_coverage = student_turn["match_coverage"]
        
        # Skip only if operation is "other" AND no explicit reference AND no mathematical content
        explicit_ref = find_explicit_reference(text)
        has_math_content = bool(re.search(r"\b(\d+|\+|\-|\*|/|=|equals|equal to)\b", text))
        
        # Include if: has operation, has explicit ref, has math content, or has previous ref
        if op == "other" and not explicit_ref and not PREVIOUS_REF.search(text) and not has_math_content:
            continue
        
        # Determine dependency (priority order)
        depends_on = None
        has_explicit = False
        has_concept_match = False
        has_operation_compat = False
        
        # Priority 1: Explicit reference
        if explicit_ref:
            # Convert explicit step number to student turn index
            # Note: explicit refs might be 1-based, and might refer to conversation turns
            # For simplicity, try to map to student turn indices
            # If ref is within reasonable range, use it
            ref_num = explicit_ref
            if 0 <= ref_num - 1 < len(student_turns):
                depends_on = ref_num - 1  # Convert to 0-based
                has_explicit = True
            elif ref_num <= len(student_turns):
                depends_on = ref_num - 1
                has_explicit = True
        elif PREVIOUS_REF.search(text):
            # Previous reference - use last valid student turn
            if last_valid_student_id is not None:
                depends_on = last_valid_student_id
                has_explicit = True
        
        # Priority 2: Concept alignment (variable usage)
        if depends_on is None:
            concept_match = find_concept_alignment(text, student_turns[:idx])
            if concept_match is not None:
                depends_on = concept_match
                has_concept_match = True
        
        # Priority 3: Operation compatibility
        if depends_on is None:
            compat_match = check_operation_compatibility(op, student_turns[:idx])
            if compat_match is not None:
                depends_on = compat_match
                has_operation_compat = True
        
        # Priority 4: Proximity default (last valid non-other student turn)
        if depends_on is None and last_valid_student_id is not None:
            depends_on = last_valid_student_id
        
        # Calculate confidence
        is_root = (depends_on is None)
        confidence = calculate_confidence(
            op, has_explicit, has_concept_match, has_operation_compat,
            match_coverage, is_root
        )
        
        # Create dependency entry (format: qid first as in requirements)
        dep_entry = {
            "qid": None,  # Will be set later in run()
            "turn_id": turn_id,
            "operation": op,
            "depends_on": depends_on,
            "confidence": round(confidence, 2)
        }
        
        dependencies.append(dep_entry)
        
        # Update last valid student turn (for proximity heuristic)
        # Include any turn that has an operation or mathematical content
        if op != "other" or has_math_content:
            last_valid_student_id = turn_id
    
    return dependencies


def run(input_path: str, output_path: str):
    """
    Main function to process MathDial data and extract dependencies.
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Process input file (JSONL format)
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                ex = json.loads(line)
                qid = ex.get("qid")
                conversation_str = ex.get("conversation", "")
                
                if not conversation_str:
                    continue
                
                # Parse conversation
                conversation = parse_conversation(conversation_str)
                
                # Extract dependencies
                deps = extract_dependencies(conversation)
                
                # Write each dependency as a separate JSONL line
                # Format: qid first (as in requirements example)
                for dep in deps:
                    dep["qid"] = qid
                    # Reorder to match requirement format: qid, turn_id, operation, depends_on, confidence
                    ordered_dep = {
                        "qid": dep["qid"],
                        "turn_id": dep["turn_id"],
                        "operation": dep["operation"],
                        "depends_on": dep["depends_on"],
                        "confidence": dep["confidence"]
                    }
                    f_out.write(json.dumps(ordered_dep, ensure_ascii=False) + "\n")
            
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON line: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line: {e}")
                continue


if __name__ == "__main__":
    import sys
    
    # Default paths
    default_input = "data/example.jsonl"
    default_output = "results/dependencies.jsonl"
    
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    # Resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(project_root, input_path) if not os.path.isabs(input_path) else input_path
    output_path = os.path.join(project_root, output_path) if not os.path.isabs(output_path) else output_path
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    run(input_path, output_path)
    print("Dependency extraction completed!")

