# 8803-final-project
✅ Stage A — Dependency Extraction (Researcher A)
🎯 Goal

从学生-老师数学对话中，识别学生的子步骤之间的依赖关系，构建 student-step dependency graph。

🔍 Motivation

原 MathDial 数据集中没有 turn-level 行为结构。
我们新增：

每个 turn 的操作/技能分类 (operation)

当前 turn 依赖哪个先前 turn (depends_on)

模型对依赖判断的可信度 (confidence)

为后续教学反馈奠定结构基础。

📥 Input Files
mathdial-main/data/train.jsonl
mathdial-main/data/test.jsonl

📤 Output Files
research_mathdial/results/dependencies_train.jsonl
research_mathdial/results/dependencies_test.jsonl

✅ Output Format (example)
{"qid": 5000012, "turn_id": 1, "operation": "simplify", "depends_on": 0, "confidence": 0.41}

📌 Dependencies extracted:

What step is this turn building on?

Where did the student misunderstand?

Graph edges for later reference.

✅ Stage B — Teacher Signal Generation (Researcher B)
🎯 Goal

使用 LLM 生成更丰富、更教育学导向的 teacher feedback signals。

你扩展了原 MathDial（只有 CoT）的 supervision：

✅ chain-of-thought reasoning (thought)
✅ dependency justification (dependency_reason)
✅ hint taxonomy (hint_type)
✅ symbolic math comment (symbolic_comment)
✅ calibrated confidence

🔍 Motivation

让 tutor:

更 explainable

更 interactive

更 curricular progression

📥 Input
mathdial-main/data/train.jsonl                    (raw dialogue)
research_mathdial/results/dependencies_train.jsonl (Stage A)

📤 Output
research_mathdial/results/teacher_signals_train.jsonl

✅ Output Format (example)
{
  "qid": "5000012",
  "turn_id": 1,
  "thought": "The student used addition instead of multiplication.",
  "dependency_reason": "This builds on turn 0, where operation was misapplied.",
  "ref_turn": 0,
  "hint_type": "procedural",
  "symbolic_comment": "Check parentheses and units.",
  "confidence": 0.71
}

📌 New supervision labels created:

3-way pedagogy hint taxonomy

explicit reference turn

symbolic mathematical heuristics

dependency reasoning

✅ Stage C — Fine-Tuning & Ablation Training (Researcher C)
🎯 目标

让你的模型学会自动生成：

thought

dependency_reason

ref_turn

hint_type

symbolic_comment

confidence calibration

并且测试不同 supervision 的贡献。

📁 输入文件

来自 A / B 阶段：

mathdial-main/data/train.jsonl
research_mathdial/results/dependencies_train.jsonl
research_mathdial/results/teacher_signals_train.jsonl


你需要将这三类信息 merge 成训练样本。

🧩 数据打包格式

最终训练样本应含如下字段：

{
  "qid": "5000012",
  "turn_id": 1,
  "input": "<student turn text> ...",
  "dependency_edges": [0],
  "previous_operations": ["calculate", "other"],
  "previous_confusion_summary": "misunderstood multiplication",
  "output": {
    "thought": "...",
    "dependency_reason": "...",
    "ref_turn": 0,
    "hint_type": "procedural",
    "symbolic_comment": "Check parentheses and units.",
    "confidence": 0.71
  }
}

🧠 数据处理步骤
Step C1. 合并 raw turns + dependency
python research_mathdial/backend/build_training_examples.py \
  --data mathdial-main/data/train.jsonl \
  --deps research_mathdial/results/dependencies_train.jsonl \
  --signals research_mathdial/results/teacher_signals_train.jsonl \
  --out research_mathdial/results/training_examples.jsonl


输出：

research_mathdial/results/training_examples.jsonl

🧪 Ablation Conditions

你将训练四个模型：

Condition	Data
Baseline	student turn only
+CoT	+thought
+Dependency	+dependency_reason +ref_turn
Full Pedagogy	+symbolic_comment +confidence
🐦 模型推荐

Qwen 1.5B instruct

Llama3 2B

Mistral 7B

（小，大都可以——重点是对比）

🚀 Fine-tuning 命令示例（LoRA）
python train.py \
  --dataset research_mathdial/results/training_examples.jsonl \
  --model qwen2.5:1.5b-instruct \
  --epochs 2 \
  --lr 3e-5 \
  --batch-size 4 \
  --save checkpoints/stageC_full.pt

📤 输出文件
checkpoints/stageC_full.pt
checkpoints/stageC_cot.pt
checkpoints/stageC_dep.pt
checkpoints/stageC_base.pt

📏 Stage C 评估步骤
Step C2. Evaluate with held-out validation set
python evaluate.py \
  --model checkpoints/stageC_full.pt \
  --data research_mathdial/results/validation_examples.jsonl \
  --metrics accuracy,ref_turn_f1,hint_type_f1


产生：

research_mathdial/results/eval_full.json

📈 Stage C 输出指标

hint_type-F1

ref_turn accuracy

symbolic_comment coverage

confidence calibration MAE

dependency_reason BLEU/METEOR

✅ Stage D — Interactive Demo + Evaluation (Researcher D)
🎯 目标

将 C 训练好的模型作为 tutor：

和学生交互

输出结构化反馈

修正错误

比较原混乱输出 vs ours

📁 输入文件
checkpoints/stageC_full.pt
research_mathdial/results/dependencies_test.jsonl
mathdial-main/data/test.jsonl

🧑‍💻 Step D1. 启动 Streamlit
streamlit run research_mathdial/app/interactive_tutor.py

🧩 Streamlit UI 结构

UI 页面包含：

总览 Panel

学生回答文本框

dependency graph 可视化（可选）

模型反馈卡片区

信心条（confidence horizontal bar）

Symbolic comment badge

Ref-turn highlight

🧠 互动逻辑

每次用户输入 → 模型输出：

{
  "tutor_turn": 3,
  "ref_turn": 1,
  "hint_type": "procedural",
  "dependency_reason": "This turn builds on misunderstanding in turn #1.",
  "symbolic_comment": "Check parentheses when multiplying.",
  "confidence": 0.74,
  "next_question": "Try recomputing using multiplication, not repetition."
}


存为：

research_mathdial/results/interactive_logs/qid_5000012_ours.jsonl

🔁 Step D2. Replay 原始 MathDial
python research_mathdial/backend/replay_original.py \
  --qid 5000012 \
  --out research_mathdial/results/interactive_logs/qid_5000012_original.jsonl

🔍 Step D3. 差异比较
python research_mathdial/backend/compare_transcripts.py \
  --original research_mathdial/results/interactive_logs/qid_5000012_original.jsonl \
  --ours research_mathdial/results/interactive_logs/qid_5000012_ours.jsonl \
  --out research_mathdial/results/diffs/qid_5000012_diff.json


输出包含：

turn count difference

hint precision

dependency coverage

📊 Step D4. Chart Generation
python research_mathdial/backend/generate_charts.py \
  --logs research_mathdial/results/interactive_logs \
  --out research_mathdial/results/charts


生成：

charts/hint_types.png
charts/confidence_hist.png
charts/turn_efficiency.png
charts/ref_turn_usage.png
charts/symbolic_comment_rate.png


这些图可以放 PPT。

🎤 Step D5. Explainability Visualization

Optional：

Sankey diagram (hint type transitions)

Dependency heatmap

🧪 Evaluation Metrics for Stage D
Metric	Why
turn_efficiency	fewer turns ⇒ better remediation
hint_type precision	pedagogy correctness
ref_turn accuracy	context awareness
confidence calibration	trustworthy
symbolic_comment rate	actionable