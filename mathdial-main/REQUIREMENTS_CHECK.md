# 需求实现检查清单

## ✅ 核心功能

### 输入/输出
- [x] **输入**：支持 `dataset/mathdial.json` 或官方数据路径（JSONL格式）
- [x] **产出**：`results/dependencies.jsonl` - 一行一条依赖边或根步骤
- [x] **格式**：`{"qid": ..., "turn_id": ..., "operation": ..., "depends_on": ..., "confidence": ...}` ✓

### 步骤标注
- [x] **仅遍历 role == "student" 的 turn** ✓
- [x] **基于规则+词典识别操作类型** ✓
- [x] **取主操作**（按优先级）✓

### 操作类型词典（需求中的所有类型）
- [x] `substitute`: substitut|plug in|let x=|replace x with ✓
- [x] `simplify`: simplif|reduce|combine (like )?terms|collect terms|cancel ✓
- [x] `expand`: expand|distribute ✓
- [x] `factor`: factor|take out|common factor ✓
- [x] `isolate`: isolate|move .* to the other side|bring .* over ✓
- [x] `differentiate/integrate` ✓
- [x] `solve`: solve for ✓
- [x] `check`: check|verify|compare ✓
- [x] `other`: fallback ✓

**扩展**（超出需求，但合理）：
- `calculate`: 自然语言模式（figure out, calculated等）
- `divide/multiply/add/subtract`: 基础运算识别

### 依赖启发式（优先级从高到低）
- [x] **优先级1：显式引用** - step 2, from (turn|line) 2, previous substitution ✓
- [x] **优先级2：同概念对齐** - x=3的使用，回溯产生x=3的步骤 ✓
- [x] **优先级3：操作相容性** - simplify依赖expand/substitute/isolate ✓
- [x] **优先级4：近因默认** - 依赖最近一个"非other"的学生步骤 ✓

### 置信度打分
- [x] **0-1分数**，基于：
  - 规则匹配覆盖率 ✓
  - 显式线索 ✓
  - 概念对齐 ✓
  - 操作兼容性 ✓

### 验收标准
- [x] **所有条目含** qid/turn_id/operation/depends_on/confidence ✓
- [x] **根步骤比例** 17.9% （在15-35%合理范围内）✓
- [x] **格式正确** - depends_on 为 null（根步骤）或数字（依赖步骤）✓

### 常见坑处理
- [x] **不混入 teacher 回合** ✓
- [x] **多个操作词取主操作** ✓
- [x] **边界约束** - 使用 `\bstep\b` 避免误匹配 ✓

## 📊 验证结果

### 数据统计（train.jsonl）
- 总条目：12,019
- 根步骤：2,155 (17.9%)
- 操作类型分布：
  - other: 64.9%
  - multiply: 9.0%
  - calculate: 7.5%
  - subtract: 7.0%
  - add: 5.3%
  - divide: 4.5%
  - solve: 1.2%
  - check: 0.3%
  - substitute: 0.3%
  - simplify/factor/expand/isolate: <0.5%

### 输出示例
```json
{"qid": "5000794", "turn_id": 0, "operation": "calculate", "depends_on": null, "confidence": 0.5}
{"qid": "5000794", "turn_id": 1, "operation": "other", "depends_on": 0, "confidence": 0.4}
```

## ✅ 结论

**所有需求核心功能已实现** ✓

