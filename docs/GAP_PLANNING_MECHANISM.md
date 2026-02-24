# GAP Planning Mechanism: Paper vs Code (Chi tiet ky thuat)

## 1) Muc tieu tai lieu
Tai lieu nay mo ta day du co che planning trong GAP theo hai goc nhin:
- **Muc tieu thuat toan** trong paper (Section 3: Graph-based Agent Planning Paradigm).
- **Hanh vi runtime thuc te** trong ma nguon public hien tai.

Muc dich la giup team:
- Hieu ro prompt planning, planning format, cach parser xu ly tool call.
- Hieu ro dependency duoc xu ly nhu the nao trong runtime.
- Nhan dien khoang cach giua paper-claim va implementation de co huong nang cap.

---

## 2) Tong quan ngan: Planning trong GAP la gi?
Voi GAP, "planning" khong chi la chia bai toan lon thanh task nho. Day la to hop 4 viec:
1. Tach bai toan thanh sub-task.
2. Mo ta dependency giua sub-task.
3. Chon tool cho moi sub-task.
4. Quyet dinh task nao chay song song, task nao chay tuan tu.

Trong paper, (2) va (4) duoc mo ta duoi dang **DAG + topological levels**. Trong code public, phan runtime hien tai tap trung vao **parallel tool calls** theo tung turn, chua co module DAG scheduler tach rieng.

---

## 3) Planning paradigm trong paper (Section 3)
Nguon: `GAP-2510.25320.pdf` (Section 3, page 3-5).

### 3.1 Problem Formulation
Paper dinh nghia:
- Query phuc tap `q`.
- Tap cong cu `T = {t1..tn}`.
- Tap sub-task `S = {s1..sm}`.
- Do thi phu thuoc co huong khong chu trinh `G = (V, E)`.

Y nghia:
- Node = sub-task.
- Edge `(si -> sj)` = `sj` can output cua `si`.
- Khong co edge = doc lap => co the chay song song.

### 3.2 Graph-based Task Decomposition
Paper mo ta 3 buoc:
1. Sub-task identification.
2. Dependency analysis.
3. Graph construction.

Paper dua ra format graph ro rang, vi du:
```xml
<graph>
  <node id="s1">search("capital of France")</node>
  <node id="s2">search("capital of Germany")</node>
  <node id="s3" depends="s1">search("population of {s1}")</node>
  <node id="s4" depends="s2">search("population of {s2}")</node>
</graph>
```

### 3.3 Dependency-Aware Execution Strategies
Paper mo ta scheduling theo level:
- `L0`: node khong co incoming edge.
- `Li`: node co dependency nam trong `L0..L(i-1)`.
- Moi level chay song song.
- Giua cac level: barrier/wait day du ket qua roi moi sang level tiep theo.

Algorithm 1 trong paper co cac buoc:
1. Sinh plan.
2. `ParseGraph(plan)`.
3. `TopologicalSort(G)`.
4. Vong lap level-wise execution.

---

## 4) Prompt planning trong ma nguon

### 4.1 Prompt variants
File: `Agent/data/mhqa_agent/sys_prompts.py`

Co 3 cum prompt chinh lien quan planning:
- `MHQA_PROMPT`: plan co ban (sub-question decomposition).
- `MHQA_PROMPT_ONE_SHOT`: bo sung one-shot trajectory.
- `ENHANCED_MHQA_PROMPT`: yeu cau task co `Task ID` + `Dependencies`, va huong dan parallel bang dau `|` trong `wiki_search`.

### 4.2 Planning format duoc huong dan cho model
Trong `ENHANCED_MHQA_PROMPT`, format plan mang tinh graph-like:
- Task ID (`T1`, `T2`, ...)
- Description
- Dependencies (`none` hoac danh sach task)

Vi du trong prompt mau:
```xml
<plan>
T1: ...
- Dependencies: none
T2: ...
- Dependencies: none
T3: ...
- Dependencies: T1, T2
</plan>
```

Va thuc thi song song:
```xml
<wiki_search>query_1|query_2</wiki_search>
```

### 4.3 Prompt nao dang duoc dung trong pipeline mac dinh?
File: `Agent/data/mhqa_agent/prepare_data.py`

`SYS_PROMPT_MAP` dang map:
- `SR1`
- `MHQA_PROMPT` -> `MHQA_PROMPT_ONE_SHOT`
- `WEB_AGENT_PROMPT`

**Luu y quan trong**: `ENHANCED_MHQA_PROMPT` da import nhung **chua duoc map trong `SYS_PROMPT_MAP`**.
Do do, neu khong sua code/arg, pipeline du lieu mac dinh khong bat buoc format dependency-chat nhu ban enhanced.

Ngoai ra, script `Agent/data/mhqa_agent/prepare.sh` dang dung:
```bash
--sys-prompt-key MHQA_PROMPT
```

---

## 5) Planning va execution trong runtime (code path thuc te)

### 5.1 Tool awareness va planning context
Runtime nap tool tu config YAML:
- `tool_config_path` -> tao `tool_schemas` va `tool_map`.
- Model chi thay va goi cac tool co trong schemas.

Cac diem chinh:
- `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py` (init tools/parsers)
- `verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml` (single tool: `wiki_search`)
- `Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh` (dang tro den config wiki-only)

=> Planner co awareness ve tool theo runtime config, nhung awareness nay mang tinh schema-driven.

### 5.2 Parser hien tai parse cai gi?
File: `verl/verl/tools/xml_tool_parser.py`

Parser hien tai:
- Parse cac tag tool (`<wiki_search>...</wiki_search>`, ...).
- Dac biet voi `wiki_search`: split query theo `|` de tao nhieu tool call.

Parser **khong parse `<plan>` thanh DAG**.
Parser cung khong parse `<graph>` node/edge theo dung mo ta Section 3 paper.

### 5.3 Event loop runtime multi-turn
File: `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`

Vong lap xu ly request:
1. Sinh text tu model.
2. Kiem tra text co tool-call tag khong.
3. Neu co, parse thanh danh sach `parsed_tool_calls`.
4. Chay tool calls (co the song song) bang `asyncio.gather(...)`.
5. Them `tool responses` vao history.
6. Quay lai model cho turn tiep.

Mau luong:
- `RUNNING` -> detect tool tags -> `TOOL_CALLING`.
- Trong `TOOL_CALLING`: execute concurrent tasks, append observations.
- Back ve `RUNNING`.

### 5.4 Parallel execution thuc te
Parallel o runtime xay ra theo 2 cach:
- Trong cung mot turn, neu parser tra ve nhieu tool calls (vi du tu `|`) -> gather song song.
- Giua cac turn van tuan tu (turn barrier): phai co observation xong moi sinh tiep.

### 5.5 Dependency handling thuc te
Co che dependency hien tai la **implicit dependency**:
- Model tu nho trang thai trong context (`messages`).
- Model tu quyet dinh task tiep theo dua tren observation vua nhan.
- Runtime khong co DAG validator/topological scheduler de enforce hard dependency.

Noi cach khac:
- Paper: dependency explicit (do thi + level).
- Runtime: dependency emergent tu trajectory multi-turn.

### 5.6 Guardrails trong runtime
Co mot so guardrails:
- `max_turns`.
- `max_model_len`.
- duplicate query guard:
  - Neu query da search/crawl truoc do, tra canned response thay vi goi lai tool that.

Guardrails nay giup tiet kiem tai nguyen va han che loop, nhung khong phai dependency scheduler.

---

## 6) Cach parse "ke hoach" hien tai: thuc te va he qua

### 6.1 Co parse `<plan>` khong?
**Khong** o runtime.
`<plan>` chu yeu la token de huong dan model lap luan va tao trajectory co cau truc.

### 6.2 Co parse `<graph>` khong?
Trong code public hien tai khong thay:
- `ParseGraph(...)`
- `TopologicalSort(...)`
- executor theo level `L0, L1, ...`

### 6.3 He qua
Uu diem:
- Kien truc don gian.
- De huan luyen va rollout.
- Parallel call da co hieu qua nhat dinh.

Han che:
- Khong co hard guarantee rang task phu thuoc duoc ton trong.
- Khong co static check cho plan correctness.
- Kho debug planning error theo do thi (vi khong co IR graph).

---

## 7) Format planning: de xuat chuan hoa de giam ambiguity

Neu muon planning thuc su graph-native, nen dung 2 tang format:

### Tang 1: Human-readable plan
```xml
<plan>
T1: ...
- Dependencies: none
T2: ...
- Dependencies: T1
</plan>
```

### Tang 2: Machine-readable graph IR
```json
{
  "nodes": [
    {"id": "T1", "tool": "wiki_search", "args": {"query": "..."}, "depends_on": []},
    {"id": "T2", "tool": "wiki_search", "args": {"query": "..."}, "depends_on": ["T1"]}
  ]
}
```

Parser chi can consume Tang 2 de scheduler.
Tang 1 giu cho kha nang giai thich.

---

## 8) Co che parse plan + xu ly dependency de xuat (tham chieu implementation)

### 8.1 Parse
- Parse JSON graph IR thanh struct noi bo `PlanGraph`.
- Validate:
  - node IDs duy nhat.
  - tool nam trong `tool_map`.
  - dependency refs hop le.
  - detect cycle.

### 8.2 Scheduling
- Chay Kahn topological sort de tao levels.
- Moi level tao `tool batch`.
- Execute `asyncio.gather` tren level.
- Merge observations vao state memory.

Pseudo-code:
```python
graph = parse_and_validate(plan_json)
levels = topo_levels(graph)
state = {}
for level in levels:
    calls = [materialize(node, state) for node in level]
    results = await asyncio.gather(*[exec_tool(c) for c in calls])
    state.update(bind_results(level, results))
final = synthesize_answer(state)
```

### 8.3 Error handling
- Tool fail -> retry theo policy hoac mark node failed.
- Dependency failed -> skip downstream node hoac fallback re-plan.
- JSON parse fail -> fallback ve che do current trajectory (best-effort).

### 8.4 Noi chen vao codebase hien tai
Vi tri de mo rong:
- `verl/verl/tools/xml_tool_parser.py`: them `plan/graph parser`.
- `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`: chen scheduler mode truoc khi execute tool calls.
- Reward shaping (`verl/verl/utils/reward_score/mhqa_train.py`): bo sung signal efficiency/dependency compliance neu can.

---

## 9) Doi chieu nhanh: Paper claim vs code public

### Phu hop
- Co decomposition mindset qua prompt.
- Co parallel tool execution trong cung turn.
- Co training pipeline multi-turn tool-integrated reasoning.

### Chua phu hop hoan toan
- Chua thay explicit `ParseGraph + TopologicalSort` runtime.
- Chua thay executor level-wise theo DAG.
- Dependency duoc xu ly ngam qua context, khong co hard enforcement.

Ket luan ky thuat:
- Implementation hien tai la **graph-inspired planning runtime**.
- Chua dat muc **explicit graph-orchestrated runtime** nhu pseudocode Algorithm 1.

---

## 10) Checklist khi review planning trong repo nay
1. Dang dung prompt key nao (`MHQA_PROMPT` hay enhanced)?
2. Tool config co bao nhieu tool va ten tool la gi?
3. Parser co parse plan/graph hay chi parse tool tag?
4. Parallel dang xay ra o dau (within-turn hay cross-level)?
5. Dependency dang duoc enforce bang scheduler hay chi nguyen tac trong prompt?
6. Reward co khuyen khich planning efficiency/dependency compliance khong?

---

## 11) Nguon ma da doi chieu
- `Agent/data/mhqa_agent/sys_prompts.py`
- `Agent/data/mhqa_agent/prepare_data.py`
- `Agent/data/mhqa_agent/prepare.sh`
- `Agent/train/mhqa_agent/rl/train_dapo_mhqa_agent_wiki.sh`
- `verl/verl/tools/xml_tool_parser.py`
- `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`
- `verl/verl/tools/config/search_tool_config/wiki_rag_config.yaml`
- `verl/verl/utils/reward_score/mhqa_train.py`
- `GAP-2510.25320.pdf` (Section 3, Algorithm 1)

