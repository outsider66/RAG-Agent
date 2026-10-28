# 智扫通 · 扫地机器人智能客服（RAG + Agent 学习项目）

一个基于 **LangChain 1.x + LangGraph + Streamlit + 阿里云百炼（DashScope）** 的智能客服 Demo：
把本地知识库做成 RAG 检索工具，再交给 ReAct Agent 自主决定「要不要检索、查什么、要不要再查一次」，最后在 Streamlit 里以对话形式输出。

适合学习以下内容：

- RAG 的完整链路：文档加载 → 分片 → 向量化 → 入库 → 检索 → 拼接上下文 → 让模型总结
- LangChain 1.x 的 Agent 构建方式：`create_agent` + `system_prompt` + `tools` + `middleware`
- 工具（Tool）的定义与调用、动态提示词切换、工具调用监控
- Streamlit 的会话状态（`session_state`）、token 级流式输出与主题定制

---

## 一、整体架构

```
用户提问
   │
   ▼
Streamlit (app.py)
   │  记录消息 → rerun → 消费 (kind, text) 事件，逐 token 渲染
   ▼
ReactAgent (agent/react_agent.py)  ← 由 create_agent 构建
   │
   ├─ middleware（agent/tools/middleware.py）
   │     ├─ before_model    : 记录每次调用模型前的状态
   │     ├─ wrap_tool_call  : 记录工具入参/结果；识别报告场景
   │     └─ dynamic_prompt  : 运行时切换「普通问答」/「报告生成」提示词
   │
   ├─ tools（agent/tools/agent_tools.py）
   │     ├─ rag_summarize          → RAG 检索并总结（核心）
   │     ├─ get_weather             → 模拟天气
   │     ├─ get_user_location       → 模拟定位
   │     ├─ get_user_id             → 模拟用户 ID
   │     ├─ get_current_month       → 模拟当前月份
   │     ├─ fetch_external_data     → 读 data/external/records.csv
   │     └─ fill_context_for_report → 只用于把 context["report"] 置为 True
   │
   └─ 模型（model/factory.py → ChatTongyi / DashScope）
            │
            └─ 被 rag_summarize 内部再次调用（总结检索结果）
```

要点：**同一个模型会被调用两次**——一次是 Agent 决定「调用哪个工具」，一次是 RAG 工具内部「根据参考资料写答案」。

---

## 二、目录结构

```
AI大模型RAG与智能体开发_Agent项目/
├── app.py                      # Streamlit 入口（UI + 会话状态 + 流式输出）
├── .streamlit/
│   └── config.toml             # 主题、配色、圆角、工具栏模式
├── config/                     # 所有可调参数（YAML）
│   ├── rag.yml                 # 模型名、embedding 名
│   ├── chroma.yml              # 向量库、分片参数、知识库目录
│   ├── agent.yml               # 外部数据文件路径
│   └── prompts.yml             # 各提示词文件路径
├── prompts/                    # 提示词正文（纯文本，便于随时改）
│   ├── main_prompt.txt         # 主 Agent 系统提示词
│   ├── rag_summarize.txt       # RAG 总结提示词（含 {input} {context}）
│   └── report_prompt.txt       # 报告生成提示词（场景切换后使用）
├── agent/
│   ├── react_agent.py          # Agent 组装 + 流式执行
│   └── tools/
│       ├── agent_tools.py      # 工具实现
│       └── middleware.py       # 中间件（监控 / 日志 / 提示词切换）
├── rag/
│   ├── vector_store.py         # 加载文档、MD5 增量去重、自动灌库/自愈
│   └── rag_service.py          # 检索 + 提示词 + 模型 = 总结链
├── model/factory.py            # 模型工厂（Chat / Embedding）
├── utils/
│   ├── config_handler.py       # 读取 YAML 配置
│   ├── path_tool.py            # 统一绝对路径
│   ├── logger_handler.py       # 日志（控制台 + 按天文件）
│   ├── file_handler.py         # MD5、目录扫描、PDF/TXT 加载
│   └── prompt_loader.py        # 读取提示词文件
├── data/                       # 知识库原始文档（txt / pdf）
│   └── external/records.csv    # 模拟的用户使用记录
├── chroma_db/                  # 向量库持久化目录（自动生成）
├── md5.text                    # 已入库文件的 MD5，用于去重（自动生成）
└── logs/                       # 运行日志（按天分文件）
```

---

## 三、核心模块逐个讲

### 1. 配置层 `utils/config_handler.py`

模块被 import 时就把 4 个 YAML 一次性读进内存（`rag_conf` / `chroma_conf` / `prompts_conf` / `agent_conf`），全局共享。

> 学习点：配置与代码分离。换模型、调分片大小、换提示词都不用改 Python。

### 2. 路径层 `utils/path_tool.py`

`get_abs_path("config/rag.yml")` → 基于**当前文件位置**反推工程根目录，再拼相对路径。
好处是不依赖「从哪个目录启动」，路径永远相对于项目根。

### 3. 向量库 `rag/vector_store.py`

`load_document()` 的流程：

1. `listdir_with_allowed_type()` 扫描 `data/` 下允许的后缀（txt / pdf）
2. `get_file_md5_hex()` 算文件 MD5
3. `check_md5_hex()` 查 `md5.text`，命中就跳过（**增量入库、避免重复向量化**）
4. `txt_loader` / `pdf_loader` 读取为 LangChain `Document`
5. `RecursiveCharacterTextSplitter` 按 `chunk_size=200 / overlap=20` 切片
6. `vector_store.add_documents()` 写入 Chroma，并记录 MD5

另外有两个兜底方法：

- `is_empty()`：判断向量库分片数是否为 0
- `ensure_loaded()`：先跑一次增量加载；如果 `chroma_db` 被清空但 `md5.text` 还留着旧记录，
  会自动重置加载记录并重新灌库，避免「本地有资料却检索不到」

> `RagSummarizeService.__init__` 会调用 `ensure_loaded()`，所以首次启动、新增文档、清空过向量库等情况都会自动处理，
> 不再依赖手动执行 `python -m rag.vector_store`（该命令仍可用于强制排查/重建）。

### 4. RAG 总结链 `rag/rag_service.py`

```python
chain = prompt_template | print_prompt | model | StrOutputParser()
```

`rag_summarize(query)`：检索 top-k 文档 → 拼成 `【参考资料n】...` → 和用户问题一起喂给模型 → 返回总结文本。
`print_prompt` 是调试用的，会把最终 prompt 打到控制台（生产环境应删除）。

### 5. 模型工厂 `model/factory.py`

```python
chat_model  = ChatTongyi(model=rag_conf["chat_model_name"], streaming=True)  # 对话模型
embed_model = DashScopeEmbeddings(model=rag_conf["embedding_model_name"])    # 向量模型
```

> `streaming=True` 必须显式打开：默认 `False` 时模型一次性返回整段文本，
> 即使 Agent 用 `stream_mode="messages"` 也拿不到 token，前端就看不到逐字效果。

> **重要**：`ChatTongyi` 走的是 DashScope **原生 API**（`/api/v1/services/...`）。
> 并非所有模型都在该端点上存在——用不存在的模型会报 `400 InvalidParameter: url error`。

### 6. 工具定义 `agent/tools/agent_tools.py`

用 `@tool(description="...")` 装饰普通函数即可。**description 是给模型看的**，写得越明确，模型选工具越准。
函数签名（参数名 + 类型）会自动变成工具的 JSON Schema，模型据此填参。

`fetch_external_data` 首次调用时用 `generate_external_data()` 懒加载 CSV，构建成
`{user_id: {month: {特征/效率/耗材/对比}}}` 的嵌套字典。

### 7. 中间件 `agent/tools/middleware.py`

三种中间件（LangChain 1.x 新特性，等价于「钩子」）：

| 装饰器 | 时机 | 本项目的用途 |
|---|---|---|
| `@before_model` | 每次调模型前 | 打印消息条数、最后一条消息 |
| `@wrap_tool_call` | 包住每次工具执行 | 记录工具名/入参/成功失败；检测到 `fill_context_for_report` 就把 `runtime.context["report"]` 置 True |
| `@dynamic_prompt` | 每次生成提示词前 | 按 `context["report"]` 决定用主提示词还是报告提示词 |

这就是「**运行时提示词切换**」的实现方式：模型先按主提示词思考，决定调用 `fill_context_for_report`；
工具执行后 context 被改写，下一轮模型自动换成报告专用提示词。

### 8. Agent 组装 `agent/react_agent.py`

```python
create_agent(model=..., system_prompt=..., tools=[...], middleware=[...])
```

`execute_stream(query)` 用 `stream_mode="messages"`，逐 token 产出事件元组 `(kind, text)`：

- `("answer", 文本)`：最终回答片段，调用方直接拼接并渲染
- `("reset", "")`：这一轮模型其实要调工具，先前的“思考文字”需要从界面撤回

判断逻辑：

1. 只处理 `metadata["langgraph_node"] == "model"` 的 `AIMessageChunk`，工具节点直接跳过
2. 用 `metadata["langgraph_step"]` 区分每一轮模型调用
3. 当前轮出现 `tool_call_chunks` → 说明要调工具，抛一次 `reset`，并屏蔽该轮后续文本
4. 没有工具调用的轮次，文本才作为正式回答输出

这样既做到 token 级流式，又不会把“我需要先检索…”这类中间思考混进最终回复。

### 9. UI 层 `app.py`

- `get_agent()`：`@st.cache_resource` 懒加载 Agent，首屏不初始化模型，也避免每次 rerun 重建
- `session_state["message"]`：聊天历史（role / content / time / meta），页面重跑时按序重放
- `session_state["pending_prompt"]`：快捷提问按钮回调写入的问题
- `session_state["awaiting_reply"]`：两阶段渲染——先把用户消息入库并 `st.rerun()`，
  下一轮再真正生成回答，保证侧边栏“对话轮数”等统计当轮就正确
- `handle_prompt()`：`st.status` 展示「连接 → 理解 → 调工具 → 生成」阶段，
  自己消费 `(kind, text)` 事件增量更新 Markdown，并带 `▍` 光标；`reset` 事件会清空刚渲染的思考文本
- 视觉：`.streamlit/config.toml` 定义主题；`app.py` 内用 `st.html` 注入 Hero 横幅与聊天排版样式
- 欢迎页每轮都渲染，有对话时用 `.st-key-welcome` 的 CSS 隐藏，避免生成过程中旧卡片以半透明残影残留
- 失败兜底：`remember_failed_answer()` 把错误回答也写入历史，附带可展开的错误详情

---

## 四、运行

### 环境要求

- Python **≥ 3.10**（本项目实测 3.12）
- 主要依赖（`rag_env` 实测版本）：

```
langchain 1.3.11        langchain-core 1.4.8      langchain-community 0.4.2
langgraph 1.2.6         langchain-chroma 1.1.0    langchain-openai 1.3.3
streamlit 1.60.0        chromadb 1.5.9            dashscope 1.26.0
pypdf 6.14.2            PyPDFLoader / TextLoader（langchain-community）
```

- 环境变量：

```powershell
$env:DASHSCOPE_API_KEY = "sk-xxxx"
```

### 启动命令（必须在项目根目录执行，否则 `agent` / `rag` / `utils` 包导入失败）

```powershell
# 1.（可选）手动灌库 / 排查向量库状态；正常启动会自动加载与增量更新
python -m rag.vector_store

# 2. 启动界面
streamlit run app.py
```

如只想单独测 Agent（不经过 UI）：

```powershell
python -m agent.react_agent
```

---

## 五、配置速查

`config/rag.yml`

```yaml
chat_model_name: qwen3.7-max-2026-05-20   # 必须是 DashScope 原生 API 支持的模型
embedding_model_name: qwen3.7-text-embedding-flash
```

原生 API 实测可用的对话模型：`qwen3.7-max`、`qwen3.7-max-2026-05-20`、`qwen-plus`、`qwen-plus-latest`、`qwen-max`。
`qwen3.8-flash` / `qwen3.8-max-0902` 只在 **OpenAI 兼容模式** 可用，直接给 `ChatTongyi` 会报 `url error`。

`config/chroma.yml`

```yaml
collection_name: agent
persist_directory: chroma_db
k: 3                      # 检索条数
data_path: data
md5_hex_store: md5.text
allow_knowledge_file_type: ["txt", "pdf"]
chunk_size: 200
chunk_overlap: 20
```

`.streamlit/config.toml`（主题）

```toml
[client]
toolbarMode = "minimal"      # 精简右上角工具栏

[theme]
base = "light"
primaryColor = "#4f46e5"
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#1e293b"
baseRadius = "0.65rem"
buttonRadius = "0.65rem"

[theme.dark]                 # 跟随系统深色模式时的配色
primaryColor = "#818cf8"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"
```

---

## 六、踩坑记录（排错顺序）

1. **`ModuleNotFoundError: No module named 'path_tool' / 'config_handler'`**
   → `utils/` 内部互相导入必须写 `from utils.xxx import ...`（或相对导入），不能写成顶层模块名。

2. **`TypeError: unsupported operand type(s) for |`**
   → `Embeddings | BaseChatModel` 是 PEP 604 语法，Py3.9 不支持，必须 3.10+。

3. **`403 AllocationQuota.FreeTierOnly`**
   → 百炼账号免费额度用尽，且控制台开着「仅使用免费额度」。充值或关闭该开关，或换 Key。

4. **`400 InvalidParameter: url error, please check url`**
   → 模型名在原生 API 上不存在（例如用了 3.8 系列）。换模型，或改用兼容模式（`ChatOpenAI` + `base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`）。

5. **控制台刷屏 `[transformers] Accessing '__path__' ...`**
   → `langchain_core/language_models/base.py` 会 `try: from transformers import GPT2TokenizerFast`，装了 transformers 5.x 就会刷警告。在 `app.py` **最顶部**（早于任何 langchain 导入）加
   `os.environ["TRANSFORMERS_VERBOSITY"] = "error"` 即可静音，或直接卸载 transformers。

6. **知识库明明有文档，检索却总为空**
   → `md5.text` 与 `chroma_db` 不同步：向量库被清空后，MD5 记录还把所有文件判为「已入库」而跳过。
   现在 `ensure_loaded()` 会自动发现「库为空 + 有 MD5 记录」并重置记录、重新灌库；
   若仍异常，可手动删除 `md5.text` 后重启应用。

7. **回答不是逐字出现，而是整段蹦出**
   → 两个条件缺一不可：`ChatTongyi(..., streaming=True)`，且 Agent 用 `stream_mode="messages"`。
   `values` 模式只在节点结束时给完整消息，天然不是 token 级。

8. **回复里混入“我需要先检索…”这类中间思考**
   → 模型在调用工具前会先输出一段思考文本。`execute_stream` 通过 `tool_call_chunks` 判断当前轮要调工具，
   发出 `reset` 事件让前端撤回已渲染文本，并屏蔽该轮后续内容；只有非工具轮的文本才作为正式回答。

9. **侧边栏统计与实际对话差一轮**
   → 不要在“生成回答的同一次 rerun”里改历史。先记录用户消息并 `st.rerun()`，下一轮再生成回答，
   统计与消息列表就都准确了。

---

## 七、可以继续练习的方向

- 给 `rag_summarize` 加 rerank / 多 query 检索，观察对答质量的影响
- 把模拟工具（天气、定位、用户 ID）换成真实 API
- 加 `SqliteSaver` 做多轮会话记忆（LangGraph checkpointer）
- 把 `fetch_external_data` 的返回值统一成字符串（当前声明 `-> str` 实际返回 dict）
- 把工具调用过程可视化（在 `st.status` 里逐条列出工具名/入参/耗时），方便演示 Agent 的思考链路
- 给 `chroma.yml` 的 `k` 做成侧边栏可调项，配合检索分数对比不同 top-k 的效果
