# 新增功能：
# 1. 更多工具：web_search (用 requests 模拟简单搜索，实际用 duckduckgo_search 包)
#    - 注意：需 pip install duckduckgo-search
# 2. 简单向量记忆召回：用 sentence-transformers + faiss 做长期记忆召回
#    - 注意：需 pip install sentence-transformers faiss-cpu
#    - 记忆现在分短期 (list) + 长期 (向量DB)
# 3. 支持 /commands 命令列表
# 4. 错误处理更健壮 + 日志记录
# 5. 可配置温度 + max_tokens
# 用法：python AI_agent.py

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# 新增依赖（pip install duckduckgo-search sentence-transformers faiss-cpu）
try:
    from duckduckgo_search import DDGS
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except ImportError as e:
    print(f"缺少依赖：{e} 请 pip install duckduckgo-search sentence-transformers faiss-cpu")
    exit(1)

load_dotenv()

# ==================== 配置 ====================
MODEL = "gpt-4o-mini"                  # 或其他 OpenAI 兼容模型
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = None

client = OpenAI(api_key=API_KEY or "dummy", base_url=BASE_URL)

MEMORY_FILE = "claw_lite_memory.json"  # 短期记忆
VECTOR_DB_FILE = "claw_lite_vector_db.index"  # 长期向量DB
VECTOR_METADATA_FILE = "claw_lite_vector_metadata.json"  # 元数据

MAX_SHORT_MEMORY = 30                  # 短期记忆消息数
VECTOR_DIM = 384                       # sentence-transformers 模型维度 (all-MiniLM-L6-v2)
MAX_LONG_MEMORY_RECALL = 5             # 每次召回多少长期记忆

TEMPERATURE = 0.65
MAX_TOKENS = 2000

# 设置日志
logging.basicConfig(filename='claw_lite.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# ==================== 向量记忆管理 ====================
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_vector_db():
    if os.path.exists(VECTOR_DB_FILE):
        index = faiss.read_index(VECTOR_DB_FILE)
    else:
        index = faiss.IndexFlatL2(VECTOR_DIM)
    if os.path.exists(VECTOR_METADATA_FILE):
        with open(VECTOR_METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []
    return index, metadata

def save_vector_db(index, metadata):
    faiss.write_index(index, VECTOR_DB_FILE)
    with open(VECTOR_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

index, vector_metadata = load_vector_db()

def add_to_long_memory(message: Dict[str, Any]):
    content = message.get("content", "")
    if not content:
        return
    embedding = embedder.encode([content])[0]
    index.add(np.array([embedding]))
    vector_metadata.append(message)
    save_vector_db(index, vector_metadata)
    logging.info(f"添加长期记忆: {content[:50]}...")

def recall_long_memory(query: str, k: int = MAX_LONG_MEMORY_RECALL) -> List[Dict[str, Any]]:
    if index.ntotal == 0:
        return []
    query_emb = embedder.encode([query])[0]
    _, indices = index.search(np.array([query_emb]), k)
    recalled = [vector_metadata[i] for i in indices[0] if i < len(vector_metadata)]
    logging.info(f"召回 {len(recalled)} 条长期记忆 for query: {query[:50]}")
    return recalled

# ==================== 短期记忆管理 ====================
def load_short_memory() -> List[Dict[str, Any]]:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data[-MAX_SHORT_MEMORY:]
    except:
        return []

def save_short_memory(messages: List[Dict[str, Any]]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages[-MAX_SHORT_MEMORY * 2:], f, ensure_ascii=False, indent=2)

# ==================== 工具 / Skill 定义 ====================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前本地时间，常用于回答'现在几点'、'今天是星期几'等",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "执行基础数学计算，支持 + - * / ^ () 等，例如 '2*(3+4)^2'",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "数学表达式字符串"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "保存一条简短笔记/待办/想法到本地文件，方便以后回忆",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "要保存的内容"}},
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "用 DuckDuckGo 搜索网络，返回前 N 条结果的标题+摘要+链接",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "num_results": {"type": "integer", "description": "结果数，默认3", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    # 可以继续加更多...
]

def get_current_time(_=None) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S 周%w %Z")

def simple_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"结果：{result}"
    except Exception as e:
        return f"计算失败：{str(e)}"

def save_note(content: str) -> str:
    note_file = "claw_notes.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(note_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {content}\n")
    return f"已保存笔记到 {note_file}"

def web_search(query: str, num_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        formatted = "\n".join([f"{r['title']}: {r['body']} ({r['href']})" for r in results])
        return formatted or "无结果"
    except Exception as e:
        return f"搜索失败：{str(e)}"

TOOL_EXECUTORS = {
    "get_current_time": get_current_time,
    "simple_calculator": simple_calculator,
    "save_note": save_note,
    "web_search": web_search,
}

# ==================== System Prompt（核心人格） ====================
SYSTEM_PROMPT = """你现在是 Claw-Lite — 一个极简、务实、有点直男的个人 AI 助手。
核心风格：
- 说话直接、少废话、带点幽默
- 能用工具就用工具，别自己猜答案
- 事情尽量一次做完，不要反复问
- 如果需求不清晰，只问最关键的一个问题
- 长期记忆召回：我会提供相关历史，如果你觉得有用就参考

可用工具：get_current_time, simple_calculator, save_note, web_search

记住：你是用户的“数字分身”，目标是帮他省时间、减负担，而不是闲聊。
"""

# ==================== 命令帮助 ====================
COMMANDS = {
    "/quit": "退出程序",
    "/exit": "同上",
    "/clear": "清空短期记忆（长期记忆保留）",
    "/commands": "显示这个命令列表",
    "/recall": "手动召回长期记忆（需跟查询，如 /recall 我的计划）",
    "/temp": "设置温度，如 /temp 0.8",
    "/maxtokens": "设置 max_tokens，如 /maxtokens 1500",
}

def print_commands():
    print("\n可用命令：")
    for cmd, desc in COMMANDS.items():
        print(f"{cmd}: {desc}")

# ==================== 主程序 ====================
def run_claw_lite():
    short_messages = load_short_memory()
    print("Claw-Lite 已启动（轻量风格 + 扩展）")
    print("新增：web_search 工具 + 向量长期记忆 + 更多命令")
    print_commands()
    print("-" * 60)

    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue

        # 处理命令
        if user_input.lower() in {"/quit", "/exit", "quit", "exit", "q"}:
            print("再见～")
            break

        if user_input.lower() == "/clear":
            short_messages = []
            save_short_memory(short_messages)
            print("短期记忆已清空。")
            continue

        if user_input.lower() == "/commands":
            print_commands()
            continue

        if user_input.lower().startswith("/temp "):
            try:
                global TEMPERATURE
                TEMPERATURE = float(user_input.split()[1])
                print(f"温度设为 {TEMPERATURE}")
            except:
                print("无效：/temp <float>")
            continue

        if user_input.lower().startswith("/maxtokens "):
            try:
                global MAX_TOKENS
                MAX_TOKENS = int(user_input.split()[1])
                print(f"max_tokens 设为 {MAX_TOKENS}")
            except:
                print("无效：/maxtokens <int>")
            continue

        if user_input.lower().startswith("/recall "):
            query = user_input[8:].strip()
            recalled = recall_long_memory(query)
            print("\n召回长期记忆：")
            for msg in recalled:
                print(f"- [{msg['role']}] {msg['content'][:100]}...")
            continue

        # 正常消息：添加短期 + 召回长期
        short_messages.append({"role": "user", "content": user_input})
        recalled_long = recall_long_memory(user_input)
        recalled_str = "\n".join([f"历史: [{m['role']}] {m['content']}" for m in recalled_long])
        if recalled_str:
            user_input_with_recall = f"{user_input}\n\n相关历史记忆：\n{recalled_str}"

        # 构建上下文（system + 短期 + 召回长期）
        context = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *short_messages
        ]
        if recalled_long:
            context.insert(1, {"role": "system", "content": f"相关长期记忆（参考用）：\n{recalled_str}"})

        while True:  # ReAct 循环
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=context,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
            except Exception as e:
                logging.error(f"模型调用失败：{e}")
                print(f"模型调用失败：{e}")
                time.sleep(2)
                continue

            choice = resp.choices[0]
            msg = choice.message

            if msg.tool_calls:
                short_messages.append(msg.model_dump(exclude_none=True))

                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                        if func_name in TOOL_EXECUTORS:
                            result = TOOL_EXECUTORS[func_name](**args)
                        else:
                            result = f"未知工具：{func_name}"
                    except Exception as e:
                        result = f"工具执行出错：{str(e)}"
                        logging.error(result)

                    short_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": str(result)
                    })

                context = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *short_messages
                ]
                if recalled_long:
                    context.insert(1, {"role": "system", "content": f"相关长期记忆：\n{recalled_str}"})
                continue

            # 最终回答
            reply = msg.content or "(无回复内容)"
            print("\nClaw:", reply)
            short_messages.append({"role": "assistant", "content": reply})
            # 每轮结束添加长期记忆
            add_to_long_memory({"role": "user", "content": user_input})
            add_to_long_memory({"role": "assistant", "content": reply})
            break

        save_short_memory(short_messages)


if __name__ == "__main__":
    print("Claw-Lite v2026.04   （轻量 风格 + 扩展）")
    print("支持 web_search + 向量长期记忆 + ReAct + 更多命令")
    run_claw_lite()