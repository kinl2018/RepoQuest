import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from repo_tools import get_callers, find_symbol, get_dependencies,  read_source, search_code, list_files

load_dotenv()

ROOT = Path(__file__).resolve().parent
MODEL = os.environ["LLM_MODEL"]
client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    timeout=30.0
)

# 第一步：向 LLM 描述它能够使用什么工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_callers",
            "description": "查询仓库中哪些函数调用了指定的 Python 函数",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_file": {
                        "type": "string",
                        "description": "目标函数所在文件的相对路径，例如 ast_parser.py"
                    },
                    "target_function": {
                        "type": "string",
                        "description": "目标函数名称，例如 parse_file"
                    }
                },
                "required": ["target_file", "target_function"],
                "additionalProperties": False
            },
            #"strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": (
                "根据函数名、类名或方法名搜索代码定义，"
                "返回文件路径、行号和源码。"
                "当用户询问函数在哪里定义、源码是什么时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "要搜索的符号名称，"
                            "例如 parse_file 或 Greeter.hello"
                        )
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            },
            #"strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dependencies",
            "description": (
                "查询指定 Python 文件依赖了仓库内的哪些本地模块。"
                "当用户询问文件导入关系、模块依赖时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Python 文件的相对路径，例如 ui.py"
                    }
                },
                "required": ["name"],
                "additionalProperties": False
            },
            #"strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": (
                "读取仓库内指定 Python 文件的部分真实源码，"
                "可用于理解代码实现、解释具体语句。"
                "按行读取 Python 源码，每次最多返回80行。"
                "如果 truncated 为 true，可从 next_start_line 继续读取。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Python 文件相对路径，例如 ui.py"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "起始行号，从1开始"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "结束行号，包含该行，最多读取80行"
                    }
                },
                "required": ["file", "start_line", "end_line"],
                "additionalProperties": False
            },
            #"strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "在整个 Python 仓库中搜索包含指定关键词的代码行。"
                "当用户不知道文件名或函数名，只描述想寻找的功能时，"
                "可以先搜索相关的 API、函数调用或代码关键词来定位文件。"
                "例如寻找 LLM 调用代码，可以尝试搜索 "
                "OpenAI( 或 chat.completions.create 等关键词。"
                "这是字面匹配，不是语义搜索。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "要在源码中搜索的字符串关键词"
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "列出当前仓库中的 Python 文件路径。"
                "当用户提供的文件名可能拼错、文件路径不存在，"
                "或者不知道某个功能位于哪个文件时使用。"
                "返回文件总数、最多100个文件路径以及是否截断。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    }
]

def ask_repo(question: str, root: Path, history: list[dict] | None = None, max_rounds=5) -> tuple[str, list[dict]]:
    trace = []
    root = root.expanduser().resolve()

    if not root.is_dir():
        raise ValueError(f"不是有效的目录: {root}")

    messages = [
        {
            "role": "system",
            "content": (
                "你是 RepoQuest 代码分析助手。"
                "需要查询代码时，请使用提供的工具。"
                "用户可能会引用前面讨论的函数或文件。"
                "不要编造文件位置和调用关系。"
                "如果用户提供的文件名不存在，"
                "应先使用 list_files 核实真实路径，"
                "必要时再使用 find_symbol 或 search_code 定位源码。"
                "不要凭空编造文件名。"
            )
        }
    ]

    # 加入之前的对话，最多保留最近6条消息
    if history:
        messages.extend(history[-6:])

    # 当前用户的新问题
    messages.append({
        "role": "user",
        "content": question
    })

    for step in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        calls = message.tool_calls or []

        if not calls:
            return message.content or "", trace

        for call in calls:
            name = call.function.name
            args = json.loads(call.function.arguments)

            #print("LLM 请求调用：", name)
            #print("工具参数：", args)

            try:
                if name == "find_symbol":
                    if set(args) != {"name"} or not isinstance(args["name"], str):
                        raise ValueError("find_symbol 参数不正确")
                    result = find_symbol(root, args["name"])

                elif name == "get_callers":
                    if set(args) != {"target_file", "target_function"}:
                        raise ValueError("get_callers 参数不正确")
                    if not all(isinstance(v, str) for v in args.values()):
                        raise ValueError("get_callers 参数类型不正确")

                    result = get_callers(
                        root,
                        args["target_file"],
                        args["target_function"]
                    )

                elif name == "get_dependencies":
                    if set(args) != {"name"} or not isinstance(args["name"], str):
                        raise ValueError("get_dependencies 参数不正确")

                    result = get_dependencies(root, args["name"])

                elif name == "read_source":
                    if set(args) != {"file", "start_line", "end_line"}:
                        raise ValueError("read_source 参数不正确")

                    if not isinstance(args["file"], str):
                        raise ValueError("file 必须是字符串")

                    if type(args["start_line"]) is not int or type(args["end_line"]) is not int:
                        raise ValueError("行号必须是整数")

                    result = read_source(
                        root,
                        args["file"],
                        args["start_line"],
                        args["end_line"]
                    )

                elif name == "search_code":
                    if set(args) != {"keyword"}:
                        raise ValueError("search_code 参数不正确")

                    if not isinstance(args["keyword"], str):
                        raise ValueError("keyword 必须是字符串")

                    result = search_code(root, args["keyword"])

                elif name == "list_files":
                    if args != {}:
                        raise ValueError("list_files 不需要额外参数")

                    result = list_files(root)

                else:
                    raise ValueError(f"不支持的工具：{name}")

            except (ValueError, OSError, UnicodeError, SyntaxError) as e:
                result = {"error": f"{type(e).__name__}: {e}", "hint": "请检查文件路径或行号，必要时重新搜索并调用工具。"}

            trace.append({"tool": name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

    return (f"本次代码探索已达到 {max_rounds} 轮调用上限，尚未生成完整回答。\n已成功执行 {len(trace)} 次工具调用，"
        f"可以展开下方的工具执行记录查看已经找到的信息。\n建议缩小问题范围或分步骤继续提问。", trace)

