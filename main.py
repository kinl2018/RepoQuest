import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    env_file = root / ".env"
    ui_file = root / "ui.py"
    if not env_file.is_file():
        print("[错误] 未找到 .env 配置文件，请在项目根目录创建 .env 并填写 LLM 配置。")
        sys.exit(1)
    ''' 感觉llm非必需遂注释
    load_dotenv(env_file)
    missing = []
    for key in ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]:
        value = os.getenv(key, "").strip()
        if not value or value in ["你的api key", "你的api网址", "所选择的模型"]:
            missing.append(key)
    if missing:
        print("[错误] 以下配置尚未填写：")
        for key in missing:
            print(f"  - {key}")
        print("请检查 .env 文件后重新启动。")
        sys.exit(1)
    '''
    if not ui_file.is_file():
        print("[错误] 未找到 ui.py！")
        sys.exit(1)
    print("[成功] 环境配置检查通过，正在启动 RepoQuest...")
    print("访问地址：http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_file)], cwd=root)