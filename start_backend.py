"""启动脚本"""
import subprocess
import os
import sys

# 设置工作目录
work_dir = r"F:\DC25\part1\NeuroBreak-Reproduction\visualization"
os.chdir(work_dir)

# 安装依赖并启动后端
print("正在启动后端 API 服务器 (端口 5000)...")
subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
)

print("后端 API 已启动在 http://localhost:5000")
print("如需停止服务器，请关闭命令行窗口")
