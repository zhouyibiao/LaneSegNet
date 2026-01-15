import argparse
import os
import re
from datetime import datetime

def extract_and_calculate_time(log_file_path):
    """
    提取指定日志文件中的time数据，计算抛去第一个值后的平均值，并将结果写入文件末尾
    :param log_file_path: 日志文件的完整路径
    """
    # 1. 初始化time数据列表
    time_values = []
    
    # 2. 定义匹配time数据的正则表达式（匹配"time: 数值"格式）
    time_pattern = re.compile(r'time:\s*([\d.]+)')
    
    try:
        # 3. 读取日志文件，提取所有time值
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = time_pattern.search(line)
                if match:
                    time_val = float(match.group(1))
                    time_values.append(time_val)
        
        # 4. 计算平均值（抛去第一个值，需保证列表长度≥2）
        result_info = ""
        if len(time_values) == 0:
            result_info = f"【日志处理时间】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 未提取到任何time数据"
        elif len(time_values) == 1:
            result_info = f"【日志处理时间】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 仅提取到1个time数据（{time_values[0]}），无法计算抛去第一个值后的平均值"
        else:
            # 抛去第一个值，计算剩余数据的平均值
            valid_time_values = time_values[1:]
            avg_time = sum(valid_time_values) / len(valid_time_values)
            result_info = (
                f"【日志处理时间】{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"  - 提取到总共有{len(time_values)}个time数据\n"
                f"  - 抛去第一个值（{time_values[0]}）后，剩余{len(valid_time_values)}个数据\n"
                f"  - 剩余数据的time平均值：{avg_time:.6f}"
            )
        
        # 5. 将结果写入日志文件末尾
        with open(log_file_path, 'a', encoding='utf-8') as f:
            # 写入分隔线和结果，提升可读性
            f.write("\n" + "="*80 + "\n")
            f.write(result_info + "\n")
            f.write("="*80 + "\n\n")
        
        print(f"处理完成：{log_file_path}")
        # 修复：先执行split操作，将结果存入临时变量，再在f-string中引用
        result_summary = result_info.split('\n')[0]
        print(f"处理结果概要：{result_summary}\n")
    
    except Exception as e:
        print(f"处理失败：{log_file_path} - 错误信息：{str(e)}")

def process_target(target_path):
    """
    处理目标（单个.log文件或指定目录及其子目录下所有.log文件）
    :param target_path: 传入的文件路径或目录路径
    """
    # 1. 判断目标是否为文件
    if os.path.isfile(target_path):
        # 检查是否为.log后缀文件
        if target_path.endswith('.log'):
            extract_and_calculate_time(target_path)
        else:
            print(f"错误：指定文件不是.log后缀文件 - {target_path}")
    # 2. 判断目标是否为目录（递归遍历所有子目录）
    elif os.path.isdir(target_path):
        # 递归遍历目录及其所有子目录，筛选.log后缀文件
        for root, dirs, files in os.walk(target_path):
            for filename in files:
                if filename.endswith('.log'):
                    file_full_path = os.path.join(root, filename)
                    extract_and_calculate_time(file_full_path)
        print(f"递归处理完毕：{target_path} 及其所有子目录下的所有.log文件已处理")
    # 3. 目标不存在
    else:
        print(f"错误：指定的路径/文件不存在 - {target_path}")

def main():
    # 1. 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='日志文件time数据提取与平均值计算工具（支持递归遍历子目录）')
    # 2. 添加-p参数，用于接收文件/路径输入
    parser.add_argument('-p', required=True, help='指定日志文件路径或包含.log文件的目录路径（示例：python log_preprocess.py -p path/file.log 或 python log_preprocess.py -p path/）')
    # 3. 解析命令行参数
    args = parser.parse_args()
    
    # 4. 处理传入的目标路径
    target_path = args.p
    process_target(target_path)

if __name__ == "__main__":
    main()