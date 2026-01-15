import argparse
import re
import os

def modify_data_params(file_path, samples_new_value, workers_new_value):
    """
    修改文件中 data = dict(...) 内的 samples_per_gpu 和 workers_per_gpu 键值
    :param file_path: 目标文件路径
    :param samples_new_value: 新的 samples_per_gpu 数值（整数）
    :param workers_new_value: 新的 workers_per_gpu 数值（整数）
    """
    # 校验文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    # 校验数值类型
    try:
        samples_new_value = int(samples_new_value)
        workers_new_value = int(workers_new_value)
    except ValueError:
        raise TypeError("samples_per_gpu 和 workers_per_gpu 必须是整数")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ========== 修复核心：用 \g<1> 替代 \1，避免组引用与数字拼接冲突 ==========
    # 1. 修改 samples_per_gpu（\g<1> 明确指定第1个捕获组）
    samples_pattern = r'(samples_per_gpu=)[^,\n]+'
    new_content = re.sub(
        samples_pattern,
        r'\g<1>' + str(samples_new_value),  # 拆分组引用和数字，避免解析错误
        content
    )
    
    # 2. 修改 workers_per_gpu
    workers_pattern = r'(workers_per_gpu=)[^,\n]+'
    new_content = re.sub(
        workers_pattern,
        r'\g<1>' + str(workers_new_value),
        new_content
    )
    
    # 写回修改后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"修改完成！")
    print(f"文件路径：{file_path}")
    print(f"samples_per_gpu 已设置为：{samples_new_value}")
    print(f"workers_per_gpu 已设置为：{workers_new_value}")

if __name__ == "__main__":
    # 解析命令行参数（强制要求指定 samples 和 workers）
    parser = argparse.ArgumentParser(description='修改文件中 samples_per_gpu 和 workers_per_gpu 的值（两个参数都必须指定）')
    parser.add_argument('file_path', type=str, help='目标文件的路径（绝对路径/相对路径）')
    parser.add_argument('samples', type=str, help='新的 samples_per_gpu 数值（整数）')
    parser.add_argument('workers', type=str, help='新的 workers_per_gpu 数值（整数）')
    
    args = parser.parse_args()
    
    # 执行修改
    try:
        modify_data_params(
            file_path=args.file_path,
            samples_new_value=args.samples,
            workers_new_value=args.workers
        )
    except Exception as e:
        print(f"修改失败：{e}")
        exit(1)