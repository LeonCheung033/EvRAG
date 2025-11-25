#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 split_docs 中文本长度分布，找出超过 512 字符的文档
"""

import sys
import pickle
from pathlib import Path
from collections import Counter

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def check_text_length():
    split_docs_path = Path("/remote-home/share/liangZhang/EvRAG/data/processed_docs/split_docs.pkl")
    
    print("=" * 80)
    print("检查文档文本长度分布")
    print("=" * 80)
    
    if not split_docs_path.exists():
        print(f"❌ 文件不存在: {split_docs_path}")
        return
    
    with open(split_docs_path, 'rb') as f:
        split_docs = pickle.load(f)
    
    print(f"\n总文档数: {len(split_docs)}")
    
    # 统计长度
    lengths = []
    over_512 = []
    over_1024 = []
    over_2048 = []
    
    for i, doc in enumerate(split_docs):
        if hasattr(doc, 'page_content'):
            text = doc.page_content
        elif isinstance(doc, dict):
            text = doc.get('page_content', '')
        else:
            continue
        
        length = len(text)
        lengths.append(length)
        
        if length > 512:
            over_512.append((i, length, text[:100]))
        if length > 1024:
            over_1024.append((i, length, text[:100]))
        if length > 2048:
            over_2048.append((i, length, text[:100]))
    
    print(f"\n【长度统计】")
    if lengths:
        print(f"最小长度: {min(lengths)}")
        print(f"最大长度: {max(lengths)}")
        print(f"平均长度: {sum(lengths) / len(lengths):.1f}")
        print(f"中位数长度: {sorted(lengths)[len(lengths) // 2]}")
    
    print(f"\n【超过限制的文档数】")
    print(f"超过 512 字符: {len(over_512)} 个")
    print(f"超过 1024 字符: {len(over_1024)} 个")
    print(f"超过 2048 字符: {len(over_2048)} 个")
    
    if over_512:
        print(f"\n【超过 512 字符的文档示例（前5个）】")
        for i, (idx, length, preview) in enumerate(over_512[:5], 1):
            print(f"  {i}. 文档索引 {idx}: 长度={length}, 预览={preview}...")
    
    # 长度分布
    print(f"\n【长度分布】")
    length_ranges = {
        "0-256": 0,
        "257-512": 0,
        "513-1024": 0,
        "1025-2048": 0,
        "2049+": 0,
    }
    
    for length in lengths:
        if length <= 256:
            length_ranges["0-256"] += 1
        elif length <= 512:
            length_ranges["257-512"] += 1
        elif length <= 1024:
            length_ranges["513-1024"] += 1
        elif length <= 2048:
            length_ranges["1025-2048"] += 1
        else:
            length_ranges["2049+"] += 1
    
    for range_name, count in length_ranges.items():
        percentage = count / len(lengths) * 100 if lengths else 0
        print(f"  {range_name}: {count} 个 ({percentage:.1f}%)")
    
    print("\n" + "=" * 80)
    
    # 建议
    if over_512:
        max_length = max(lengths)
        recommended_max = max(2048, (max_length // 512 + 1) * 512)  # 向上取整到512的倍数
        print(f"\n【建议】")
        print(f"当前 MAX_TEXT_LENGTH = 512")
        print(f"文档最大长度 = {max_length}")
        print(f"建议将 MAX_TEXT_LENGTH 设置为: {recommended_max} 或更大")
        print(f"或者考虑在插入前截断过长的文本")

if __name__ == "__main__":
    check_text_length()

