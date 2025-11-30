#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM模型对比评估脚本

对比基线LLM和微调后的LLM模型，计算以下指标：
- EM (Exact Match)
- F1
- BLEU-1
- ROUGE-L
- BERTScore
- Avg Length

注意事项：
1. 数据路径：/remote-home/share/liangZhang/EvRAG/data/summary_data/test.json
2. 所有数据都是中文，使用jieba进行分词
3. 模型名称在每次请求时显式传递，确保使用正确的模型
4. 基线LLM服务：http://localhost:8000/v1
5. 微调LLM服务：http://localhost:8001/v1
"""

import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from evrag.client import LocalLLMClient

# BERTScore
try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("⚠ 警告: bert_score未安装，将跳过BERTScore计算")


def calculate_rouge_l(predicted: str, expected: str, jieba) -> float:
    """计算ROUGE-L分数（最长公共子序列）"""
    if not predicted or not expected:
        return 0.0
    
    try:
        # 使用jieba分词
        pred_tokens = list(jieba.cut(predicted, cut_all=False))
        exp_tokens = list(jieba.cut(expected, cut_all=False))
        
        if not pred_tokens or not exp_tokens:
            return 0.0
        
        # 计算最长公共子序列（LCS）
        m, n = len(pred_tokens), len(exp_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == exp_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # 计算precision和recall
        precision = lcs_length / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = lcs_length / len(exp_tokens) if len(exp_tokens) > 0 else 0.0
        
        # 计算F-measure
        if precision + recall == 0:
            return 0.0
        rouge_l = 2 * precision * recall / (precision + recall)
        
        return rouge_l
    except Exception:
        return 0.0


def extract_citation(text: str) -> Tuple[str, List[int]]:
    """提取答案和引用标记"""
    # 匹配【引用编号】格式
    citation_pattern = r'【(\d+(?:,\s*\d+)*)】'
    match = re.search(citation_pattern, text)
    
    if match:
        citation_str = match.group(1)
        citations = [int(x.strip()) for x in citation_str.split(',')]
        # 移除引用标记，只保留答案内容
        answer = re.sub(citation_pattern, '', text).strip()
    else:
        citations = []
        answer = text.strip()
    
    return answer, citations


def calculate_exact_match(predicted: str, expected: str) -> bool:
    """计算Exact Match"""
    # 移除引用标记后比较
    pred_clean, _ = extract_citation(predicted)
    exp_clean, _ = extract_citation(expected)
    
    # 转换为小写并去除空白字符后比较
    pred_normalized = re.sub(r'\s+', '', pred_clean.lower())
    exp_normalized = re.sub(r'\s+', '', exp_clean.lower())
    
    return pred_normalized == exp_normalized


def calculate_f1(predicted: str, expected: str) -> float:
    """计算F1分数（基于字符级别）"""
    pred_clean, _ = extract_citation(predicted)
    exp_clean, _ = extract_citation(expected)
    
    if not pred_clean or not exp_clean:
        return 0.0
    
    # 使用字符级别计算F1
    pred_chars = set(pred_clean)
    exp_chars = set(exp_clean)
    
    if not pred_chars or not exp_chars:
        return 0.0
    
    common_chars = pred_chars & exp_chars
    precision = len(common_chars) / len(pred_chars) if pred_chars else 0.0
    recall = len(common_chars) / len(exp_chars) if exp_chars else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1




def batch_chat_vllm(
    llm_client: LocalLLMClient,
    messages_list: List[List[Dict[str, str]]],
    batch_size: int = 8,
) -> List[str]:
    """
    使用vLLM的批量API进行批量推理（并发处理）
    
    Args:
        llm_client: LLM客户端
        messages_list: 消息列表的列表，每个元素是一个对话的消息列表
        batch_size: 批量大小
        
    Returns:
        生成的文本列表（保持输入顺序）
    """
    from openai import OpenAI
    import concurrent.futures
    
    # 获取vLLM客户端的基础URL和API key
    base_url = llm_client.base_url
    api_key = llm_client.api_key
    model = llm_client.model
    
    # 创建OpenAI客户端用于批量请求
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    def single_request(request):
        """单个请求函数"""
        try:
            completion = client.chat.completions.create(**request)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"    ⚠️  请求失败: {e}")
            return ""  # 失败时返回空字符串
    
    # 准备批量请求
    batch_requests = []
    for messages in messages_list:
        request = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": 20,
            }
        }
        batch_requests.append(request)
    
    # 使用线程池并发执行（vLLM服务器端会处理批量优化）
    # 使用字典来保持顺序
    responses = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch_requests), batch_size * 2)) as executor:
        # 提交所有请求，并记录索引
        future_to_idx = {
            executor.submit(single_request, req): idx 
            for idx, req in enumerate(batch_requests)
        }
        
        # 收集结果，保持顺序
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                response = future.result()
                responses[idx] = response
            except Exception as e:
                print(f"    ⚠️  请求失败 (索引 {idx}): {e}")
                responses[idx] = ""  # 失败时返回空字符串
    
    # 按索引顺序返回结果
    return [responses[i] for i in range(len(batch_requests))]


def evaluate_llm(
    llm_client: LocalLLMClient,
    test_data: List[Dict],
    model_name: str = "LLM",
    batch_size: int = 8
) -> Dict[str, float]:
    """评估LLM模型（使用批量并发处理）"""
    print(f"\n评估 {model_name}...")
    
    # 导入jieba用于中文分词
    import jieba
    
    # 准备所有样本
    valid_items = []
    for item in test_data:
        query = item.get("query", "")
        instruction = item.get("instruction", "")
        if query and instruction:
            valid_items.append(item)
    
    total = len(valid_items)
    print(f"  有效样本数: {total}")
    
    # 批量生成答案
    print(f"  使用批量并发处理（batch_size={batch_size}）...")
    all_messages = []
    all_items = []
    
    for item in valid_items:
        instruction = item.get("instruction", "")
        messages = [{"role": "user", "content": instruction}]
        all_messages.append(messages)
        all_items.append(item)
    
    # 批量调用LLM（并发处理）
    all_predictions = []
    for i in tqdm(range(0, len(all_messages), batch_size), desc=f"批量生成{model_name}"):
        batch_messages = all_messages[i:i+batch_size]
        batch_predictions = batch_chat_vllm(llm_client, batch_messages, batch_size=batch_size)
        all_predictions.extend(batch_predictions)
    
    # 计算指标
    print(f"  计算评估指标...")
    em_scores = []
    f1_scores = []
    bleu1_scores = []
    rouge_l_scores = []
    bertscore_scores = []
    lengths = []
    
    # 收集所有预测和参考，用于批量计算BERTScore
    bertscore_predictions = []
    bertscore_references = []
    
    for idx, (item, predicted_output) in enumerate(tqdm(zip(all_items, all_predictions), desc=f"计算指标{model_name}", total=len(all_items))):
        expected_output = item.get("output", "")
        context = item.get("context", "")
        
        if not predicted_output:
            # 如果生成失败，跳过
            continue
        
        try:
            
            # 计算指标
            # 1. Exact Match
            em = calculate_exact_match(predicted_output, expected_output)
            em_scores.append(em)
            
            # 2. F1
            f1 = calculate_f1(predicted_output, expected_output)
            f1_scores.append(f1)
            
            # 3. BLEU-1 (只计算1-gram的BLEU)
            # 使用jieba分词后计算1-gram重叠
            pred_tokens = list(jieba.cut(predicted_output, cut_all=False))
            exp_tokens = list(jieba.cut(expected_output, cut_all=False))
            
            if pred_tokens and exp_tokens:
                pred_unigrams = set(pred_tokens)
                exp_unigrams = set(exp_tokens)
                common = pred_unigrams & exp_unigrams
                if len(exp_unigrams) > 0:
                    bleu1 = len(common) / len(exp_unigrams)
                else:
                    bleu1 = 0.0
            else:
                bleu1 = 0.0
            bleu1_scores.append(bleu1)
            
            # 4. ROUGE-L (使用jieba分词后手动计算)
            rouge_l = calculate_rouge_l(predicted_output, expected_output, jieba)
            rouge_l_scores.append(rouge_l)
            
            # 5. BERTScore（批量计算）- 只添加成功处理的样本
            bertscore_predictions.append(predicted_output)
            bertscore_references.append(expected_output)
            
            # 6. Length
            pred_clean, _ = extract_citation(predicted_output)
            lengths.append(len(pred_clean))
            
        except Exception as e:
            print(f"\n  警告: 处理样本 {idx} 失败: {e}")
            continue
    
            # 批量计算BERTScore
    # 确保预测和参考数量一致（只计算成功处理的样本）
    if BERTSCORE_AVAILABLE and bertscore_predictions and bertscore_references:
        if len(bertscore_predictions) == len(bertscore_references):
            try:
                print(f"  计算BERTScore（{len(bertscore_predictions)}个样本）...")
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                P, R, F1 = bert_score_func(
                    bertscore_predictions,
                    bertscore_references,
                    lang='zh',
                    verbose=False,
                    device=device
                )
                bertscore_scores = F1.tolist()
            except Exception as e:
                print(f"  警告: BERTScore计算失败: {e}")
                bertscore_scores = [0.0] * len(bertscore_predictions)
        else:
            print(f"  警告: 预测数量({len(bertscore_predictions)})与参考数量({len(bertscore_references)})不一致，跳过BERTScore计算")
            bertscore_scores = [0.0] * len(bertscore_predictions)
    else:
        bertscore_scores = [0.0] * len(bertscore_predictions) if bertscore_predictions else []
    
    # 计算平均值
    results = {
        "total": total,
        "em": np.mean(em_scores) if em_scores else 0.0,
        "f1": np.mean(f1_scores) if f1_scores else 0.0,
        "bleu1": np.mean(bleu1_scores) if bleu1_scores else 0.0,
        "rouge_l": np.mean(rouge_l_scores) if rouge_l_scores else 0.0,
        "bertscore": np.mean(bertscore_scores) if bertscore_scores else 0.0,
        "avg_length": np.mean(lengths) if lengths else 0.0,
    }
    
    print(f"  Total: {total}")
    print(f"  EM: {results['em']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    print(f"  BLEU-1: {results['bleu1']:.4f}")
    print(f"  ROUGE-L: {results['rouge_l']:.4f}")
    print(f"  BERTScore: {results['bertscore']:.4f}")
    print(f"  Avg Length: {results['avg_length']:.1f}")
    
    return results


def main():
    """主函数"""
    print("=" * 80)
    print("LLM模型对比评估")
    print("=" * 80)
    
    # 加载测试数据
    test_data_path = Path("/remote-home/share/liangZhang/EvRAG/data/summary_data/test.json")
    print(f"\n加载测试数据: {test_data_path}")
    
    if not test_data_path.exists():
        print(f"✗ 测试数据文件不存在: {test_data_path}")
        return
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"测试数据量: {len(test_data)} 条")
    
    # 初始化LLM客户端
    # 基线LLM: 8000端口
    # 注意：model参数会在每次请求时显式传递，确保使用正确的模型
    baseline_client = LocalLLMClient(
        base_url="http://localhost:8000/v1",
        model="Qwen3-8B"  # 默认模型名称，但会在请求时显式传递
    )
    
    # 微调LLM: 8001端口
    finetuned_client = LocalLLMClient(
        base_url="http://localhost:8001/v1",
        model="qwen3_lora_sft"  # 默认模型名称，但会在请求时显式传递
    )
    
    # 评估结果
    all_results = {}
    
    # 并发评估两个模型
    print("\n" + "=" * 80)
    print("并发评估两个LLM模型")
    print("=" * 80)
    
    import concurrent.futures
    
    def evaluate_baseline():
        try:
            return ("Baseline Qwen", evaluate_llm(
                baseline_client, test_data, "Baseline Qwen", batch_size=8
            ))
        except Exception as e:
            print(f"✗ 基线LLM评估失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_finetuned():
        try:
            return ("Fine-tuned Qwen", evaluate_llm(
                finetuned_client, test_data, "Fine-tuned Qwen", batch_size=8
            ))
        except Exception as e:
            print(f"✗ 微调LLM评估失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # 使用线程池并发执行两个评估任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_baseline = executor.submit(evaluate_baseline)
        future_finetuned = executor.submit(evaluate_finetuned)
        
        # 等待结果
        result_baseline = future_baseline.result()
        result_finetuned = future_finetuned.result()
        
        if result_baseline:
            name, results = result_baseline
            all_results[name] = results
        
        if result_finetuned:
            name, results = result_finetuned
            all_results[name] = results
    
    # 输出对比结果
    print("\n" + "=" * 80)
    print("评估结果对比")
    print("=" * 80)
    
    if len(all_results) == 2:
        baseline = all_results["Baseline Qwen"]
        finetuned = all_results["Fine-tuned Qwen"]
        
        print(f"\n| {'Model':<20} | {'EM':>8} | {'F1':>8} | {'BLEU-1':>10} | {'ROUGE-L':>10} | {'BERTScore':>12} | {'Avg Length':>12} |")
        print("|" + "-" * 20 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 13 + "|" + "-" * 13 + "|")
        print(f"| {'Baseline Qwen':<20} | {baseline['em']:>8.2f} | {baseline['f1']:>8.2f} | {baseline['bleu1']:>10.2f} | {baseline['rouge_l']:>10.2f} | {baseline['bertscore']:>12.2f} | {baseline['avg_length']:>12.0f} |")
        # 使用格式化字符串确保数字正确对齐
        em_str = f"**{finetuned['em']:.2f}**"
        f1_str = f"**{finetuned['f1']:.2f}**"
        bleu1_str = f"**{finetuned['bleu1']:.2f}**"
        rouge_l_str = f"**{finetuned['rouge_l']:.2f}**"
        bertscore_str = f"**{finetuned['bertscore']:.2f}**"
        print(f"| {'Fine-tuned Qwen':<20} | {em_str:>8} | {f1_str:>8} | {bleu1_str:>10} | {rouge_l_str:>10} | {bertscore_str:>12} | {finetuned['avg_length']:>12.0f} |")
        
        # 保存结果
        output_path = project_root / "rag_test_reports" / "llm_comparison_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 结果已保存到: {output_path}")
        
        # 生成Markdown报告
        report_path = project_root / "reports" / "evaluation" / "llm" / "llm_comparison_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# LLM模型对比评估报告\n\n")
            f.write("## 评估结果\n\n")
            f.write("| Model | EM | F1 | BLEU-1 | ROUGE-L | BERTScore | Avg Length |\n")
            f.write("|-------|----|----|--------|---------|-----------|------------|\n")
            f.write(f"| Baseline Qwen | {baseline['em']:.2f} | {baseline['f1']:.2f} | {baseline['bleu1']:.2f} | {baseline['rouge_l']:.2f} | {baseline['bertscore']:.2f} | {baseline['avg_length']:.0f} |\n")
            f.write(f"| Fine-tuned Qwen | **{finetuned['em']:.2f}** | **{finetuned['f1']:.2f}** | **{finetuned['bleu1']:.2f}** | **{finetuned['rouge_l']:.2f}** | **{finetuned['bertscore']:.2f}** | {finetuned['avg_length']:.0f} |\n")
        print(f"✓ Markdown报告已保存到: {report_path}")
    else:
        print("\n✗ 评估未完成，无法生成对比结果")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

