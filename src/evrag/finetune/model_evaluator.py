"""模型评估模块

在验证集上评估微调后的模型，支持基线模型和微调后模型的对比评估
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# vLLM支持
try:
    from ..client.local_client import LocalLLMClient
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# 评估指标相关导入
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # 下载必要的NLTK数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# 中文分词支持
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


class ModelEvaluator:
    """模型评估器"""

    def __init__(
        self,
        model_path: Path,
        model_type: str = "llm",
        base_model_path: Optional[Path] = None,
        is_lora: bool = False,
    ):
        """
        初始化模型评估器

        Args:
            model_path: 模型路径（对于LoRA，这是adapter路径）
            model_type: 模型类型（"llm" 或 "reranker"）
            base_model_path: 基础模型路径（仅当is_lora=True时需要）
            is_lora: 是否为LoRA模型
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.base_model_path = Path(base_model_path) if base_model_path else None
        self.is_lora = is_lora
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.vllm_client = None  # vLLM客户端（如果使用）

    def _load_llm_model(self):
        """加载LLM模型（支持LoRA）"""
        if self.model is not None:
            return

        if self.is_lora:
            # 加载LoRA模型
            from peft import PeftModel
            # AutoModelForCausalLM已在文件顶部导入，无需重复导入

            if self.base_model_path is None:
                raise ValueError("LoRA模型需要指定base_model_path")

            print(f"加载基础模型: {self.base_model_path}")
            import time
            load_start = time.time()
            # 对于推理，使用单GPU通常更快（避免跨GPU通信）
            base_model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model_path),
                trust_remote_code=True,
                dtype=torch.bfloat16,  # 使用dtype替代torch_dtype
                device_map="cuda:0",  # 使用单GPU，避免跨GPU通信开销
            )
            print(f"  基础模型加载完成 ({time.time() - load_start:.2f}秒)")
            
            # 显示GPU分配信息
            if hasattr(base_model, 'hf_device_map'):
                print(f"  GPU分配: {base_model.hf_device_map}")
            else:
                device = next(base_model.parameters()).device
                print(f"  模型设备: {device}")

            print(f"加载LoRA adapter: {self.model_path}")
            adapter_start = time.time()
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            print(f"  LoRA adapter加载完成 ({time.time() - adapter_start:.2f}秒)")
            self.model.eval()
            
            # 显示最终模型设备
            device = next(self.model.parameters()).device
            print(f"  最终模型设备: {device}")

            # Tokenizer使用基础模型的
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.base_model_path),
                trust_remote_code=True,
            )
        else:
            # 加载完整模型
            print(f"加载模型: {self.model_path}")
            import time
            load_start = time.time()
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            print(f"  Tokenizer加载完成 ({time.time() - load_start:.2f}秒)")
            
            model_start = time.time()
            # 对于推理，使用单GPU通常更快（避免跨GPU通信）
            # 如果显存足够，建议使用device_map="cuda:0"
            # 如果显存不足，可以使用device_map="auto"自动分配
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                dtype=torch.bfloat16,  # 使用dtype替代torch_dtype
                device_map="cuda:0",  # 使用单GPU，避免跨GPU通信开销
            )
            print(f"  模型加载完成 ({time.time() - model_start:.2f}秒)")
            
            # 显示GPU分配信息
            if hasattr(self.model, 'hf_device_map'):
                print(f"  GPU分配: {self.model.hf_device_map}")
            else:
                device = next(self.model.parameters()).device
                print(f"  模型设备: {device}")
            
            self.model.eval()
            print(f"  总加载时间: {time.time() - load_start:.2f}秒")

    def _load_reranker_model(self):
        """加载Reranker模型"""
        if self.model is not None:
            return

        print(f"加载Reranker模型: {self.model_path}")
        import time
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        print(f"  Tokenizer加载完成 ({time.time() - start_time:.2f}秒)")
        
        load_start = time.time()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            dtype=torch.float16,  # 使用dtype替代torch_dtype
            device_map="auto",
        )
        print(f"  模型加载完成 ({time.time() - load_start:.2f}秒)")
        
        self.model.eval()
        print(f"  总加载时间: {time.time() - start_time:.2f}秒")

    def _is_chinese_text(self, text: str) -> bool:
        """判断文本是否主要是中文"""
        if not text:
            return False
        # 统计中文字符数量
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        # 如果中文字符占比超过30%，认为是中文文本
        return chinese_chars / len(text) > 0.3 if len(text) > 0 else False

    def _tokenize_text(self, text: str) -> List[str]:
        """根据文本语言进行分词"""
        if not text:
            return []
        
        # 判断是否为中文文本
        if self._is_chinese_text(text):
            # 使用jieba进行中文分词
            if JIEBA_AVAILABLE:
                return list(jieba.cut(text, cut_all=False))
            else:
                # 如果没有jieba，使用字符级别分词
                return list(text)
        else:
            # 使用NLTK进行英文分词
            if NLTK_AVAILABLE:
                return word_tokenize(text.lower())
            else:
                # 如果没有NLTK，使用空格分词
                return text.lower().split()

    def _calculate_bleu(self, predicted: str, expected: str) -> float:
        """计算BLEU分数（支持中文和英文）"""
        if not predicted or not expected:
            return 0.0

        try:
            # 根据文本语言进行分词
            pred_tokens = self._tokenize_text(predicted)
            exp_tokens = self._tokenize_text(expected)

            if not pred_tokens or not exp_tokens:
                return 0.0

            # 使用NLTK计算BLEU分数
            if NLTK_AVAILABLE:
                smoothing = SmoothingFunction().method1
                score = sentence_bleu([exp_tokens], pred_tokens, smoothing_function=smoothing)
                return float(score)
            else:
                # 如果没有NLTK，返回简单的n-gram匹配分数
                # 计算1-gram和2-gram的重叠率
                pred_bigrams = set(zip(pred_tokens, pred_tokens[1:])) if len(pred_tokens) > 1 else set()
                exp_bigrams = set(zip(exp_tokens, exp_tokens[1:])) if len(exp_tokens) > 1 else set()
                
                pred_unigrams = set(pred_tokens)
                exp_unigrams = set(exp_tokens)
                
                unigram_overlap = len(pred_unigrams & exp_unigrams) / len(exp_unigrams) if exp_unigrams else 0.0
                bigram_overlap = len(pred_bigrams & exp_bigrams) / len(exp_bigrams) if exp_bigrams else 0.0
                
                return (unigram_overlap + bigram_overlap) / 2.0
        except Exception as e:
            # 如果计算失败，返回0.0
            return 0.0

    def _calculate_rouge(self, predicted: str, expected: str) -> Dict[str, float]:
        """计算ROUGE分数（支持中文和英文）"""
        if not predicted or not expected:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        try:
            # 判断是否为中文文本
            is_chinese = self._is_chinese_text(predicted) or self._is_chinese_text(expected)
            
            if is_chinese:
                # 对于中文文本，使用jieba分词后手动计算ROUGE
                return self._calculate_rouge_chinese(predicted, expected)
            else:
                # 对于英文文本，使用rouge_score库
                if not ROUGE_AVAILABLE:
                    return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
                
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                scores = scorer.score(expected, predicted)
                return {
                    "rouge1": scores['rouge1'].fmeasure,
                    "rouge2": scores['rouge2'].fmeasure,
                    "rougeL": scores['rougeL'].fmeasure,
                }
        except Exception as e:
            # 如果计算失败，返回0.0
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def _calculate_rouge_chinese(self, predicted: str, expected: str) -> Dict[str, float]:
        """计算中文文本的ROUGE分数（使用jieba分词）"""
        try:
            # 使用jieba分词
            if JIEBA_AVAILABLE:
                pred_tokens = list(jieba.cut(predicted, cut_all=False))
                exp_tokens = list(jieba.cut(expected, cut_all=False))
            else:
                # 如果没有jieba，使用字符级别
                pred_tokens = list(predicted)
                exp_tokens = list(expected)
            
            if not pred_tokens or not exp_tokens:
                return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
            # 计算ROUGE-1（unigram overlap）
            pred_unigrams = set(pred_tokens)
            exp_unigrams = set(exp_tokens)
            common_unigrams = pred_unigrams & exp_unigrams
            
            if len(exp_unigrams) == 0:
                rouge1 = 0.0
            else:
                precision_1 = len(common_unigrams) / len(pred_unigrams) if len(pred_unigrams) > 0 else 0.0
                recall_1 = len(common_unigrams) / len(exp_unigrams)
                rouge1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
            
            # 计算ROUGE-2（bigram overlap）
            if len(pred_tokens) < 2 or len(exp_tokens) < 2:
                rouge2 = 0.0
            else:
                pred_bigrams = set(zip(pred_tokens, pred_tokens[1:]))
                exp_bigrams = set(zip(exp_tokens, exp_tokens[1:]))
                common_bigrams = pred_bigrams & exp_bigrams
                
                if len(exp_bigrams) == 0:
                    rouge2 = 0.0
                else:
                    precision_2 = len(common_bigrams) / len(pred_bigrams) if len(pred_bigrams) > 0 else 0.0
                    recall_2 = len(common_bigrams) / len(exp_bigrams)
                    rouge2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2) if (precision_2 + recall_2) > 0 else 0.0
            
            # 计算ROUGE-L（最长公共子序列）
            rougeL = self._calculate_lcs(pred_tokens, exp_tokens)
            
            return {
                "rouge1": rouge1,
                "rouge2": rouge2,
                "rougeL": rougeL,
            }
        except Exception:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def _calculate_lcs(self, seq1: List[str], seq2: List[str]) -> float:
        """计算最长公共子序列（LCS）的ROUGE-L分数"""
        if not seq1 or not seq2:
            return 0.0
        
        # 动态规划计算LCS长度
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        if lcs_length == 0:
            return 0.0
        
        # ROUGE-L = (LCS长度 / 参考长度 + LCS长度 / 候选长度) / 2
        precision_l = lcs_length / len(seq1) if len(seq1) > 0 else 0.0
        recall_l = lcs_length / len(seq2) if len(seq2) > 0 else 0.0
        
        if precision_l + recall_l == 0:
            return 0.0
        
        return 2 * precision_l * recall_l / (precision_l + recall_l)

    def _extract_citations(self, text: str) -> List[int]:
        """从答案中提取引用编号"""
        # 匹配格式：【1, 2, 3】或【1,2,3】
        pattern = r'【(\d+(?:,\s*\d+)*)】'
        match = re.search(pattern, text)
        if match:
            citations_str = match.group(1)
            citations = [int(x.strip()) for x in citations_str.split(',')]
            return citations
        return []

    def evaluate_llm(
        self,
        test_data_path: Path,
        tokenizer_path: Optional[Path] = None,
        max_length: int = 2048,  # 总长度限制（输入+输出），默认2048以支持长输入
        temperature: float = 0.0,  # 默认使用贪心解码（temperature=0）
        do_sample: bool = False,  # 默认不使用采样，贪心解码更快
        use_vllm: bool = False,  # 是否使用vLLM API（更快）
        vllm_base_url: Optional[str] = None,  # vLLM服务地址
        vllm_model: Optional[str] = None,  # vLLM模型名称
        batch_size: int = 8,  # 批量生成大小（仅vLLM）
    ) -> Dict[str, Any]:
        """
        评估LLM模型

        Args:
            test_data_path: 测试数据路径
            tokenizer_path: Tokenizer路径（如果与模型路径不同）
            max_length: 最大生成长度
            temperature: 生成温度
            do_sample: 是否采样

        Returns:
            评估结果字典
        """
        # 仅支持vLLM模式
        if not use_vllm:
            raise ValueError("当前仅支持使用vLLM进行LLM评估")
        
        if not VLLM_AVAILABLE:
            raise ValueError("vLLM不可用，请确保已安装vLLM相关依赖")
        
        if not vllm_base_url:
            raise ValueError("使用vLLM时必须指定 vllm_base_url")
        
        if not vllm_model:
            raise ValueError("使用vLLM时必须显式指定 vllm_model")
        
        print("  使用vLLM API进行推理...")
        self.vllm_client = LocalLLMClient(base_url=vllm_base_url, model=vllm_model)
        print("  ✓ vLLM客户端已初始化")

        # 加载测试数据
        print(f"  正在加载测试数据: {test_data_path}")
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        print(f"  ✓ 已加载 {len(test_data)} 条测试数据")

        results = {
            "total_samples": len(test_data),
            "predictions": [],
            "metrics": {},
        }

        # 评估每个样本
        all_bleu_scores = []
        all_rouge1_scores = []
        all_rouge2_scores = []
        all_rougeL_scores = []

        total_samples = len(test_data)
        print(f"  开始评估 {total_samples} 个样本...")
        
        # 使用vLLM批量生成
        if not self.vllm_client:
            raise ValueError("vLLM客户端未初始化")
        
        return self._evaluate_llm_with_vllm(
            test_data, max_length, temperature, do_sample, batch_size
        )

    def _batch_chat_vllm(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        使用vLLM的批量API进行批量推理（真正的批量处理，比逐个调用快得多）
        
        Args:
            messages_list: 消息列表的列表，每个元素是一个对话的消息列表
            temperature: 生成温度
            max_tokens: 最大生成token数
            
        Returns:
            生成的文本列表
        """
        if not self.vllm_client:
            raise ValueError("vLLM客户端未初始化")
        
        # vLLM的OpenAI兼容API支持批量请求
        # 通过传递多个请求对象来实现批量处理
        from openai import OpenAI
        
        # 获取vLLM客户端的基础URL和API key
        base_url = self.vllm_client.base_url
        api_key = self.vllm_client.api_key
        model = self.vllm_client.model
        
        # 创建OpenAI客户端用于批量请求
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 准备批量请求
        batch_requests = []
        for messages in messages_list:
            request = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": 20,
                }
            }
            batch_requests.append(request)
        
        # 使用vLLM的批量API（如果支持）或并发请求
        # 注意：vLLM的OpenAI兼容API可能不支持真正的批量端点
        # 因此我们使用并发请求来模拟批量处理
        import concurrent.futures
        
        def single_request(request):
            completion = client.chat.completions.create(**request)
            return completion.choices[0].message.content
        
        # 使用线程池并发执行（vLLM服务器端会处理批量优化）
        # 使用字典来保持顺序
        responses = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(batch_requests), 16)) as executor:
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

    def _evaluate_llm_with_vllm(
        self,
        test_data: List[Dict],
        max_length: int,
        temperature: float,
        do_sample: bool,
        batch_size: int,
    ) -> Dict[str, Any]:
        """使用vLLM API进行批量评估（更快）"""
        results = {
            "total_samples": len(test_data),
            "predictions": [],
            "metrics": {},
        }

        all_bleu_scores = []
        all_rouge1_scores = []
        all_rouge2_scores = []
        all_rougeL_scores = []

        total_samples = len(test_data)
        print(f"  使用vLLM批量生成（batch_size={batch_size}）...")
        
        # 准备所有prompts
        prompts = []
        items_data = []
        for item in test_data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            prompt = f"{instruction}\n{input_text}" if input_text else instruction
            prompts.append(prompt)
            items_data.append(item)

        # 批量生成（使用vLLM的批量API）
        import time
        total_start = time.time()
        predicted_texts = []
        
        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            batch_items = items_data[batch_idx:batch_idx + batch_size]
            
            print(f"  处理批次 {batch_idx//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} "
                  f"（样本 {batch_idx+1}-{min(batch_idx+batch_size, len(prompts))}/{total_samples}）...")
            
            batch_start = time.time()
            
            # 构建消息列表
            messages_list = [
                [{"role": "user", "content": prompt}]
                for prompt in batch_prompts
            ]
            
            # 使用vLLM的批量API（真正的批量推理）
            max_new_tokens = min(512, max_length // 4)  # 预留空间
            batch_responses = self._batch_chat_vllm(
                messages_list=messages_list,
                temperature=temperature if do_sample else 0.0,
                max_tokens=max_new_tokens,
            )
            
            batch_time = time.time() - batch_start
            tokens_generated = sum(len(resp) // 3 for resp in batch_responses)  # 粗略估算
            tokens_per_sec = tokens_generated / batch_time if batch_time > 0 else 0
            print(f"    批次完成（{batch_time:.2f}秒，约{tokens_generated} tokens，{tokens_per_sec:.1f} tokens/s）")
            
            predicted_texts.extend(batch_responses)
        
        total_time = time.time() - total_start
        print(f"  ✓ 所有样本生成完成（总时间: {total_time:.2f}秒，平均: {total_time/total_samples:.2f}秒/样本）")
        
        # 计算指标
        print(f"  正在计算指标...")
        for idx, (item, predicted_text) in enumerate(zip(items_data, predicted_texts), 1):
            expected_output = item.get("output", "")
            
            # 提取引用
            expected_citations = self._extract_citations(expected_output)
            predicted_citations = self._extract_citations(predicted_text)
            # 移除引用标记用于计算指标
            expected_clean = re.sub(r'【.*?】', '', expected_output).strip()
            predicted_clean = re.sub(r'【.*?】', '', predicted_text).strip()

            # 计算指标
            bleu_score = self._calculate_bleu(predicted_clean, expected_clean)
            rouge_scores = self._calculate_rouge(predicted_clean, expected_clean)

            all_bleu_scores.append(bleu_score)
            all_rouge1_scores.append(rouge_scores["rouge1"])
            all_rouge2_scores.append(rouge_scores["rouge2"])
            all_rougeL_scores.append(rouge_scores["rougeL"])

            # 计算引用准确率
            citation_accuracy = 0.0
            if expected_citations and predicted_citations:
                intersection = set(expected_citations) & set(predicted_citations)
                if expected_citations:
                    citation_accuracy = len(intersection) / len(expected_citations)

            results["predictions"].append({
                "expected": expected_output,
                "predicted": predicted_text,
                "expected_clean": expected_clean,
                "predicted_clean": predicted_clean,
                "bleu": bleu_score,
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"],
                "citation_accuracy": citation_accuracy,
                "expected_citations": expected_citations,
                "predicted_citations": predicted_citations,
            })

        # 计算平均指标
        results["metrics"] = {
            "bleu": np.mean(all_bleu_scores) if all_bleu_scores else 0.0,
            "rouge1": np.mean(all_rouge1_scores) if all_rouge1_scores else 0.0,
            "rouge2": np.mean(all_rouge2_scores) if all_rouge2_scores else 0.0,
            "rougeL": np.mean(all_rougeL_scores) if all_rougeL_scores else 0.0,
            "citation_accuracy": np.mean([p["citation_accuracy"] for p in results["predictions"]]),
        }
        print(f"  ✓ 指标计算完成")

        print(f"✓ LLM评估完成: {results['total_samples']} 个样本")
        print(f"  BLEU: {results['metrics']['bleu']:.4f}")
        print(f"  ROUGE-1: {results['metrics']['rouge1']:.4f}")
        print(f"  ROUGE-2: {results['metrics']['rouge2']:.4f}")
        print(f"  ROUGE-L: {results['metrics']['rougeL']:.4f}")
        print(f"  Citation Accuracy: {results['metrics']['citation_accuracy']:.4f}")

        return results

    def _calculate_ndcg(self, scores: List[float], labels: List[int], k: int = 10) -> float:
        """计算NDCG@K"""
        if len(scores) == 0:
            return 0.0

        # 按分数排序
        sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        sorted_labels = [label for _, label in sorted_pairs]

        # 计算DCG@K
        dcg = 0.0
        for i, label in enumerate(sorted_labels[:k]):
            if label > 0:  # 只考虑相关文档
                dcg += (2 ** label - 1) / np.log2(i + 2)

        # 计算IDCG@K（理想情况下的DCG）
        ideal_labels = sorted(labels, reverse=True)
        idcg = 0.0
        for i, label in enumerate(ideal_labels[:k]):
            if label > 0:
                idcg += (2 ** label - 1) / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_mrr(self, scores: List[float], labels: List[int]) -> float:
        """计算MRR（平均倒数排名）"""
        if len(scores) == 0:
            return 0.0

        # 按分数排序
        sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        sorted_labels = [label for _, label in sorted_pairs]

        # 找到第一个相关文档的位置
        for i, label in enumerate(sorted_labels):
            if label > 0:
                return 1.0 / (i + 1)

        return 0.0

    def _calculate_precision_recall_at_k(
        self, scores: List[float], labels: List[int], k: int = 10
    ) -> Tuple[float, float]:
        """计算Precision@K和Recall@K"""
        if len(scores) == 0:
            return 0.0, 0.0

        # 按分数排序
        sorted_pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        sorted_labels = [label for _, label in sorted_pairs]

        # 计算相关文档总数
        total_relevant = sum(1 for label in labels if label > 0)

        # 计算Top-K中的相关文档数
        relevant_in_topk = sum(1 for label in sorted_labels[:k] if label > 0)

        precision = relevant_in_topk / k if k > 0 else 0.0
        recall = relevant_in_topk / total_relevant if total_relevant > 0 else 0.0

        return precision, recall

    def evaluate_reranker(
        self,
        test_data_path: Path,
        topk: int = 10,
    ) -> Dict[str, Any]:
        """
        评估Reranker模型

        Args:
            test_data_path: 测试数据路径
            topk: Top-K评估

        Returns:
            评估结果字典
        """
        # 加载模型
        self._load_reranker_model()

        # 加载测试数据
        print(f"  加载测试数据: {test_data_path}")
        test_data = []
        with open(test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        print(f"  已加载 {len(test_data)} 条测试数据")

        results = {
            "total_samples": len(test_data),
            "predictions": [],
            "metrics": {},
        }

        # 按query分组评估
        query_groups: Dict[str, List[Dict]] = {}
        for item in test_data:
            query = item.get("query", "")
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append(item)

        all_ndcg_scores = []
        all_mrr_scores = []
        all_precision_scores = []
        all_recall_scores = []

        total_queries = len(query_groups)
        total_items = sum(len(items) for items in query_groups.values())
        print(f"  共有 {total_queries} 个查询，{total_items} 个文档对，开始评估...")

        processed_items = 0
        # 对每个query进行评估
        for idx, (query, items) in enumerate(query_groups.items(), 1):
            if idx % 50 == 0 or idx == 1:
                print(f"  处理查询 {idx}/{total_queries} (已处理 {processed_items}/{total_items} 个文档对)...")
            
            scores = []
            labels = []

            # 计算每个文档的分数
            for item in items:
                # content可能是字符串或列表
                content_field = item.get("content", "")
                label = item.get("label", 0)

                # 处理content字段（可能是列表或字符串）
                if isinstance(content_field, list):
                    # 如果是列表，遍历所有元素
                    for content_item in content_field:
                        content = content_item if isinstance(content_item, str) else str(content_item)
                        
                        # 计算相关性分数
                        inputs = self.tokenizer(
                            query,
                            content,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512,
                            padding=True,
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            score = outputs.logits[0][0].item()

                        scores.append(score)
                        labels.append(label)  # 每个content_item使用相同的label
                        processed_items += 1
                else:
                    # 如果是字符串，直接使用
                    content = str(content_field)
                    
                    # 计算相关性分数
                    inputs = self.tokenizer(
                        query,
                        content,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        score = outputs.logits[0][0].item()

                    scores.append(score)
                    labels.append(label)
                    processed_items += 1

            # 计算指标
            ndcg = self._calculate_ndcg(scores, labels, k=topk)
            mrr = self._calculate_mrr(scores, labels)
            precision, recall = self._calculate_precision_recall_at_k(scores, labels, k=topk)

            all_ndcg_scores.append(ndcg)
            all_mrr_scores.append(mrr)
            all_precision_scores.append(precision)
            all_recall_scores.append(recall)

            results["predictions"].append({
                "query": query,
                "ndcg": ndcg,
                "mrr": mrr,
                "precision": precision,
                "recall": recall,
                "num_docs": len(items),
            })

        # 计算平均指标
        results["metrics"] = {
            f"ndcg@{topk}": np.mean(all_ndcg_scores) if all_ndcg_scores else 0.0,
            "mrr": np.mean(all_mrr_scores) if all_mrr_scores else 0.0,
            f"precision@{topk}": np.mean(all_precision_scores) if all_precision_scores else 0.0,
            f"recall@{topk}": np.mean(all_recall_scores) if all_recall_scores else 0.0,
        }

        print(f"✓ Reranker评估完成: {results['total_samples']} 个样本, {len(query_groups)} 个查询")
        print(f"  NDCG@{topk}: {results['metrics'][f'ndcg@{topk}']:.4f}")
        print(f"  MRR: {results['metrics']['mrr']:.4f}")
        print(f"  Precision@{topk}: {results['metrics'][f'precision@{topk}']:.4f}")
        print(f"  Recall@{topk}: {results['metrics'][f'recall@{topk}']:.4f}")

        return results

    def evaluate(
        self,
        test_data_path: Path,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        评估模型（根据模型类型自动选择评估方法）

        Args:
            test_data_path: 测试数据路径
            **kwargs: 其他评估参数

        Returns:
            评估结果字典
        """
        if self.model_type == "llm":
            return self.evaluate_llm(test_data_path, **kwargs)
        elif self.model_type == "reranker":
            return self.evaluate_reranker(test_data_path, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
