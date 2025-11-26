"""
微调数据生成模块

从train_qa_pair.json生成微调训练数据，包括：
1. train_data.json - RAG检索数据
2. summary_data - SFT训练数据
3. rerank_data - Reranker训练数据

参考原项目：EVRAG/generate_sft_data.py
"""

import json
import re
import random
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langchain_core.documents import Document

from ..retriever import BM25Retriever, MilvusRetriever
from ..reranker import BGEReranker, SiliconFlowReranker
from ..client import ChatClient, BaseLLMClient, MongoDBClient
from ..tool_func import merge_docs


class SFTDataGenerator:
    """
    微调数据生成器
    
    实现从train_qa_pair.json生成微调训练数据的完整流程。
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        mongodb_client: Optional[MongoDBClient] = None,
        bm25_topk: int = 5,
        milvus_topk: int = 10,
        reranker_topk: int = 5,
        max_input_size: int = 4096,
        test_rate: float = 0.08,
        rerank_dev_size: int = 1000,
        max_workers: int = 10,
        seed: int = 42,
        use_siliconflow_reranker: bool = True,
    ):
        """
        初始化微调数据生成器
        
        Args:
            llm_client: LLM客户端实例（用于生成问答数据，应使用Deepseek API）
            mongodb_client: MongoDB客户端（用于merge_docs）
            bm25_topk: BM25检索数量
            milvus_topk: Milvus检索数量
            reranker_topk: Reranker重排序数量
            max_input_size: 最大输入长度
            test_rate: 测试集比例
            rerank_dev_size: Reranker开发集大小
            max_workers: 最大并发工作线程数
            seed: 随机种子
            use_siliconflow_reranker: 是否使用SiliconFlow Reranker（默认True，用于生成Reranker数据）
        """
        self.llm_client = llm_client
        self.mongodb_client = mongodb_client
        self.bm25_topk = bm25_topk
        self.milvus_topk = milvus_topk
        self.reranker_topk = reranker_topk
        self.max_input_size = max_input_size
        self.test_rate = test_rate
        self.rerank_dev_size = rerank_dev_size
        self.max_workers = max_workers
        random.seed(seed)
        
        # 初始化检索器
        self.bm25_retriever = BM25Retriever(docs=None, retrieve=True)
        self.milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
        
        # 初始化重排序器（根据参数选择）
        if use_siliconflow_reranker:
            try:
                self.reranker = SiliconFlowReranker()
                print("✓ Using SiliconFlow Reranker for data generation")
            except Exception as e:
                print(f"⚠ Failed to initialize SiliconFlow Reranker: {e}")
                print("   Falling back to BGE Reranker")
                self.reranker = BGEReranker()
        else:
            self.reranker = BGEReranker()
        
        self.chat_client = ChatClient(llm_client)
        
        # 文件锁（用于并发写入）
        self.file_lock = threading.Lock()
    
    def generate_train_data(
        self,
        train_qa_path: Path,
        output_path: Path,
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        """
        生成train_data.json（RAG检索+LLM生成）
        
        Args:
            train_qa_path: 训练集QA对文件路径
            output_path: 输出文件路径（train_data.json）
            checkpoint_path: checkpoint文件路径（可选）
            
        参考原项目：EVRAG/generate_sft_data.py:48-69
        """
        # 加载QA对
        with open(train_qa_path, "r", encoding="utf-8") as f:
            train_qa_pairs = json.load(f)
        
        print(f"Loaded {len(train_qa_pairs)} QA pairs from {train_qa_path}")
        
        # 加载checkpoint（如果存在）
        processed_queries = set()
        checkpoint_file = checkpoint_path if checkpoint_path else output_path
        if checkpoint_file.exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            info = json.loads(line)
                            processed_queries.add(info.get("query", ""))
                        except:
                            pass
            print(f"Loaded checkpoint: {len(processed_queries)} processed queries")
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理每个QA对
        def process_qa_pair(qa_pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """处理单个QA对"""
            query = qa_pair.get("question", "").strip()
            if not query:
                return None
            
            # 检查是否已处理
            if query in processed_queries:
                return None
            
            # 重试机制：最多重试3次（针对连接错误）
            max_retries = 3
            import time
            
            for attempt in range(max_retries):
                try:
                    # RAG检索
                    bm25_docs = self.bm25_retriever.retrieve_topk(query, topk=self.bm25_topk)
                    milvus_docs = self.milvus_retriever.retrieve_topk(query, topk=self.milvus_topk)
                    merged_docs = merge_docs(bm25_docs, milvus_docs, mongodb_client=self.mongodb_client)
                    ranked_docs = self.reranker.rank(query, merged_docs, topk=self.reranker_topk)
                    
                    # 构建上下文
                    context_str = "\n".join([
                        f"{idx+1}.{doc.page_content}" 
                        for idx, doc in enumerate(ranked_docs)
                    ])
                    
                    # LLM生成响应
                    response = self.chat_client.chat(query, context_str)
                    
                    # 清理response：去除思考过程和占位符
                    import re
                    # 1. 去除 <think> 标签及其内容（Qwen3思考模式输出）
                    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
                    # 2. 去除 "答案内容：" 前缀（模型可能输出的格式）
                    response = re.sub(r'^答案内容[：:]\s*', '', response, flags=re.MULTILINE)
                    # 3. 去除 {答案} 或 {从...} 占位符，保留实际答案
                    response = re.sub(r'\{答案\}', '', response)
                    response = re.sub(r'\{从[^}]*\}', '', response)
                    # 4. 去除多余的换行和空格
                    response = re.sub(r'\n\s*\n+', '\n', response).strip()
                    # 5. 如果response只包含引用标记没有实际答案，标记为"无答案"
                    if re.match(r'^[\s【】0-9,，]*$', response.strip()):
                        response = "无答案"
                    
                    # 提取文档内容
                    context_list = [doc.page_content for doc in ranked_docs]
                    merged_docs_list = [doc.page_content for doc in merged_docs]
                    
                    return {
                        "query": query,
                        "context": context_list,
                        "response": response,
                        "merged_docs": merged_docs_list,
                    }
                except Exception as e:
                    error_str = str(e)
                    # 判断是否为连接错误
                    is_connection_error = (
                        "Connection" in error_str or 
                        "connection" in error_str.lower() or
                        "timeout" in error_str.lower() or
                        "refused" in error_str.lower()
                    )
                    
                    if is_connection_error and attempt < max_retries - 1:
                        # 连接错误，等待后重试（指数退避）
                        wait_time = min((attempt + 1) * 3, 10)  # 最多等待10秒
                        print(f"[WARNING] Connection error for query '{query[:50]}...': {error_str[:100]}")
                        print(f"         Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        # 非连接错误或最后一次重试失败
                        if is_connection_error:
                            print(f"[ERROR] Connection error for query '{query[:50]}...': {error_str[:100]}, failed after {max_retries} retries")
                        else:
                            print(f"[ERROR] Error processing query '{query[:50]}...': {error_str[:100]}")
                        return None
            
            return None
        
        # 并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                i: executor.submit(process_qa_pair, qa_pair)
                for i, qa_pair in enumerate(train_qa_pairs)
            }
            
            # 写入结果
            with open(output_path, "a" if checkpoint_file.exists() else "w", encoding="utf-8") as f:
                for i in tqdm(futures, desc="Generating train_data"):
                    future = futures[i]
                    result = future.result()
                    if result:
                        self.file_lock.acquire()
                        try:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()
                        finally:
                            self.file_lock.release()
        
        print(f"Train data saved to: {output_path}")
    
    def generate_summary_data(
        self,
        train_data_path: Path,
        output_dir: Path,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        生成summary_data（SFT训练数据）
        
        Args:
            train_data_path: train_data.json文件路径
            output_dir: 输出目录
            
        Returns:
            (训练集, 测试集) 元组
            
        参考原项目：EVRAG/generate_sft_data.py:86-117
        """
        from ..client.chat_client import LLM_CHAT_PROMPT
        
        summary_train = []
        summary_test = []
        
        # 读取train_data.json
        with open(train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    info = json.loads(line)
                    response = info.get("response", "")
                    query = info.get("query", "").strip()
                    context_list = info.get("context", [])
                    
                    # 解析引用标记
                    all_cites = re.findall(r"[【](.*?)[】]", response)
                    cites = []
                    for cite in all_cites:
                        cite = re.sub(r"[{} 【】]", "", cite)
                        cite = cite.replace(",", "，")
                        cite_numbers = [int(k) for k in cite.split("，") if k.strip().isdigit()]
                        cites.extend(cite_numbers)
                    
                    cites = sorted(list(set(cites)))
                    cites_str = ",".join([str(c) for c in cites])
                    
                    # 提取答案
                    answer = re.sub(r"[【](.*?)[】]", "", response)
                    answer = re.sub(r"[{}【】]", "", answer)
                    # 去除 "答案内容：" 前缀（如果存在）
                    answer = re.sub(r'^答案内容[：:]\s*', '', answer, flags=re.MULTILINE).strip()
                    
                    # 格式化答案
                    if cites_str:
                        format_answer = answer + f"【{cites_str}】"
                    else:
                        format_answer = "无答案"
                    
                    # 构建context字符串
                    context_str = "\n".join([
                        f"{idx+1}.{doc}" 
                        for idx, doc in enumerate(context_list)
                    ])
                    
                    # 限制context长度
                    if len(context_str) > self.max_input_size:
                        context_str = context_str[:self.max_input_size]
                    
                    # 构建instruction
                    instruction = LLM_CHAT_PROMPT.format(query=query, context=context_str)
                    
                    item = {
                        "query": query,
                        "context": context_str,
                        "instruction": instruction,
                        "input": "",
                        "output": format_answer,
                    }
                    
                    # 切分训练集和测试集
                    if random.random() < self.test_rate:
                        summary_test.append(item)
                    else:
                        summary_train.append(item)
                        
                except Exception as e:
                    print(f"Error processing train_data line: {e}")
                    continue
        
        # 保存文件
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / "summary_data" / "train.json"
        test_path = output_dir / "summary_data" / "test.json"
        
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(summary_train, f, ensure_ascii=False, indent=2)
        print(f"Summary train data saved to: {train_path}, {len(summary_train)} items")
        
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(summary_test, f, ensure_ascii=False, indent=2)
        print(f"Summary test data saved to: {test_path}, {len(summary_test)} items")
        
        return summary_train, summary_test
    
    def generate_rerank_data(
        self,
        train_data_path: Path,
        output_dir: Path,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        生成rerank_data（Reranker训练数据）
        
        Args:
            train_data_path: train_data.json文件路径
            output_dir: 输出目录
            
        Returns:
            (训练集, 开发集, 测试集) 元组
            
        参考原项目：EVRAG/generate_sft_data.py:118-142
        """
        rerank_train = []
        rerank_test = []
        
        # 读取train_data.json
        with open(train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    info = json.loads(line)
                    query = info.get("query", "").strip()
                    context_list = info.get("context", [])
                    merged_docs_list = info.get("merged_docs", [])
                    response = info.get("response", "")
                    
                    # 解析答案
                    all_cites = re.findall(r"[【](.*?)[】]", response)
                    cites = []
                    for cite in all_cites:
                        cite = re.sub(r"[{} 【】]", "", cite)
                        cite = cite.replace(",", "，")
                        cite_numbers = [int(k) for k in cite.split("，") if k.strip().isdigit()]
                        cites.extend(cite_numbers)
                    
                    cites = sorted(list(set(cites)))
                    answer = re.sub(r"[【](.*?)[】]", "", response)
                    answer = re.sub(r"[{}【】]", "", answer).strip()
                    
                    format_answer = answer + f"【{','.join([str(c) for c in cites])}】" if cites else "无答案"
                    
                    # 负样本文档（不在context中的merged_docs）
                    neg_docs = [doc for doc in merged_docs_list if doc not in context_list]
                    
                    # 判断是否加入测试集
                    is_test = random.random() < self.test_rate
                    
                    if format_answer != "无答案":
                        if is_test:
                            # 测试集：生成包含正样本、中等样本和负样本的列表
                            if context_list:
                                content_list = [context_list[0]]
                                if len(context_list) >= 2:
                                    content_list.append(random.choice(context_list[-2:]))
                                if neg_docs:
                                    content_list.append(random.choice(neg_docs))
                                # 测试集格式：{"query": "...", "content": [...]}
                                rerank_test.append({"query": query, "content": content_list})
                        else:
                            # 训练集：生成正样本、中等样本、负样本
                            if context_list:
                                positive = context_list[0]
                                rerank_train.append({"query": query, "content": positive, "label": 2})
                                
                                if len(context_list) >= 2:
                                    middle = random.choice(context_list[-2:])
                                    rerank_train.append({"query": query, "content": middle, "label": 1})
                                
                                if neg_docs:
                                    negative = random.choice(neg_docs)
                                    rerank_train.append({"query": query, "content": negative, "label": 0})
                    else:
                        # 答案为"无答案"
                        if not is_test:
                            # 训练集：生成负样本
                            if merged_docs_list:
                                negative = random.choice(merged_docs_list)
                                rerank_train.append({"query": query, "content": negative, "label": 0})
                    
                except Exception as e:
                    print(f"Error processing train_data line for rerank: {e}")
                    continue
        
        # 过滤空内容
        rerank_train = [
            item for item in rerank_train 
            if len(item.get("query", "")) > 0 and len(item.get("content", "")) > 0
        ]
        
        # 切分开发集
        rerank_dev = rerank_train[-self.rerank_dev_size:]
        rerank_train = rerank_train[:-self.rerank_dev_size]
        
        # 打乱训练集
        random.shuffle(rerank_train)
        
        # 保存文件
        output_dir.mkdir(parents=True, exist_ok=True)
        train_path = output_dir / "rerank_data" / "train.json"
        dev_path = output_dir / "rerank_data" / "dev.json"
        test_path = output_dir / "rerank_data" / "test.json"
        
        train_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(train_path, "w", encoding="utf-8") as f:
            for item in rerank_train:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Rerank train data saved to: {train_path}, {len(rerank_train)} items")
        
        with open(dev_path, "w", encoding="utf-8") as f:
            for item in rerank_dev:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Rerank dev data saved to: {dev_path}, {len(rerank_dev)} items")
        
        with open(test_path, "w", encoding="utf-8") as f:
            for item in rerank_test:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Rerank test data saved to: {test_path}, {len(rerank_test)} items")
        
        return rerank_train, rerank_dev, rerank_test
    
    def generate_test_verify_data(
        self,
        test_qa_path: Path,
        output_path: Path,
        verify_ratio: float = 0.1,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        从test_qa_pair.json筛选验证集（test_qa_pair_verify.json）
        
        Args:
            test_qa_path: 测试集QA对文件路径
            output_path: 输出文件路径（test_qa_pair_verify.json）
            verify_ratio: 验证集比例（默认0.1，即10%）
            seed: 随机种子
            
        Returns:
            验证集列表
        """
        import random
        random.seed(seed)
        
        # 加载测试集
        with open(test_qa_path, "r", encoding="utf-8") as f:
            test_qa_pairs = json.load(f)
        
        print(f"Loaded {len(test_qa_pairs)} test QA pairs from {test_qa_path}")
        
        # 随机筛选验证集
        verify_size = max(1, int(len(test_qa_pairs) * verify_ratio))
        verify_pairs = random.sample(test_qa_pairs, verify_size)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存验证集
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(verify_pairs, f, ensure_ascii=False, indent=4)
        
        print(f"Verify data saved to: {output_path} ({len(verify_pairs)} items)")
        return verify_pairs
    
    def generate_test_pred_data(
        self,
        test_verify_path: Path,
        output_path: Path,
        bm25_topk: Optional[int] = None,
        milvus_topk: Optional[int] = None,
        reranker_topk: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        对test_qa_pair_verify.json进行RAG预测，生成test_qa_pair_pred.json
        
        Args:
            test_verify_path: 验证集文件路径（test_qa_pair_verify.json）
            output_path: 输出文件路径（test_qa_pair_pred.json）
            bm25_topk: BM25检索数量（默认使用初始化值）
            milvus_topk: Milvus检索数量（默认使用初始化值）
            reranker_topk: Reranker重排序数量（默认使用初始化值）
            
        Returns:
            预测结果列表
            
        参考原项目：EVRAG/final_score.py:133-159
        """
        from ..tool_func import post_processing
        
        # 使用传入参数或默认值
        bm25_k = bm25_topk if bm25_topk is not None else self.bm25_topk
        milvus_k = milvus_topk if milvus_topk is not None else self.milvus_topk
        reranker_k = reranker_topk if reranker_topk is not None else self.reranker_topk
        
        # 加载验证集
        with open(test_verify_path, "r", encoding="utf-8") as f:
            test_qa_pairs = json.load(f)
        
        print(f"Loaded {len(test_qa_pairs)} verify QA pairs from {test_verify_path}")
        
        result = []
        
        # 处理每个QA对
        def process_qa_pair(qa_pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """处理单个QA对"""
            query = qa_pair.get("question", "").strip()
            if not query:
                return None
            
            # 重试机制：最多重试3次（针对连接错误）
            max_retries = 3
            import time
            
            for attempt in range(max_retries):
                try:
                    # RAG检索
                    bm25_docs = self.bm25_retriever.retrieve_topk(query, topk=bm25_k)
                    milvus_docs = self.milvus_retriever.retrieve_topk(query, topk=milvus_k)
                    merged_docs = merge_docs(bm25_docs, milvus_docs, mongodb_client=self.mongodb_client)
                    ranked_docs = self.reranker.rank(query, merged_docs, topk=reranker_k)
                    
                    # 构建上下文（用于保存）
                    context_str = "\n".join([
                        f"{idx+1}.{doc.page_content}" 
                        for idx, doc in enumerate(ranked_docs)
                    ])
                    
                    # LLM生成响应
                    response = self.chat_client.chat(query, context_str)
                    
                    # 清理response：去除思考过程和占位符
                    import re
                    # 1. 去除 <think> 标签及其内容（Qwen3思考模式输出）
                    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
                    # 2. 去除 "答案内容：" 前缀（模型可能输出的格式）
                    response = re.sub(r'^答案内容[：:]\s*', '', response, flags=re.MULTILINE)
                    # 3. 去除 {答案} 或 {从...} 占位符，保留实际答案
                    response = re.sub(r'\{答案\}', '', response)
                    response = re.sub(r'\{从[^}]*\}', '', response)
                    # 4. 去除多余的换行和空格
                    response = re.sub(r'\n\s*\n+', '\n', response).strip()
                    # 5. 如果response只包含引用标记没有实际答案，标记为"无答案"
                    if re.match(r'^[\s【】0-9,，]*$', response.strip()):
                        response = "无答案"
                    
                    # 后处理：提取答案、引用页面、相关图片
                    pred = post_processing(response, ranked_docs)
                    
                    # 构建结果
                    result_item = qa_pair.copy()
                    result_item["pred"] = pred
                    result_item["context"] = context_str
                    
                    return result_item
                except Exception as e:
                    error_str = str(e)
                    # 判断是否为连接错误
                    is_connection_error = (
                        "Connection" in error_str or 
                        "connection" in error_str.lower() or
                        "timeout" in error_str.lower() or
                        "refused" in error_str.lower()
                    )
                    
                    if is_connection_error and attempt < max_retries - 1:
                        # 连接错误，等待后重试（指数退避）
                        wait_time = min((attempt + 1) * 3, 10)  # 最多等待10秒
                        print(f"[WARNING] Connection error for query '{query[:50]}...': {error_str[:100]}")
                        print(f"         Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        # 非连接错误或最后一次重试失败
                        if is_connection_error:
                            print(f"[ERROR] Connection error for query '{query[:50]}...': {error_str[:100]}, failed after {max_retries} retries")
                        else:
                            print(f"[ERROR] Error processing query '{query[:50]}...': {error_str[:100]}")
                        return None
            
            return None
        
        # 并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                i: executor.submit(process_qa_pair, qa_pair)
                for i, qa_pair in enumerate(test_qa_pairs)
            }
            
            # 收集结果
            for i in tqdm(futures, desc="Generating test predictions"):
                future = futures[i]
                result_item = future.result()
                if result_item:
                    result.append(result_item)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print(f"Test prediction data saved to: {output_path} ({len(result)} items)")
        return result

