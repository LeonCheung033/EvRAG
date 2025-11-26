#!/usr/bin/env python3
"""
测试SFT数据生成
使用真实QA对测试本地模型的回复格式，并与原项目数据对比
"""

import json
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evrag.client import OpenAIClient, ChatClient, MongoDBClient
from src.evrag.retriever import BM25Retriever, MilvusRetriever
from src.evrag.reranker import BGEReranker, SiliconFlowReranker
from src.evrag.tool_func import merge_docs
from src.evrag.config import get_settings

def load_reference_data(reference_path: Path, num_samples: int = 3):
    """加载原项目的参考数据"""
    reference_samples = []
    with open(reference_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            if line.strip():
                try:
                    data = json.loads(line)
                    reference_samples.append(data)
                except:
                    continue
    return reference_samples

def test_single_qa(query: str, bm25_retriever, milvus_retriever, reranker, chat_client, mongodb_client=None):
    """测试单个QA对的生成"""
    print(f"\n{'='*80}")
    print(f"测试问题: {query}")
    print(f"{'='*80}")
    
    # RAG检索
    print("\n[1] 执行RAG检索...")
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=5)
    milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
    merged_docs = merge_docs(bm25_docs, milvus_docs, mongodb_client=mongodb_client)
    ranked_docs = reranker.rank(query, merged_docs, topk=5)
    
    print(f"  - BM25检索: {len(bm25_docs)} 个文档")
    print(f"  - Milvus检索: {len(milvus_docs)} 个文档")
    print(f"  - 合并后: {len(merged_docs)} 个文档")
    print(f"  - Reranker排序后: {len(ranked_docs)} 个文档")
    
    # 构建上下文
    context_str = "\n".join([
        f"{idx+1}.{doc.page_content}" 
        for idx, doc in enumerate(ranked_docs)
    ])
    
    print(f"\n[2] 上下文长度: {len(context_str)} 字符")
    print(f"    前200字符: {context_str[:200]}...")
    
    # LLM生成响应
    print("\n[3] 调用LLM生成响应...")
    response = chat_client.chat(query, context_str)
    
    print(f"\n[4] 原始响应:")
    print(f"{'─'*80}")
    print(response)
    print(f"{'─'*80}")
    
    # 分析响应格式
    print(f"\n[5] 响应格式分析:")
    print(f"  - 总长度: {len(response)} 字符")
    
    # 检查是否包含思考过程
    has_thinking = "<think>" in response.lower() or "<think>" in response.lower()
    print(f"  - 包含思考过程: {has_thinking}")
    
    # 检查引用标记
    import re
    cites = re.findall(r"[【](.*?)[】]", response)
    print(f"  - 引用标记数量: {len(cites)}")
    if cites:
        print(f"  - 引用标记: {cites}")
    
    # 提取答案（去除引用标记）
    answer = re.sub(r"[【](.*?)[】]", "", response)
    answer = re.sub(r"[{}【】]", "", answer).strip()
    print(f"  - 答案长度: {len(answer)} 字符")
    print(f"  - 答案预览: {answer[:100]}...")
    
    return {
        "query": query,
        "response": response,
        "context": [doc.page_content for doc in ranked_docs],
        "has_thinking": has_thinking,
        "cites": cites,
        "answer": answer,
    }

def main():
    """主函数"""
    print("="*80)
    print("SFT数据生成格式测试")
    print("="*80)
    
    # 加载参考数据
    reference_path = Path("/remote-home/share/liangZhang/EVRAG/data/qa_pairs/train_data.json")
    print(f"\n[加载参考数据] {reference_path}")
    reference_samples = load_reference_data(reference_path, num_samples=3)
    print(f"  加载了 {len(reference_samples)} 个参考样本")
    
    # 显示参考格式
    if reference_samples:
        print(f"\n[参考格式示例]")
        print(f"{'─'*80}")
        ref = reference_samples[0]
        print(f"Query: {ref.get('query', '')[:50]}...")
        print(f"Response长度: {len(ref.get('response', ''))} 字符")
        print(f"Response预览:")
        response_preview = ref.get('response', '')[:300]
        print(response_preview)
        if len(ref.get('response', '')) > 300:
            print("...")
        print(f"{'─'*80}")
    
    # 初始化组件
    print(f"\n[初始化组件]")
    settings = get_settings()
    
    # 使用 Deepseek API 生成问答数据（与生产环境一致）
    print("  - 初始化 LLM 客户端（Deepseek API）...")
    try:
        llm_client = OpenAIClient(service="deepseek")
        print("    ✓ Deepseek API 客户端初始化成功")
    except Exception as e:
        print(f"    ✗ Deepseek API 初始化失败: {e}")
        print("    请检查 config.yaml 中的 deepseek_api_key 配置")
        sys.exit(1)
    
    chat_client = ChatClient(llm_client)
    
    # 初始化检索器
    print("  - 初始化检索器...")
    bm25_retriever = BM25Retriever(docs=None, retrieve=True)
    milvus_retriever = MilvusRetriever(docs=None, retrieve=True)
    print("    ✓ 检索器初始化完成")
    
    # 初始化 Reranker（优先使用 SiliconFlow，否则使用 BGE）
    print("  - 初始化 Reranker...")
    reranker = None
    if settings.siliconflow_api_key:
        try:
            reranker = SiliconFlowReranker()
            print("    ✓ SiliconFlow Reranker 初始化成功")
        except Exception as e:
            print(f"    ⚠ SiliconFlow Reranker 初始化失败: {e}")
            print("    回退到 BGE Reranker...")
            reranker = BGEReranker()
            print("    ✓ BGE Reranker 初始化成功")
    else:
        print("    ⚠ SiliconFlow API key 未配置，使用 BGE Reranker")
        reranker = BGEReranker()
        print("    ✓ BGE Reranker 初始化成功")
    
    # 初始化 MongoDB 客户端（可选）
    mongodb_client = None
    try:
        if settings.mongodb_host and settings.mongodb_port:
            mongodb_client = MongoDBClient()
            print("    ✓ MongoDB 客户端初始化成功")
    except Exception as e:
        print(f"    ⚠ MongoDB 连接失败: {e}（继续执行，不使用 MongoDB）")
    
    print("  ✓ 所有组件初始化完成")
    
    # 从train_qa_pair.json加载测试问题
    train_qa_path = project_root / "data" / "qa_pairs" / "train_qa_pair.json"
    print(f"\n[加载测试问题] {train_qa_path}")
    
    test_queries = []
    try:
        with open(train_qa_path, "r", encoding="utf-8") as f:
            # 尝试作为JSON数组读取
            try:
                qa_pairs = json.load(f)
                if isinstance(qa_pairs, list):
                    for qa_pair in qa_pairs[:3]:  # 只测试3个问题
                        query = qa_pair.get("question", "").strip()
                        if query:
                            test_queries.append(query)
                else:
                    # 单个对象
                    query = qa_pairs.get("question", "").strip()
                    if query:
                        test_queries.append(query)
            except json.JSONDecodeError:
                # 如果失败，尝试作为JSONL读取
                f.seek(0)
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    if line.strip():
                        try:
                            qa_pair = json.loads(line)
                            query = qa_pair.get("question", "").strip()
                            if query:
                                test_queries.append(query)
                        except:
                            continue
    except Exception as e:
        print(f"  错误: 无法读取文件: {e}")
    
    print(f"  加载了 {len(test_queries)} 个测试问题")
    
    # 测试每个问题
    results = []
    for query in test_queries:
        try:
            result = test_single_qa(
                query, 
                bm25_retriever, 
                milvus_retriever, 
                reranker, 
                chat_client,
                mongodb_client
            )
            results.append(result)
        except Exception as e:
            print(f"\n[错误] 处理问题时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 对比分析
    print(f"\n{'='*80}")
    print("格式对比分析")
    print(f"{'='*80}")
    
    if reference_samples and results:
        print("\n[参考数据格式特征]")
        ref_responses = [r.get('response', '') for r in reference_samples]
        print(f"  - 平均响应长度: {sum(len(r) for r in ref_responses) / len(ref_responses):.0f} 字符")
        
        # 检查参考数据中的引用格式
        import re
        ref_cites = []
        for resp in ref_responses:
            cites = re.findall(r"[【](.*?)[】]", resp)
            ref_cites.extend(cites)
        print(f"  - 引用标记示例: {ref_cites[:3] if ref_cites else '无'}")
        
        print("\n[当前生成格式特征]")
        current_responses = [r['response'] for r in results]
        print(f"  - 平均响应长度: {sum(len(r) for r in current_responses) / len(current_responses):.0f} 字符")
        
        current_cites = []
        for resp in current_responses:
            cites = re.findall(r"[【](.*?)[】]", resp)
            current_cites.extend(cites)
        print(f"  - 引用标记示例: {current_cites[:3] if current_cites else '无'}")
        
        # 检查思考过程
        has_thinking_count = sum(1 for r in results if r['has_thinking'])
        print(f"  - 包含思考过程的数量: {has_thinking_count}/{len(results)}")
    
    # 保存测试结果
    output_path = project_root / "logs" / "sft_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[测试结果已保存] {output_path}")
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

