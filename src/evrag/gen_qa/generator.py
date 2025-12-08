"""
QA生成模块

从文档生成问答对，支持问题泛化、关键词提取和QA质量评分。
"""

import json
import re
import random
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langchain_core.documents import Document

from ..client import BaseLLMClient

# QA生成的prompt模板
CONTEXT_PROMPT_TPL = """
我会给你一段文本（<document></document>之间的部分），你需要阅读这段文本，分别针对这段文本生成5个问题，和基于这段文本对问题的回答，回答请保持完整，无须重复问题。

对问题、答案的要求：
1.问题：问题要与这段文本相关，不要询问类似"这个问题的答案在哪一章"这样的问题;
2.答案：回答请保持完整且简洁，无须重复问题。答案要能够独立回答问题，而不是引用其他章节和页码，例如答案内容不能出现请参阅xx页码;
3.5个问题里面至少要包含一个需要综合*大段*文本才能回答的问题，但不要问类似"这一段主要讲了什么内容"这样的问题;

对输出的要求：
1.返回结果以JSON形式组织，格式为[{{"question": "...", "answer": "..."}}, ...]。
2.如果当前文本主要是目录，或者是一些人名、地址、电子邮箱等没有办法生成有意义的问题时，可以返回[]。

下方是文本：
<document>
{{document}}
</document>

请生成结果：
"""

# 问题泛化的prompt模板
GENERALIZE_PROMPT_TPL = """
你是一个造句大师，请根据我输入的问题，生成具有意思相近的5个问题。

要求：
1.生成的问题要表达相近的意思，请探索采用不同的问法。
2.生成问题尽量口语化一点，可以不遵循原来的句式，例如：怎么打开车窗=》这个车的窗子要怎么才能开启
3.每一个问题用回车符连接，前面用序号开头，例如：1., 2., 3.

注意：不是回答问题，任务是输出5个同义句。

下方是输入的问题：
<question>
{{document}}
</question>

请生成结果：
"""

# 关键词提取的prompt模板
KEYWORDS_PROMPT_TPL = """
你是一名专业的汽车领域NLP工程师，任务是从给定的汽车行业文本中提取核心关键词。请按以下要求操作：

抽取原则：
1.优先提取汽车专业术语（如"涡轮增压"、"ADAS系统"）
2.保留产品型号/规格（如"MQB平台"、"2023款Model Y"）
3.包含技术特征（如"L2级自动驾驶"、"48V轻混"）
4.提取关键动作（如"召回"、"OTA升级"）

输出要求：
1.重点关注：动力总成/车身结构/汽车零部件/智能网联/辅助驾驶/新能源技术/充电设施/售后服务
2.过滤通用词汇（如"使用"、"包括"）
3.请输出最重要的关键词，关键词数量不得超过5个
4.如果没有关键词，请直接输出"无"

输出格式：
关键词列表，用逗号分隔
例如：行车记录仪,探测功能,辅助驾驶,车辆功率

下方是输入的问题：
<question>
{{document}}
</question>

请生成结果：
"""

# QA质量评分的prompt模板
QA_QUALITY_PROMPT_TPL = """
你是一个汽车领域的专家，现在有人根据一份汽车用车手册，构造了一些问题，并对问题进行了回答。
你的任务是对这些问题（<question></question>之间的部分）和回答（<answer></answer>）进行打分。

结果请以JSON形式组织，格式如下（<result></result>之间的部分）：
<result>
{{"score": ..., "reason": ...}}
</result>
其中score是对问题-回答的打分，分值是一个int类型的值，取值范围为1-5。reason是打分的理由。

好的问题，应该是询问事实、观点等，不好的问题，通常要求做一些文本摘要等初级文字处理工作，类似于"这一段描述了什么"，"文本描述了什么"；或者询问的内容是图相关的，例如"图4展示了什么数据？"。
好的答案，应该能够回应问题，而不是回答无关的内容，不好的回答，会给出在原文中的引用，例如"第3章"等。

问题：
<question>
{{question}}
</question>

答案：
<answer>
{{answer}}
</answer>

请进返回JSON格式的数据即可，不要添加其他任何描述性信息。
"""


class QAGenerator:
    """
    QA生成器

    从文档生成问答对，支持问题泛化、关键词提取等功能。
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        min_chunk_size: int = 100,
        max_workers: int = 20,
        max_retry: int = 3,
        seed: int = 42,
    ):
        """
        初始化QA生成器

        Args:
            llm_client: LLM客户端实例
            min_chunk_size: 最小文档块大小（字符数）
            max_workers: 最大并发工作线程数
            max_retry: 最大重试次数
            seed: 随机种子
        """
        self.llm_client = llm_client
        self.min_chunk_size = min_chunk_size
        self.max_workers = max_workers
        self.max_retry = max_retry
        random.seed(seed)

    def _build_prompt(self, template: str, text: str) -> str:
        """
        构建prompt

        Args:
            template: prompt模板
            text: 要填充的文本

        Returns:
            构建好的prompt
        """
        return (
            template.replace("{{document}}", text)
            .replace("{{question}}", text)
            .replace("{{answer}}", text)
            .strip()
        )

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.85,
        top_p: float = 0.95,
    ) -> Optional[str]:
        """
        调用LLM生成内容（带重试机制）

        Args:
            prompt: 输入prompt
            temperature: 温度参数
            top_p: top_p参数

        Returns:
            LLM返回的内容，失败时返回None
        """
        messages = [
            {"role": "system", "content": "你是一个有用的人工智能助手."},
            {"role": "user", "content": prompt},
        ]

        for attempt in range(self.max_retry):
            try:
                result = self.llm_client.chat(
                    messages=messages,
                    model=None,  # 使用客户端默认模型
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                )
                return result
            except Exception as e:
                error_str = str(e)
                # 如果是400错误且提示等待，增加等待时间
                if "400" in error_str and "wait" in error_str.lower():
                    sleep_seconds = 60  # 等待1分钟
                    print(
                        f"Error: {error_str}, API rate limit detected, waiting {sleep_seconds}s..."
                    )
                elif attempt < self.max_retry - 1:
                    sleep_seconds = random.randint(2, 5)  # 增加重试间隔
                    print(
                        f"Error: {error_str}, remain retry: {self.max_retry - attempt - 1}, sleeping {sleep_seconds}s"
                    )
                else:
                    print(f"Failed after {self.max_retry} retries: {error_str}")
                    return None

                import time

                time.sleep(sleep_seconds)
        return None

    def generate_qa_from_documents(
        self,
        documents: List[Document],
        output_file: Optional[Path] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        从文档生成QA对

        Args:
            documents: 文档列表
            output_file: 输出文件路径（可选，用于保存checkpoint）
            checkpoint: 已有的checkpoint（可选，用于跳过已处理的文档）

        Returns:
            QA对字典，key为unique_id，value为包含unique_id和raw_resp的字典
        """
        qa_checkpoint = checkpoint or {}
        file_lock = threading.Lock()

        # 统计信息
        stats = {
            "total_docs": len(documents),
            "duplicate_unique_ids": 0,
            "missing_unique_ids": 0,
            "too_short": 0,
            "in_checkpoint": len(qa_checkpoint),
            "processed": 0,
            "failed": 0,
        }

        # 检查重复的unique_id
        unique_id_count = {}
        for doc in documents:
            unique_id = doc.metadata.get("unique_id")
            if unique_id:
                unique_id_count[unique_id] = unique_id_count.get(unique_id, 0) + 1
            else:
                stats["missing_unique_ids"] += 1

        # 统计重复的unique_id数量
        duplicate_ids = {
            uid: count for uid, count in unique_id_count.items() if count > 1
        }
        stats["duplicate_unique_ids"] = sum(
            count - 1 for count in duplicate_ids.values()
        )

        if duplicate_ids:
            print(
                f"Warning: Found {len(duplicate_ids)} duplicate unique_ids, {stats['duplicate_unique_ids']} documents will be skipped"
            )

        def process_doc(doc: Document) -> Optional[Dict[str, Any]]:
            """处理单个文档"""
            unique_id = doc.metadata.get("unique_id")
            if not unique_id:
                stats["missing_unique_ids"] += 1
                return None

            if unique_id in qa_checkpoint:
                return None

            # 过滤太短的文档
            content_length = len(doc.page_content.replace("\n", ""))
            if content_length < self.min_chunk_size:
                stats["too_short"] += 1
                return None

            prompt = self._build_prompt(CONTEXT_PROMPT_TPL, doc.page_content)
            result = self._call_llm(prompt, temperature=0.85, top_p=0.95)

            if result is None:
                stats["failed"] += 1
                return None

            # 清理响应中的 markdown 代码块标记
            cleaned_resp = self.clean_markdown_code_blocks(result)

            stats["processed"] += 1
            item = {"unique_id": unique_id, "raw_resp": cleaned_resp}

            # 如果指定了输出文件，追加写入
            if output_file:
                file_lock.acquire()
                try:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error writing to file: {e}")
                finally:
                    file_lock.release()

            return item

        # 使用文档索引作为key，避免重复unique_id覆盖问题
        # 但最终结果仍以unique_id为key（如果unique_id重复，后面的会覆盖前面的）
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                i: executor.submit(process_doc, doc) for i, doc in enumerate(documents)
            }

            for idx in tqdm(futures, desc="Generating QA pairs"):
                future = futures[idx]
                result = future.result()
                if result:
                    unique_id = result["unique_id"]
                    qa_checkpoint[unique_id] = result

        # 打印统计信息
        print("\nQA Generation Statistics:")
        print(f"  - Total documents: {stats['total_docs']}")
        print(f"  - Unique unique_ids: {len(unique_id_count)}")
        print(
            f"  - Duplicate unique_ids: {len(duplicate_ids)} ({stats['duplicate_unique_ids']} docs skipped)"
        )
        print(f"  - Missing unique_ids: {stats['missing_unique_ids']}")
        print(f"  - Too short (<{self.min_chunk_size} chars): {stats['too_short']}")
        print(f"  - In checkpoint (skipped): {stats['in_checkpoint']}")
        print(f"  - Successfully processed: {stats['processed']}")
        print(f"  - Failed: {stats['failed']}")

        return qa_checkpoint

    def generalize_questions(
        self,
        questions: List[str],
        output_file: Optional[Path] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        问题泛化：为每个问题生成同义问题

        Args:
            questions: 问题列表
            output_file: 输出文件路径（可选）

        Returns:
            泛化结果字典，key为问题内容，value为包含unique_id和raw_resp的字典
        """
        question_docs = [
            Document(page_content=q, metadata={"unique_id": str(i)})
            for i, q in enumerate(questions)
        ]

        result_dict = {}
        file_lock = threading.Lock()

        def process_question(doc: Document) -> Optional[Dict[str, Any]]:
            """处理单个问题"""
            prompt = self._build_prompt(GENERALIZE_PROMPT_TPL, doc.page_content)
            result = self._call_llm(prompt, temperature=0.85, top_p=0.95)

            if result is None:
                return None

            item = {"unique_id": doc.page_content, "raw_resp": result}

            if output_file:
                file_lock.acquire()
                try:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error writing to file: {e}")
                finally:
                    file_lock.release()

            return item

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                doc.page_content: executor.submit(process_question, doc)
                for doc in question_docs
            }

            for question in tqdm(futures, desc="Generalizing questions"):
                future = futures[question]
                result = future.result()
                if result:
                    result_dict[question] = result

        return result_dict

    def extract_keywords(
        self,
        texts: List[str],
        output_file: Optional[Path] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        从文本中提取关键词

        Args:
            texts: 文本列表
            output_file: 输出文件路径（可选）

        Returns:
            关键词提取结果字典
        """
        text_docs = [
            Document(page_content=text, metadata={"unique_id": str(i)})
            for i, text in enumerate(texts)
        ]

        result_dict = {}
        file_lock = threading.Lock()

        def process_text(doc: Document) -> Optional[Dict[str, Any]]:
            """处理单个文本"""
            prompt = self._build_prompt(KEYWORDS_PROMPT_TPL, doc.page_content)
            result = self._call_llm(
                prompt, temperature=0.001, top_p=0.1
            )  # top_p不能为0，改为0.1

            if result is None:
                return None

            item = {"unique_id": doc.page_content, "raw_resp": result}

            if output_file:
                file_lock.acquire()
                try:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"Error writing to file: {e}")
                finally:
                    file_lock.release()

            return item

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                doc.page_content: executor.submit(process_text, doc)
                for doc in text_docs
            }

            for text in tqdm(futures, desc="Extracting keywords"):
                future = futures[text]
                result = future.result()
                if result:
                    result_dict[text] = result

        return result_dict

    def score_qa_quality(
        self,
        question: str,
        answer: str,
    ) -> Optional[Dict[str, Any]]:
        """
        对QA对进行质量评分

        Args:
            question: 问题
            answer: 答案

        Returns:
            评分结果，包含score和reason，失败时返回None
        """
        prompt = (
            QA_QUALITY_PROMPT_TPL.replace("{{question}}", question)
            .replace("{{answer}}", answer)
            .strip()
        )
        result = self._call_llm(
            prompt, temperature=0.001, top_p=0.1
        )  # top_p不能为0，改为0.1

        if result is None:
            return None

        try:
            # 尝试从结果中提取JSON
            # 可能包含<result>标签或其他格式
            json_match = re.search(r"\{[^}]+\}", result)
            if json_match:
                score_data = json.loads(json_match.group())
                return score_data
            else:
                # 如果找不到JSON，尝试直接解析
                score_data = json.loads(result)
                return score_data
        except json.JSONDecodeError as e:
            print(f"Failed to parse QA quality score: {e}, result: {result}")
            return None

    @staticmethod
    def clean_markdown_code_blocks(text: str) -> str:
        """
        清理文本中的 markdown 代码块标记

        Args:
            text: 包含可能包含 markdown 代码块标记的文本

        Returns:
            清理后的文本
        """
        if not text or not isinstance(text, str):
            return text

        cleaned = text.strip()

        # 移除 ```json 和 ``` 代码块标记
        if cleaned.startswith("```json"):
            # 移除开头的 ```json
            cleaned = cleaned[7:].strip()
        elif cleaned.startswith("```"):
            # 移除开头的 ```
            cleaned = cleaned[3:].strip()

        # 移除结尾的 ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        return cleaned

    @staticmethod
    def parse_qa_response(raw_resp: str) -> List[Dict[str, str]]:
        """
        解析QA生成的响应

        Args:
            raw_resp: LLM返回的原始响应（可能已清理或未清理）

        Returns:
            QA对列表，每个元素包含question和answer
        """
        if not raw_resp or not isinstance(raw_resp, str):
            return []

        # 先清理响应，移除代码块标记
        cleaned = QAGenerator.clean_markdown_code_blocks(raw_resp)

        # 尝试解析JSON
        try:
            # 尝试直接解析清理后的JSON
            qa_list = json.loads(cleaned)
            if isinstance(qa_list, list):
                # 过滤掉非字典元素，确保每个元素都是字典
                return [
                    qa
                    for qa in qa_list
                    if isinstance(qa, dict) and "question" in qa and "answer" in qa
                ]
            else:
                return []
        except json.JSONDecodeError:
            # 如果解析失败，尝试用正则表达式提取JSON数组
            json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if json_match:
                try:
                    qa_list = json.loads(json_match.group())
                    if isinstance(qa_list, list):
                        # 过滤掉非字典元素，确保每个元素都是字典
                        return [
                            qa
                            for qa in qa_list
                            if isinstance(qa, dict)
                            and "question" in qa
                            and "answer" in qa
                        ]
                    else:
                        return []
                except json.JSONDecodeError:
                    pass
            return []

    @staticmethod
    def parse_generalized_questions(raw_resp: str) -> List[str]:
        """
        解析问题泛化的响应

        Args:
            raw_resp: LLM返回的原始响应

        Returns:
            泛化后的问题列表
        """
        questions = raw_resp.split("\n")
        # 移除序号前缀（支持多种格式：1. 1) 1、等）
        # 匹配：数字 + 点号/右括号/空格 + 可选的空格
        questions = [re.sub(r"^\d+[.)\s]+\s*", "", item).strip() for item in questions]
        # 过滤空字符串
        questions = [q for q in questions if q]
        return questions

    @staticmethod
    def parse_keywords(raw_resp: str) -> List[str]:
        """
        解析关键词提取的响应

        Args:
            raw_resp: LLM返回的原始响应

        Returns:
            关键词列表
        """
        keywords = [k.strip() for k in raw_resp.split(",")]
        # 过滤无效关键词
        keywords = [k for k in keywords if k and k not in ["无", "Model 3"]]
        return keywords
