"""
QA数据处理模块

从已生成的qa_pair.json开始，经过质量打分、过滤、问题改写、数据扩充、
训练/测试集切分，最终生成train_qa_pair.json和test_qa_pair.json。

参考原项目：EVRAG/src/gen_qa/run.py
"""

import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .generator import QAGenerator


class QAProcessor:
    """
    QA数据处理器

    整合QA质量打分、问题改写、训练/测试集切分、关键词提取和负样本添加等功能。
    """

    def __init__(
        self,
        qa_generator: QAGenerator,
        quality_threshold: int = 3,
        train_ratio: float = 0.9,
        negative_samples_train_ratio: float = 0.95,
        seed: int = 42,
    ):
        """
        初始化QA处理器

        Args:
            qa_generator: QAGenerator实例
            quality_threshold: QA质量打分阈值（默认：3）
            train_ratio: 训练集比例（默认：0.9）
            negative_samples_train_ratio: 负样本训练集比例（默认：0.95）
            seed: 随机种子（默认：42）
        """
        self.qa_generator = qa_generator
        self.quality_threshold = quality_threshold
        self.train_ratio = train_ratio
        self.negative_samples_train_ratio = negative_samples_train_ratio
        random.seed(seed)

    def load_qa_pairs(self, qa_pair_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        加载qa_pair.json文件

        Args:
            qa_pair_path: qa_pair.json文件路径

        Returns:
            QA对字典，key为unique_id，value为包含unique_id和raw_resp的字典

        参考原项目：EVRAG/src/gen_qa/run.py:226-230
        """
        qa_dict = {}
        if not qa_pair_path.exists():
            print(f"Warning: {qa_pair_path} does not exist")
            return qa_dict

        with open(qa_pair_path, "r", encoding="utf-8") as fd:
            for line in fd:
                if not line.strip():
                    continue
                try:
                    info = json.loads(line)
                    unique_id = info.get("unique_id")
                    if unique_id:
                        qa_dict[unique_id] = info
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}, line: {line[:100]}")
                    continue

        print(f"Loaded {len(qa_dict)} QA pairs from {qa_pair_path}")
        return qa_dict

    def score_and_filter_qa_pairs(
        self,
        qa_dict: Dict[str, Dict[str, Any]],
        skip_scoring: bool = False,
    ) -> List[Dict[str, str]]:
        """
        QA质量打分和过滤

        Args:
            qa_dict: QA对字典
            skip_scoring: 是否跳过质量打分（如果已打分）

        Returns:
            过滤后的QA对列表，每个元素包含question和answer

        参考原项目：EVRAG/src/gen_qa/run.py:253-254
        """
        filtered_qa_pairs = []

        # 提取所有QA对
        all_qa_pairs = []
        for unique_id, info in qa_dict.items():
            raw_resp = info.get("raw_resp", "[]")
            try:
                qa_list = self.qa_generator.parse_qa_response(raw_resp)
                for qa in qa_list:
                    question = qa.get("question", "").strip()
                    answer = qa.get("answer", "").strip()
                    if question and answer:
                        all_qa_pairs.append(
                            {
                                "question": question,
                                "answer": answer,
                                "original_unique_id": unique_id,
                            }
                        )
            except Exception as e:
                print(f"Error parsing QA pair for unique_id {unique_id}: {e}")
                continue

        print(f"Extracted {len(all_qa_pairs)} QA pairs from {len(qa_dict)} documents")

        # 过滤包含"无法准确"或"未提及"的答案（与原项目一致）
        filtered_qa_pairs = []
        for qa in all_qa_pairs:
            answer = qa["answer"]
            if "无法准确" in answer or "未提及" in answer:
                continue
            filtered_qa_pairs.append(
                {
                    "question": qa["question"],
                    "answer": qa["answer"],
                }
            )

        print(f"After filtering '无法准确'/'未提及': {len(filtered_qa_pairs)} QA pairs")

        # QA质量打分（如果未跳过）
        if not skip_scoring:
            scored_qa_pairs = []
            print(f"Scoring QA quality (threshold: {self.quality_threshold})...")

            def score_qa(qa: Dict[str, str]) -> Optional[Dict[str, Any]]:
                """对单个QA对进行质量打分"""
                question = qa["question"]
                answer = qa["answer"]
                score_result = self.qa_generator.score_qa_quality(question, answer)

                if score_result is None:
                    return None

                score = score_result.get("score", 0)
                if isinstance(score, str):
                    try:
                        score = int(score)
                    except ValueError:
                        return None

                if score >= self.quality_threshold:
                    # 将打分结果添加到QA对中
                    scored_qa = qa.copy()
                    scored_qa["quality_score"] = score
                    scored_qa["quality_reason"] = score_result.get("reason", "")
                    return scored_qa
                return None

            # 并发打分（降低并发数避免API限流）
            max_workers = min(20, self.qa_generator.max_workers)  # 限制最大并发数为5
            import time

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    i: executor.submit(score_qa, qa)
                    for i, qa in enumerate(filtered_qa_pairs)
                }

                for i in tqdm(futures, desc="Scoring QA pairs"):
                    future = futures[i]
                    try:
                        result = future.result()
                        if result:
                            scored_qa_pairs.append(result)
                        # 添加小延迟避免API限流
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Error scoring QA pair {i}: {e}")
                        continue

            print(f"After quality scoring: {len(scored_qa_pairs)} QA pairs")
            return scored_qa_pairs
        else:
            print("Skipping quality scoring")
            return filtered_qa_pairs

    def load_filtered_qa_pairs(self, filtered_path: Path) -> List[Dict[str, Any]]:
        """
        从filtered_qa_pair.json加载过滤后的QA对

        Args:
            filtered_path: filtered_qa_pair.json文件路径

        Returns:
            过滤后的QA对列表
        """
        if not filtered_path.exists():
            raise FileNotFoundError(
                f"Filtered QA pairs file not found: {filtered_path}"
            )

        with open(filtered_path, "r", encoding="utf-8") as fd:
            filtered_qa_pairs = json.load(fd)

        print(f"Loaded {len(filtered_qa_pairs)} filtered QA pairs from {filtered_path}")
        return filtered_qa_pairs

    def generalize_and_expand_questions(
        self,
        filtered_qa_pairs: List[Dict[str, Any]],
        output_file: Path,
        skip_rewriting: bool = False,
    ) -> Dict[str, List[str]]:
        """
        问题改写并生成expand_qa_pair.json（合并generalize和expand步骤）

        Args:
            filtered_qa_pairs: 过滤后的QA对列表
            output_file: 输出文件路径（expand_qa_pair.json）
            skip_rewriting: 是否跳过问题改写（如果已改写）

        Returns:
            问题到改写问题列表的映射

        参考原项目：EVRAG/src/gen_qa/run.py:213-243
        """
        if skip_rewriting and output_file.exists():
            # 从已有文件加载
            expand_qa_pairs = {}
            with open(output_file, "r", encoding="utf-8") as fd:
                for line in fd:
                    if not line.strip():
                        continue
                    try:
                        info = json.loads(line)
                        question = info.get("unique_id", "")
                        raw_resp = info.get("raw_resp", "")
                        if question and raw_resp:
                            expand_questions = (
                                self.qa_generator.parse_generalized_questions(raw_resp)
                            )
                            expand_qa_pairs[question] = expand_questions
                    except Exception as e:
                        print(f"Error parsing expand_qa_pair line: {e}")
                        continue
            print(
                f"Loaded {len(expand_qa_pairs)} expanded questions from {output_file}"
            )
            return expand_qa_pairs

        # 提取问题列表
        questions = [qa["question"] for qa in filtered_qa_pairs]

        # 生成问题改写
        print(f"Generalizing {len(questions)} questions...")
        expand_result = self.qa_generator.generalize_questions(
            questions=questions,
            output_file=output_file,
        )

        # 解析改写结果
        expand_qa_pairs = {}
        for question, info in expand_result.items():
            raw_resp = info.get("raw_resp", "")
            if raw_resp:
                expand_questions = self.qa_generator.parse_generalized_questions(
                    raw_resp
                )
                expand_qa_pairs[question] = expand_questions

        print(f"Generated expanded questions for {len(expand_qa_pairs)} questions")
        print(f"Saved to: {output_file}")
        return expand_qa_pairs

    def expand_qa_pairs(
        self,
        qa_pairs: List[Dict[str, str]],
        expand_qa_pairs: Dict[str, List[str]],
    ) -> List[Dict[str, str]]:
        """
        答案匹配和扩充

        Args:
            qa_pairs: 原始QA对列表
            expand_qa_pairs: 问题到改写问题列表的映射

        Returns:
            扩充后的QA对列表

        参考原项目：EVRAG/src/gen_qa/run.py:246-266
        """
        expanded_qa_pairs = []

        for qa in qa_pairs:
            question = qa["question"]
            answer = qa["answer"]

            # 获取改写问题列表（如果存在）
            expand_questions = expand_qa_pairs.get(question, [])

            # 合并原始问题和改写问题
            all_questions = [question] + expand_questions

            # 为每个问题生成QA对
            for query in all_questions:
                unique_id = hashlib.md5(query.encode("utf-8")).hexdigest()
                item = {
                    "unique_id": unique_id,
                    "question": query,
                    "answer": answer,
                }
                expanded_qa_pairs.append(item)

        print(f"Expanded {len(qa_pairs)} QA pairs to {len(expanded_qa_pairs)} QA pairs")
        return expanded_qa_pairs

    def split_train_test(
        self,
        qa_pairs: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        训练/测试集切分

        Args:
            qa_pairs: QA对列表

        Returns:
            (训练集, 测试集) 元组

        参考原项目：EVRAG/src/gen_qa/run.py:263-266
        """
        train_qa_pairs = []
        test_qa_pairs = []

        for item in qa_pairs:
            if random.random() < self.train_ratio:
                train_qa_pairs.append(item)
            else:
                test_qa_pairs.append(item)

        print(f"Split into train: {len(train_qa_pairs)}, test: {len(test_qa_pairs)}")
        return train_qa_pairs, test_qa_pairs

    def extract_keywords_for_test_set(
        self,
        test_qa_pairs: List[Dict[str, str]],
        output_file: Optional[Path] = None,
    ) -> Dict[str, List[str]]:
        """
        为测试集提取关键词

        Args:
            test_qa_pairs: 测试集QA对列表
            output_file: 输出文件路径（test_keywords_pair.json）

        Returns:
            答案到关键词列表的映射

        参考原项目：EVRAG/src/gen_qa/run.py:271-288
        注意：修正原项目bug（变量名kewyords应该是keywords）
        """
        # 提取唯一的答案
        unique_test_answers = list(set([item["answer"] for item in test_qa_pairs]))
        print(f"Extracting keywords for {len(unique_test_answers)} unique answers...")

        # 提取关键词
        keywords_result = self.qa_generator.extract_keywords(
            texts=unique_test_answers,
            output_file=output_file,
        )

        # 建立答案到关键词的映射
        keywords_mapping = {}
        for answer, info in keywords_result.items():
            raw_resp = info.get("raw_resp", "")
            if raw_resp:
                # 使用parse_keywords方法解析关键词（已包含过滤逻辑）
                keywords = self.qa_generator.parse_keywords(raw_resp)
                keywords_mapping[answer] = keywords

        print(f"Extracted keywords for {len(keywords_mapping)} answers")

        # 将关键词添加到测试集的每个QA对中
        for item in test_qa_pairs:
            answer = item["answer"]
            keywords = keywords_mapping.get(answer, [])
            item["keywords"] = keywords

        return keywords_mapping

    def add_negative_samples(
        self,
        train_qa_pairs: List[Dict[str, str]],
        test_qa_pairs: List[Dict[str, str]],
        negative_samples_path: Path,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        添加负样本

        Args:
            train_qa_pairs: 训练集QA对列表
            test_qa_pairs: 测试集QA对列表
            negative_samples_path: 负样本文件路径（raw_general_chats.txt）

        Returns:
            (添加负样本后的训练集, 添加负样本后的测试集) 元组

        参考原项目：EVRAG/src/gen_qa/run.py:291-311
        """
        if not negative_samples_path.exists():
            print(
                f"Warning: {negative_samples_path} does not exist, skipping negative samples"
            )
            return train_qa_pairs, test_qa_pairs

        # 读取负样本
        with open(negative_samples_path, "r", encoding="utf-8") as f:
            chats_data = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(chats_data)} negative samples from {negative_samples_path}")

        # 添加负样本
        random.seed(42)  # 与原项目保持一致
        for line in chats_data:
            unique_id = hashlib.md5(line.encode("utf-8")).hexdigest()

            if random.random() < self.negative_samples_train_ratio:
                # 加入训练集
                train_qa_pairs.append(
                    {
                        "unique_id": unique_id,
                        "question": line,
                        "answer": "无答案",
                    }
                )
            else:
                # 加入测试集
                test_qa_pairs.append(
                    {
                        "unique_id": unique_id,
                        "question": line,
                        "answer": "无答案",
                        "keywords": [],
                    }
                )

        print(
            f"Added negative samples: train: {len([q for q in train_qa_pairs if q['answer'] == '无答案'])}, "
            f"test: {len([q for q in test_qa_pairs if q['answer'] == '无答案'])}"
        )

        return train_qa_pairs, test_qa_pairs

    def save_final_data(
        self,
        train_qa_pairs: List[Dict[str, str]],
        test_qa_pairs: List[Dict[str, str]],
        train_path: Path,
        test_path: Path,
    ) -> None:
        """
        保存最终数据

        Args:
            train_qa_pairs: 训练集QA对列表
            test_qa_pairs: 测试集QA对列表
            train_path: 训练集输出路径（train_qa_pair.json）
            test_path: 测试集输出路径（test_qa_pair.json）

        参考原项目：EVRAG/src/gen_qa/run.py:313-323
        """
        # 确保输出目录存在
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        # 打乱数据（使用固定随机种子）
        random.seed(42)
        random.shuffle(train_qa_pairs)
        random.seed(42)
        random.shuffle(test_qa_pairs)

        # 保存训练集
        with open(train_path, "w", encoding="utf-8") as fd:
            json.dump(train_qa_pairs, fd, ensure_ascii=False, indent=2)
        print(f"训练集已写入: {train_path}, {len(train_qa_pairs)} 条")

        # 保存测试集
        with open(test_path, "w", encoding="utf-8") as fd:
            json.dump(test_qa_pairs, fd, ensure_ascii=False, indent=2)
        print(f"测试集已写入: {test_path}, {len(test_qa_pairs)} 条")

    def save_filtered_qa_pairs(
        self,
        filtered_qa_pairs: List[Dict[str, str]],
        output_path: Path,
    ) -> None:
        """
        保存过滤后的QA对

        Args:
            filtered_qa_pairs: 过滤后的QA对列表
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为JSON数组格式
        with open(output_path, "w", encoding="utf-8") as fd:
            json.dump(filtered_qa_pairs, fd, ensure_ascii=False, indent=2)
        print(f"过滤后的QA对已写入: {output_path}, {len(filtered_qa_pairs)} 条")

    def process(
        self,
        qa_pair_path: Path,
        output_dir: Path,
        negative_samples_path: Path,
        skip_quality_scoring: bool = False,
        skip_question_rewriting: bool = False,
    ) -> None:
        """
        完整的QA处理流程

        Args:
            qa_pair_path: qa_pair.json文件路径
            output_dir: 输出目录
            negative_samples_path: 负样本文件路径
            skip_quality_scoring: 是否跳过质量打分
            skip_question_rewriting: 是否跳过问题改写

        参考原项目：EVRAG/src/gen_qa/run.py:206-323
        """
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        # 定义输出文件路径
        expand_qa_pair_path = output_dir / "expand_qa_pair.json"
        train_path = output_dir / "train_qa_pair.json"
        test_path = output_dir / "test_qa_pair.json"
        test_keywords_path = output_dir / "test_keywords_pair.json"

        print("=" * 60)
        print("QA Data Processing Pipeline")
        print("=" * 60)

        # 步骤1: 加载QA对
        print("\n[Step 1] Loading QA pairs...")
        qa_dict = self.load_qa_pairs(qa_pair_path)
        if not qa_dict:
            print("Error: No QA pairs loaded")
            return

        # 步骤2: QA质量打分和过滤
        print("\n[Step 2] Scoring and filtering QA pairs...")
        filtered_qa_pairs = self.score_and_filter_qa_pairs(
            qa_dict=qa_dict,
            skip_scoring=skip_quality_scoring,
        )
        if not filtered_qa_pairs:
            print("Error: No QA pairs after filtering")
            return

        # 步骤3: 问题改写
        print("\n[Step 3] Generalizing questions...")
        questions = [qa["question"] for qa in filtered_qa_pairs]
        expand_qa_pairs = self.generalize_questions(
            questions=questions,
            output_file=expand_qa_pair_path,
            skip_rewriting=skip_question_rewriting,
        )

        # 步骤4: 答案匹配和扩充
        print("\n[Step 4] Expanding QA pairs...")
        expanded_qa_pairs = self.expand_qa_pairs(
            qa_pairs=filtered_qa_pairs,
            expand_qa_pairs=expand_qa_pairs,
        )

        # 步骤5: 训练/测试集切分
        print("\n[Step 5] Splitting train/test sets...")
        train_qa_pairs, test_qa_pairs = self.split_train_test(expanded_qa_pairs)

        # 步骤6: 测试集关键词提取
        print("\n[Step 6] Extracting keywords for test set...")
        keywords_mapping = self.extract_keywords_for_test_set(
            test_qa_pairs=test_qa_pairs,
            output_file=test_keywords_path,
        )

        # 步骤7: 添加负样本
        print("\n[Step 7] Adding negative samples...")
        train_qa_pairs, test_qa_pairs = self.add_negative_samples(
            train_qa_pairs=train_qa_pairs,
            test_qa_pairs=test_qa_pairs,
            negative_samples_path=negative_samples_path,
        )

        # 步骤8: 保存最终数据
        print("\n[Step 8] Saving final data...")
        self.save_final_data(
            train_qa_pairs=train_qa_pairs,
            test_qa_pairs=test_qa_pairs,
            train_path=train_path,
            test_path=test_path,
        )

        print("\n" + "=" * 60)
        print("QA Data Processing Completed!")
        print("=" * 60)
        print(f"Train set: {len(train_qa_pairs)} QA pairs")
        print(f"Test set: {len(test_qa_pairs)} QA pairs")
        print(f"Output directory: {output_dir}")
