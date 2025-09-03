#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一迭代管理器模块

该模块提供统一的迭代状态管理功能，解决工作流中迭代计数混乱、
状态管理分散和职责不清的问题。

主要功能:
1. 统一的迭代计数管理
2. 阶段状态跟踪
3. 迭代验证和限制
4. 状态同步和一致性保证
5. 调试信息和日志记录


"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, asdict
from enum import Enum


class WorkflowStage(Enum):
    """工作流阶段枚举"""
    INITIALIZED = "initialized"
    INTERPRET = "interpret_source"
    DRAFT_CONTENT = "draft_content"
    MACRO_REVIEW = "macro_review"
    MICRO_REVIEW = "micro_review"
    REFINE_CONTENT = "refine_content"
    DRAFT_TITLE = "draft_title"
    REVIEW_TITLE = "review_title"
    REFINE_TITLE = "refine_title"
    COMPLETED = "completed"


class IterationType(Enum):
    """迭代类型枚举"""
    MACRO = "macro"
    MICRO = "micro"
    TITLE = "title"


@dataclass
class IterationRecord:
    """迭代记录数据类"""
    timestamp: str
    stage: WorkflowStage
    iteration_type: IterationType
    iteration_count: int
    max_iterations: int
    is_passed: Optional[bool] = None
    feedback_length: Optional[int] = None
    context_info: Optional[Dict] = None


class IterationManager:
    """
    统一迭代管理器
    
    负责管理整个工作流中的迭代状态，包括宏观审查、微观审查和标题审查的
    迭代计数、状态跟踪和验证。确保迭代逻辑的一致性和可靠性。
    
    主要功能:
    1. 统一的迭代计数管理
    2. 阶段状态跟踪和转换
    3. 迭代限制和验证
    4. 状态同步和一致性保证
    5. 详细的调试信息和历史记录
    """
    
    def __init__(self, max_macro_iterations: int = 5, max_micro_iterations: int = 5, max_title_iterations: int = 3):
        """
        初始化迭代管理器
        
        Args:
            max_macro_iterations (int): 宏观审查最大迭代次数，默认3次
            max_micro_iterations (int): 微观审查最大迭代次数，默认3次
            max_title_iterations (int): 标题审查最大迭代次数，默认3次
        """
        # 迭代限制配置
        self.max_macro_iterations = max_macro_iterations
        self.max_micro_iterations = max_micro_iterations
        self.max_title_iterations = max_title_iterations
        
        # 当前迭代计数
        self.macro_iteration_count = 0
        self.micro_iteration_count = 0
        self.title_iteration_count = 0
        
        # 当前状态
        self.current_stage = WorkflowStage.INITIALIZED
        self.current_iteration_type: Optional[IterationType] = None
        
        # 状态标志
        self.macro_review_passed = False
        self.micro_review_passed = False
        self.title_review_passed = False
        
        # 历史记录
        self.iteration_history: List[IterationRecord] = []
        
        # 会话信息
        self.session_start_time = datetime.now().isoformat()
        self.last_update_time = self.session_start_time
        
        print(f"[迭代管理器] 初始化完成 - 宏观:{max_macro_iterations}轮, 微观:{max_micro_iterations}轮, 标题:{max_title_iterations}轮")
    
    def start_macro_iteration(self) -> Tuple[int, bool]:
        """
        开始新的宏观审查迭代
        
        Returns:
            Tuple[int, bool]: (当前迭代次数, 是否可以继续迭代)
        """
        self.macro_iteration_count += 1
        self.current_stage = WorkflowStage.MACRO_REVIEW
        self.current_iteration_type = IterationType.MACRO
        self._update_timestamp()
        
        can_continue = self.macro_iteration_count <= self.max_macro_iterations
        
        print(f"[迭代管理器] 开始宏观审查 - 第{self.macro_iteration_count}/{self.max_macro_iterations}轮")
        
        # 记录迭代开始
        self._record_iteration(
            stage=WorkflowStage.MACRO_REVIEW,
            iteration_type=IterationType.MACRO,
            iteration_count=self.macro_iteration_count,
            max_iterations=self.max_macro_iterations
        )
        
        return self.macro_iteration_count, can_continue
    
    def complete_macro_iteration(self, is_passed: bool, feedback_length: Optional[int] = None, context_info: Optional[Dict] = None) -> bool:
        """
        完成当前宏观审查迭代
        
        Args:
            is_passed (bool): 审查是否通过
            feedback_length (Optional[int]): 反馈信息长度
            context_info (Optional[Dict]): 上下文信息
            
        Returns:
            bool: 是否应该继续迭代
        """
        self.macro_review_passed = is_passed
        self._update_timestamp()
        
        # 更新最后一条记录
        if self.iteration_history:
            last_record = self.iteration_history[-1]
            if (last_record.stage == WorkflowStage.MACRO_REVIEW and 
                last_record.iteration_type == IterationType.MACRO and
                last_record.iteration_count == self.macro_iteration_count):
                last_record.is_passed = is_passed
                last_record.feedback_length = feedback_length
                last_record.context_info = context_info or {}
        
        if is_passed:
            print(f"[迭代管理器] 宏观审查通过 - 第{self.macro_iteration_count}轮")
            return False  # 不需要继续迭代
        else:
            can_continue = self.macro_iteration_count < self.max_macro_iterations
            if can_continue:
                print(f"[迭代管理器] 宏观审查未通过 - 第{self.macro_iteration_count}轮，准备下一轮")
            else:
                print(f"[迭代管理器] 宏观审查达到最大迭代次数 - {self.max_macro_iterations}轮，强制进入微观审查")
            return can_continue
    
    def start_micro_iteration(self) -> Tuple[int, bool]:
        """
        开始新的微观审查迭代
        
        Returns:
            Tuple[int, bool]: (当前迭代次数, 是否可以继续迭代)
        """
        self.micro_iteration_count += 1
        self.current_stage = WorkflowStage.MICRO_REVIEW
        self.current_iteration_type = IterationType.MICRO
        self._update_timestamp()
        
        can_continue = self.micro_iteration_count <= self.max_micro_iterations
        
        print(f"[迭代管理器] 开始微观审查 - 第{self.micro_iteration_count}/{self.max_micro_iterations}轮")
        
        # 记录迭代开始
        self._record_iteration(
            stage=WorkflowStage.MICRO_REVIEW,
            iteration_type=IterationType.MICRO,
            iteration_count=self.micro_iteration_count,
            max_iterations=self.max_micro_iterations
        )
        
        return self.micro_iteration_count, can_continue
    
    def complete_micro_iteration(self, is_passed: bool, feedback_length: Optional[int] = None, context_info: Optional[Dict] = None) -> bool:
        """
        完成当前微观审查迭代
        
        Args:
            is_passed (bool): 审查是否通过
            feedback_length (Optional[int]): 反馈信息长度
            context_info (Optional[Dict]): 上下文信息
            
        Returns:
            bool: 是否应该继续迭代
        """
        self.micro_review_passed = is_passed
        self._update_timestamp()
        
        # 更新最后一条记录
        if self.iteration_history:
            last_record = self.iteration_history[-1]
            if (last_record.stage == WorkflowStage.MICRO_REVIEW and 
                last_record.iteration_type == IterationType.MICRO and
                last_record.iteration_count == self.micro_iteration_count):
                last_record.is_passed = is_passed
                last_record.feedback_length = feedback_length
                last_record.context_info = context_info or {}
        
        if is_passed:
            print(f"[迭代管理器] 微观审查通过 - 第{self.micro_iteration_count}轮")
            return False  # 不需要继续迭代
        else:
            can_continue = self.micro_iteration_count < self.max_micro_iterations
            if can_continue:
                print(f"[迭代管理器] 微观审查未通过 - 第{self.micro_iteration_count}轮，准备下一轮")
            else:
                print(f"[迭代管理器] 微观审查达到最大迭代次数 - {self.max_micro_iterations}轮，返回当前版本")
            return can_continue
    
    def start_title_iteration(self) -> Tuple[int, bool]:
        """
        开始新的标题审查迭代
        
        Returns:
            Tuple[int, bool]: (当前迭代次数, 是否可以继续迭代)
        """
        self.title_iteration_count += 1
        self.current_stage = WorkflowStage.REVIEW_TITLE
        self.current_iteration_type = IterationType.TITLE
        self._update_timestamp()
        
        can_continue = self.title_iteration_count <= self.max_title_iterations
        
        print(f"[迭代管理器] 开始标题审查 - 第{self.title_iteration_count}/{self.max_title_iterations}轮")
        
        # 记录迭代开始
        self._record_iteration(
            stage=WorkflowStage.REVIEW_TITLE,
            iteration_type=IterationType.TITLE,
            iteration_count=self.title_iteration_count,
            max_iterations=self.max_title_iterations
        )
        
        return self.title_iteration_count, can_continue
    
    def complete_title_iteration(self, is_passed: bool, feedback_length: Optional[int] = None, context_info: Optional[Dict] = None) -> bool:
        """
        完成当前标题审查迭代
        
        Args:
            is_passed (bool): 审查是否通过
            feedback_length (Optional[int]): 反馈信息长度
            context_info (Optional[Dict]): 上下文信息
            
        Returns:
            bool: 是否应该继续迭代
        """
        self.title_review_passed = is_passed
        self._update_timestamp()
        
        # 更新最后一条记录
        if self.iteration_history:
            last_record = self.iteration_history[-1]
            if (last_record.stage == WorkflowStage.REVIEW_TITLE and 
                last_record.iteration_type == IterationType.TITLE and
                last_record.iteration_count == self.title_iteration_count):
                last_record.is_passed = is_passed
                last_record.feedback_length = feedback_length
                last_record.context_info = context_info or {}
        
        if is_passed:
            print(f"[迭代管理器] 标题审查通过 - 第{self.title_iteration_count}轮")
            return False  # 不需要继续迭代
        else:
            can_continue = self.title_iteration_count < self.max_title_iterations
            if can_continue:
                print(f"[迭代管理器] 标题审查未通过 - 第{self.title_iteration_count}轮，准备下一轮")
            else:
                print(f"[迭代管理器] 标题审查达到最大迭代次数 - {self.max_title_iterations}轮，返回当前版本")
            return can_continue
    
    def get_current_iteration_info(self) -> Dict:
        """
        获取当前迭代信息，用于LLM调用的iteration_info参数
        
        Returns:
            Dict: 包含当前迭代信息的字典
        """
        if self.current_iteration_type == IterationType.MACRO:
            return {
                "macro_iteration": self.macro_iteration_count,
                "max_macro_iterations": self.max_macro_iterations,
                "iteration_type": "macro"
            }
        elif self.current_iteration_type == IterationType.MICRO:
            return {
                "micro_iteration": self.micro_iteration_count,
                "max_micro_iterations": self.max_micro_iterations,
                "iteration_type": "micro"
            }
        elif self.current_iteration_type == IterationType.TITLE:
            return {
                "title_iteration": self.title_iteration_count,
                "max_title_iterations": self.max_title_iterations,
                "iteration_type": "title"
            }
        else:
            return {"iteration_type": "unknown"}
    
    def set_stage(self, stage: WorkflowStage) -> None:
        """
        设置当前工作流阶段
        
        Args:
            stage (WorkflowStage): 目标阶段
        """
        self.current_stage = stage
        self._update_timestamp()
        print(f"[迭代管理器] 阶段切换: {stage.value}")
    
    def reset(self) -> None:
        """
        重置迭代管理器状态
        
        清除所有迭代计数和历史记录，重新开始管理。
        """
        self.macro_iteration_count = 0
        self.micro_iteration_count = 0
        self.title_iteration_count = 0
        
        self.current_stage = WorkflowStage.INITIALIZED
        self.current_iteration_type = None
        
        self.macro_review_passed = False
        self.micro_review_passed = False
        self.title_review_passed = False
        
        self.iteration_history.clear()
        
        self.session_start_time = datetime.now().isoformat()
        self.last_update_time = self.session_start_time
        
        print("[迭代管理器] 状态已重置")
    
    def get_statistics(self) -> Dict:
        """
        获取迭代统计信息
        
        Returns:
            Dict: 包含各种统计指标的字典
        """
        return {
            "session_info": {
                "start_time": self.session_start_time,
                "last_update": self.last_update_time,
                "current_stage": self.current_stage.value,
                "current_iteration_type": self.current_iteration_type.value if self.current_iteration_type else None
            },
            "iteration_counts": {
                "macro_iterations": self.macro_iteration_count,
                "micro_iterations": self.micro_iteration_count,
                "title_iterations": self.title_iteration_count,
                "total_iterations": self.macro_iteration_count + self.micro_iteration_count + self.title_iteration_count
            },
            "iteration_limits": {
                "max_macro_iterations": self.max_macro_iterations,
                "max_micro_iterations": self.max_micro_iterations,
                "max_title_iterations": self.max_title_iterations
            },
            "review_status": {
                "macro_review_passed": self.macro_review_passed,
                "micro_review_passed": self.micro_review_passed,
                "title_review_passed": self.title_review_passed
            },
            "history_summary": {
                "total_records": len(self.iteration_history),
                "passed_iterations": len([r for r in self.iteration_history if r.is_passed is True]),
                "failed_iterations": len([r for r in self.iteration_history if r.is_passed is False])
            }
        }
    
    def validate_state(self) -> Tuple[bool, List[str]]:
        """
        验证当前迭代状态的一致性
        
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查迭代计数是否超限
        if self.macro_iteration_count > self.max_macro_iterations:
            errors.append(f"宏观迭代计数超限: {self.macro_iteration_count} > {self.max_macro_iterations}")
        
        if self.micro_iteration_count > self.max_micro_iterations:
            errors.append(f"微观迭代计数超限: {self.micro_iteration_count} > {self.max_micro_iterations}")
        
        if self.title_iteration_count > self.max_title_iterations:
            errors.append(f"标题迭代计数超限: {self.title_iteration_count} > {self.max_title_iterations}")
        
        # 检查负数计数
        if self.macro_iteration_count < 0:
            errors.append(f"宏观迭代计数为负数: {self.macro_iteration_count}")
        
        if self.micro_iteration_count < 0:
            errors.append(f"微观迭代计数为负数: {self.micro_iteration_count}")
        
        if self.title_iteration_count < 0:
            errors.append(f"标题迭代计数为负数: {self.title_iteration_count}")
        
        # 检查历史记录一致性
        macro_records = [r for r in self.iteration_history if r.iteration_type == IterationType.MACRO]
        if len(macro_records) != self.macro_iteration_count:
            errors.append(f"宏观迭代历史记录不一致: 记录{len(macro_records)}条，计数{self.macro_iteration_count}")
        
        micro_records = [r for r in self.iteration_history if r.iteration_type == IterationType.MICRO]
        if len(micro_records) != self.micro_iteration_count:
            errors.append(f"微观迭代历史记录不一致: 记录{len(micro_records)}条，计数{self.micro_iteration_count}")
        
        return len(errors) == 0, errors
    
    def _record_iteration(self, stage: WorkflowStage, iteration_type: IterationType, 
                         iteration_count: int, max_iterations: int, 
                         context_info: Optional[Dict] = None) -> None:
        """
        记录迭代信息
        
        Args:
            stage (WorkflowStage): 工作流阶段
            iteration_type (IterationType): 迭代类型
            iteration_count (int): 迭代次数
            max_iterations (int): 最大迭代次数
            context_info (Optional[Dict]): 上下文信息
        """
        record = IterationRecord(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            iteration_type=iteration_type,
            iteration_count=iteration_count,
            max_iterations=max_iterations,
            context_info=context_info or {}
        )
        
        self.iteration_history.append(record)
    
    def _update_timestamp(self) -> None:
        """
        更新最后更新时间戳
        """
        self.last_update_time = datetime.now().isoformat()
    
    def print_status(self) -> None:
        """
        打印当前迭代状态摘要
        """
        print(f"\n{'='*60}")
        print(f"迭代管理器状态摘要")
        print(f"{'='*60}")
        print(f"当前阶段: {self.current_stage.value}")
        print(f"当前迭代类型: {self.current_iteration_type.value if self.current_iteration_type else 'None'}")
        print(f"宏观审查: {self.macro_iteration_count}/{self.max_macro_iterations} (通过: {self.macro_review_passed})")
        print(f"微观审查: {self.micro_iteration_count}/{self.max_micro_iterations} (通过: {self.micro_review_passed})")
        print(f"标题审查: {self.title_iteration_count}/{self.max_title_iterations} (通过: {self.title_review_passed})")
        print(f"历史记录: {len(self.iteration_history)} 条")
        print(f"{'='*60}\n")


# 全局迭代管理器实例
_global_iteration_manager: Optional[IterationManager] = None


def get_global_iteration_manager() -> IterationManager:
    """
    获取全局迭代管理器实例
    
    Returns:
        IterationManager: 全局迭代管理器实例
    """
    global _global_iteration_manager
    if _global_iteration_manager is None:
        _global_iteration_manager = IterationManager()
    return _global_iteration_manager


def reset_global_iteration_manager() -> None:
    """
    重置全局迭代管理器
    """
    global _global_iteration_manager
    if _global_iteration_manager is not None:
        _global_iteration_manager.reset()
    else:
        _global_iteration_manager = IterationManager()


if __name__ == "__main__":
    # 测试代码
    manager = IterationManager()
    
    print("=== 迭代管理器测试 ===")
    
    # 测试宏观审查迭代
    print("\n1. 测试宏观审查迭代")
    for i in range(4):  # 测试超限情况
        iteration_count, can_continue = manager.start_macro_iteration()
        print(f"宏观迭代 {iteration_count}: 可继续={can_continue}")
        
        # 模拟审查结果
        is_passed = i == 2  # 第3轮通过
        should_continue = manager.complete_macro_iteration(is_passed, feedback_length=100)
        print(f"审查结果: 通过={is_passed}, 应继续={should_continue}")
        
        if not should_continue:
            break
    
    # 测试微观审查迭代
    print("\n2. 测试微观审查迭代")
    for i in range(3):
        iteration_count, can_continue = manager.start_micro_iteration()
        print(f"微观迭代 {iteration_count}: 可继续={can_continue}")
        
        # 模拟审查结果
        is_passed = i == 1  # 第2轮通过
        should_continue = manager.complete_micro_iteration(is_passed, feedback_length=50)
        print(f"审查结果: 通过={is_passed}, 应继续={should_continue}")
        
        if not should_continue:
            break
    
    # 打印统计信息
    print("\n3. 统计信息")
    stats = manager.get_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    
    # 验证状态
    print("\n4. 状态验证")
    is_valid, errors = manager.validate_state()
    print(f"状态有效: {is_valid}")
    if errors:
        for error in errors:
            print(f"错误: {error}")
    
    # 打印状态摘要
    manager.print_status()