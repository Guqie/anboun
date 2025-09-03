#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的长度跟踪器模块

该模块提供独立的简报内容长度统计、动态更新和上下文一致性管理功能。
主要目的是在系统运行过程中对简报内容字符判断保持上下文一致，
同时为工作流提供实时的长度统计信息。

作者: AI智学导师
创建时间: 2025-01-29
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from utils import validate_content_length_characters, validate_574_character_standard, calculate_length_by_mode, LengthMode


@dataclass
class LengthValidationRecord:
    """
    长度校验记录数据类
    
    用于记录每次长度校验的详细信息，包括时间戳、内容长度、
    校验结果和上下文信息。
    """
    timestamp: str
    content_length: int
    validation_result: Optional[str]
    is_valid: bool
    stage: str  # 'draft', 'macro_review', 'micro_review', 'refine'
    iteration_count: int
    context_info: Dict[str, any]


class BriefContentLengthTracker:
    """
    简报内容长度跟踪器
    
    负责跟踪简报生成过程中每个阶段的内容长度变化，
    提供动态统计、历史记录和上下文一致性管理功能。
    
    主要功能:
    1. 实时长度统计和校验
    2. 历史记录管理
    3. 动态长度信息生成
    4. 上下文状态跟踪
    5. 统计报告生成
    """
    
    def __init__(self, target_min: int = 430, target_max: int = 550, ideal_target: int = 480):
        """
        初始化长度跟踪器
        
        Args:
            target_min (int): 最小目标长度，默认430字符
            target_max (int): 最大目标长度，默认550字符
            ideal_target (int): 理想目标长度，默认480字符
        """
        self.target_min = target_min
        self.target_max = target_max
        self.ideal_target = ideal_target
        
        # 历史记录存储
        self.validation_history: List[LengthValidationRecord] = []
        
        # 当前状态跟踪
        self.current_length = 0
        self.current_stage = "initialized"
        # 注意：迭代计数现在由IterationManager统一管理
        # 这里保留字段是为了向后兼容，但不再主动更新
        self.macro_iteration_count = 0
        self.micro_iteration_count = 0
        
        # 统计信息
        self.total_validations = 0
        self.valid_count = 0
        self.invalid_count = 0
        
        # 上下文信息
        self.session_start_time = datetime.now().isoformat()
        self.last_validation_time = None
    
    def validate_content_length(self, content: str, stage: str, iteration_count: int = 0, 
                              context_info: Optional[Dict] = None) -> Tuple[bool, Optional[str], str]:
        """
        执行内容长度校验并记录结果
        
        Args:
            content (str): 要校验的简报内容
            stage (str): 当前阶段标识 ('draft', 'macro_review', 'micro_review', 'refine')
            iteration_count (int): 当前迭代次数
            context_info (Dict, optional): 额外的上下文信息
            
        Returns:
            Tuple[bool, Optional[str], str]: (是否有效, 校验反馈, 动态长度信息)
        """
        # 更新当前状态（使用字符计算）
        self.current_length = calculate_length_by_mode(content, LengthMode.SEMANTIC_WITH_PUNCT)
        self.current_stage = stage
        
        # 注意：迭代计数现在由IterationManager统一管理
        # 这里不再更新内部的迭代计数，而是使用传入的iteration_count参数进行记录
        
        # 执行程序化长度校验（使用574字符标准）
        validation_feedback, actual_length, detail_info = validate_574_character_standard(content)
        is_valid = validation_feedback is None
        
        # 更新当前长度为校验函数返回的实际长度，确保一致性
        self.current_length = actual_length
        
        # 生成动态长度信息
        dynamic_info = self._generate_dynamic_length_info(validation_feedback)
        
        # 记录校验结果
        record = LengthValidationRecord(
            timestamp=datetime.now().isoformat(),
            content_length=actual_length,  # 使用实际校验长度
            validation_result=validation_feedback,
            is_valid=is_valid,
            stage=stage,
            iteration_count=iteration_count,
            context_info=context_info or {}
        )
        
        self.validation_history.append(record)
        self._update_statistics(is_valid)
        
        # 输出到终端（传递实际长度和详细信息）
        self._print_validation_result(validation_feedback, actual_length, detail_info)
        
        return is_valid, validation_feedback, dynamic_info
    
    def _generate_dynamic_length_info(self, validation_feedback: Optional[str]) -> str:
        """
        生成动态长度信息字符串（基于字符）
        
        Args:
            validation_feedback (Optional[str]): 校验反馈信息
            
        Returns:
            str: 格式化的动态长度信息
        """
        length_status = "符合要求" if validation_feedback is None else "需要调整"
        
        # 计算与理想目标的偏差（使用字符）
        deviation = abs(self.current_length - self.ideal_target)
        deviation_percentage = (deviation / self.ideal_target) * 100
        
        dynamic_info = f"""
### 当前内容长度信息 ###
- **精确字符数**: {self.current_length} 字符
- **目标范围**: {self.target_min}-{self.target_max} 字符（理想目标：{self.ideal_target}字符左右）
- **长度状态**: {length_status}
- **当前阶段**: {self.current_stage}
- **与理想目标偏差**: {deviation} 字符 ({deviation_percentage:.1f}%)
- **宏观迭代**: 第 {self.macro_iteration_count} 轮
- **微观迭代**: 第 {self.micro_iteration_count} 轮
- **574标准**: 目标480字符 (理想区间: 430-550)
"""
        
        if validation_feedback:
            dynamic_info += f"\n- **程序化校验结果**: {validation_feedback.strip()}"
        
        # 添加历史趋势信息（使用字符）
        if len(self.validation_history) > 1:
            previous_length = self.validation_history[-2].content_length
            length_change = self.current_length - previous_length
            change_direction = "增加" if length_change > 0 else "减少" if length_change < 0 else "保持不变"
            dynamic_info += f"\n- **长度变化**: 相比上次{change_direction} {abs(length_change)} 字符"
        
        return dynamic_info
    
    def _print_validation_result(self, validation_feedback: Optional[str], actual_length: int, detail_info: Dict) -> None:
        """
        输出长度校验结果到终端（基于574字符标准）
        
        Args:
            validation_feedback (Optional[str]): 校验反馈信息
            actual_length (int): 校验函数返回的实际长度
            detail_info (Dict): 详细的长度信息
        """
        # 根据阶段显示不同的标题
        stage_names = {
            'draft': '📝 草稿阶段',
            'macro_review': '🔍 宏观审查',
            'micro_review': '🔬 微观审查', 
            'refine': '🔄 润色阶段'
        }
        stage_display = stage_names.get(self.current_stage, self.current_stage)
        
        print(f"\n=== 统一字符判断标准 ({stage_display}) ===")
        print(f"当前内容长度: {actual_length} 字符")
        print(f"目标长度: {detail_info.get('target_length', 475)} 字符")
        print(f"计算模式: {detail_info.get('mode', 'semantic_with_punct')}模式")
        
        if validation_feedback:
            print(f"校验反馈: {validation_feedback.strip()}")
        else:
            print("校验状态: 长度符合要求（574字符标准）")
        
        # 优化阶段信息显示
        if self.current_stage in ['macro_review', 'micro_review']:
            if self.current_stage == 'macro_review':
                print(f"迭代信息: 宏观第{self.macro_iteration_count}轮")
            else:
                print(f"迭代信息: 微观第{self.micro_iteration_count}轮")
        else:
            print(f"阶段信息: {stage_display} - 宏观第{self.macro_iteration_count}轮, 微观第{self.micro_iteration_count}轮")
        
        print("=" * 50)
    
    def _update_statistics(self, is_valid: bool) -> None:
        """
        更新统计信息
        
        Args:
            is_valid (bool): 本次校验是否有效
        """
        self.total_validations += 1
        self.last_validation_time = datetime.now().isoformat()
        
        if is_valid:
            self.valid_count += 1
        else:
            self.invalid_count += 1
    
    def get_length_statistics(self) -> Dict[str, any]:
        """
        获取长度统计信息（基于字符）
        
        Returns:
            Dict[str, any]: 包含各种统计指标的字典
        """
        if not self.validation_history:
            return {"message": "暂无校验记录"}
        
        # 所有长度数据都是字符
        lengths = [record.content_length for record in self.validation_history]
        
        return {
            "session_info": {
                "start_time": self.session_start_time,
                "last_validation": self.last_validation_time,
                "current_stage": self.current_stage,
                "macro_iterations": self.macro_iteration_count,
                "micro_iterations": self.micro_iteration_count
            },
            "length_metrics": {
                "current_length": self.current_length,
                "min_length": min(lengths),
                "max_length": max(lengths),
                "avg_length": sum(lengths) / len(lengths),
                "target_range": f"{self.target_min}-{self.target_max}",
                "ideal_target": self.ideal_target,
                "unit": "字符"
            },
            "validation_metrics": {
                "total_validations": self.total_validations,
                "valid_count": self.valid_count,
                "invalid_count": self.invalid_count,
                "success_rate": (self.valid_count / self.total_validations * 100) if self.total_validations > 0 else 0
            },
            "trend_analysis": self._analyze_length_trend()
        }
    
    def _analyze_length_trend(self) -> Dict[str, any]:
        """
        分析长度变化趋势（基于字符）
        
        Returns:
            Dict[str, any]: 趋势分析结果
        """
        if len(self.validation_history) < 2:
            return {"message": "数据不足，无法分析趋势"}
        
        # 所有长度数据都是字符
        lengths = [record.content_length for record in self.validation_history]
        
        # 计算总体趋势（字符变化）
        total_change = lengths[-1] - lengths[0]
        
        # 计算稳定性（标准差，字符）
        avg_length = sum(lengths) / len(lengths)
        variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
        stability = variance ** 0.5
        
        # 分析收敛性
        recent_changes = [abs(lengths[i] - lengths[i-1]) for i in range(1, len(lengths))]
        is_converging = len(recent_changes) >= 3 and all(
            recent_changes[i] <= recent_changes[i-1] for i in range(1, min(3, len(recent_changes)))
        )
        
        return {
            "total_change": total_change,
            "stability_score": round(stability, 2),
            "is_converging": is_converging,
            "recent_volatility": round(sum(recent_changes[-3:]) / min(3, len(recent_changes)), 2) if recent_changes else 0,
            "unit": "字符"
        }
    
    def export_validation_history(self, file_path: Optional[str] = None) -> str:
        """
        导出校验历史记录
        
        Args:
            file_path (Optional[str]): 导出文件路径，如果为None则自动生成
            
        Returns:
            str: 导出文件的路径
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"length_validation_history_{timestamp}.json"
        
        export_data = {
            "session_info": {
                "start_time": self.session_start_time,
                "export_time": datetime.now().isoformat(),
                "target_config": {
                    "min": self.target_min,
                    "max": self.target_max,
                    "ideal": self.ideal_target
                }
            },
            "statistics": self.get_length_statistics(),
            "validation_history": [asdict(record) for record in self.validation_history]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return file_path
    
    def reset_tracker(self) -> None:
        """
        重置跟踪器状态
        
        清除所有历史记录和统计信息，重新开始跟踪。
        """
        self.validation_history.clear()
        self.current_length = 0
        self.current_stage = "initialized"
        self.macro_iteration_count = 0
        self.micro_iteration_count = 0
        self.total_validations = 0
        self.valid_count = 0
        self.invalid_count = 0
        self.session_start_time = datetime.now().isoformat()
        self.last_validation_time = None
        
        print("长度跟踪器已重置")


# 全局跟踪器实例
# 在工作流中可以直接导入使用
length_tracker = BriefContentLengthTracker()


def get_global_length_tracker() -> BriefContentLengthTracker:
    """
    获取全局长度跟踪器实例
    
    Returns:
        BriefContentLengthTracker: 全局跟踪器实例
    """
    return length_tracker


def reset_global_tracker() -> None:
    """
    重置全局长度跟踪器
    """
    global length_tracker
    length_tracker.reset_tracker()


if __name__ == "__main__":
    # 测试代码
    tracker = BriefContentLengthTracker()
    
    # 模拟测试（使用字符计算）
    test_content = "这是一个测试简报内容" * 20  # 约400字符
    
    is_valid, feedback, dynamic_info = tracker.validate_content_length(
        test_content, "draft", 1, {"test": True}
    )
    
    print("\n=== 动态长度信息（字符） ===")
    print(dynamic_info)
    
    print("\n=== 统计信息（字符） ===")
    stats = tracker.get_length_statistics()
    print(json.dumps(stats, ensure_ascii=False, indent=2))