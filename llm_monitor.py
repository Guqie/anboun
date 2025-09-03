# llm_monitor.py
# LLM调用监控和日志系统

import time
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LLMCallStatus(Enum):
    """LLM调用状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRY = "retry"

@dataclass
class LLMCallRecord:
    """LLM调用记录数据类"""
    call_id: str
    timestamp: str
    function_name: str
    model: str
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    system_message_length: int
    user_message_length: int
    response_length: int
    response_time: float
    attempt_count: int
    status: str
    error_message: Optional[str] = None
    stage: Optional[str] = None
    iteration_info: Optional[Dict] = None

class LLMMonitor:
    """LLM调用监控器"""
    
    def __init__(self, log_file: str = "llm_monitor.log", enable_console: bool = True):
        """
        初始化LLM监控器
        
        Args:
            log_file (str): 日志文件路径
            enable_console (bool): 是否启用控制台输出
        """
        self.log_file = log_file
        self.enable_console = enable_console
        self.call_records: List[LLMCallRecord] = []
        self.session_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_response_time": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0
        }
        
        # 先创建日志目录，避免FileNotFoundError
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 设置日志记录器
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger("LLMMonitor")
        self.logger.setLevel(logging.DEBUG)
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def generate_call_id(self) -> str:
        """生成唯一的调用ID"""
        return f"llm_{int(time.time() * 1000)}_{len(self.call_records)}"
    
    def log_call_start(self, call_id: str, function_name: str, model: str, 
                      system_message: str, user_message: str, 
                      temperature: float = 0.5, top_p: float = 0.95,
                      frequency_penalty: float = 0, presence_penalty: float = 0,
                      stage: str = None, iteration_info: Dict = None) -> None:
        """记录LLM调用开始"""
        
        log_message = f" LLM调用开始 [ID: {call_id}]"
        log_message += f"\n  阶段: {stage or '未指定'}"
        log_message += f"\n  模型: {model}"
        log_message += f"\n  函数: {function_name}"
        log_message += f"\n  参数: T={temperature}, P={top_p}, FP={frequency_penalty}, PP={presence_penalty}"
        log_message += f"\n  输入长度: 系统消息={len(system_message)}字符, 用户消息={len(user_message)}字符"
        
        if iteration_info:
            log_message += f"\n  迭代信息: {iteration_info}"
        
        self.logger.info(log_message)
        
        # 详细输入内容记录到DEBUG级别
        self.logger.debug(f"系统消息内容 [ID: {call_id}]: {system_message[:200]}...")
        self.logger.debug(f"用户消息内容 [ID: {call_id}]: {user_message[:200]}...")
    
    def log_call_end(self, call_id: str, function_name: str, model: str,
                    system_message: str, user_message: str, response: Optional[str],
                    response_time: float, attempt_count: int, status: LLMCallStatus,
                    error_message: Optional[str] = None, stage: str = None,
                    iteration_info: Dict = None, temperature: float = 0.5, 
                    top_p: float = 0.95, frequency_penalty: float = 0, 
                    presence_penalty: float = 0) -> None:
        """记录LLM调用结束"""
        
        # 创建调用记录
        record = LLMCallRecord(
            call_id=call_id,
            timestamp=datetime.now().isoformat(),
            function_name=function_name,
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            system_message_length=len(system_message),
            user_message_length=len(user_message),
            response_length=len(response) if response else 0,
            response_time=response_time,
            attempt_count=attempt_count,
            status=status.value,
            error_message=error_message,
            stage=stage,
            iteration_info=iteration_info
        )
        
        self.call_records.append(record)
        
        # 更新统计信息
        self._update_stats(record)
        
        # 记录日志
        status_emoji = "成功" if status == LLMCallStatus.SUCCESS else "失败"
        log_message = f"{status_emoji} LLM调用完成 [ID: {call_id}]"
        log_message += f"\n  阶段: {stage or '未指定'}"
        log_message += f"\n  模型: {model}"
        log_message += f"\n  函数: {function_name}"
        log_message += f"\n  ⏱响应时间: {response_time:.2f}秒"
        log_message += f"\n  尝试次数: {attempt_count}"
        log_message += f"\n  输出长度: {len(response) if response else 0}字符"
        log_message += f"\n  状态: {status.value}"
        
        if error_message:
            log_message += f"\n  错误: {error_message}"
        
        if iteration_info:
            log_message += f"\n  迭代信息: {iteration_info}"
        
        if status == LLMCallStatus.SUCCESS:
            self.logger.info(log_message)
            # 详细响应内容记录到DEBUG级别
            if response:
                self.logger.debug(f"响应内容 [ID: {call_id}]: {response[:200]}...")
        else:
            self.logger.error(log_message)
    
    def _update_stats(self, record: LLMCallRecord) -> None:
        """更新统计信息"""
        self.session_stats["total_calls"] += 1
        self.session_stats["total_response_time"] += record.response_time
        self.session_stats["total_input_tokens"] += record.system_message_length + record.user_message_length
        self.session_stats["total_output_tokens"] += record.response_length
        
        if record.status == LLMCallStatus.SUCCESS.value:
            self.session_stats["successful_calls"] += 1
        else:
            self.session_stats["failed_calls"] += 1
    
    def log_workflow_stage(self, stage: str, description: str, level: LogLevel = LogLevel.INFO) -> None:
        """记录工作流阶段信息"""
        log_message = f" 工作流阶段: {stage}\n   描述: {description}"
        
        if level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_message)

    def log_stage_context(self, stage: str, context: Dict[str, Any], level: LogLevel = LogLevel.DEBUG) -> None:
        """
        记录结构化的阶段上下文信息，便于后续分析迭代原因与数据流动态。
        
        参数:
            stage (str): 当前阶段标识，例如 "macro_review.pre", "macro_review.decision" 等。
            context (Dict[str, Any]): 需要记录的上下文字段，建议为可JSON序列化的字典。
            level (LogLevel): 日志级别，默认DEBUG以避免干扰常规INFO输出。
        """
        try:
            payload = {
                "stage": stage,
                "context": context
            }
            msg = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            # 兜底：如果不可序列化，退化为字符串输出
            msg = f"stage={stage} context={str(context)}"
        
        if level == LogLevel.DEBUG:
            self.logger.debug(msg)
        elif level == LogLevel.INFO:
            self.logger.info(msg)
        elif level == LogLevel.WARNING:
            self.logger.warning(msg)
        elif level == LogLevel.ERROR:
            self.logger.error(msg)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(msg)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        stats = self.session_stats.copy()
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_calls"] / stats["total_calls"] * 100
            stats["average_response_time"] = stats["total_response_time"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0
            stats["average_response_time"] = 0
        
        return stats
    
    def print_session_summary(self) -> None:
        """打印会话摘要"""
        stats = self.get_session_stats()
        
        summary = f"""
{'='*60}
📊 LLM调用会话摘要
{'='*60}
 总调用次数: {stats['total_calls']}
 成功调用: {stats['successful_calls']}
 失败调用: {stats['failed_calls']}
 成功率: {stats['success_rate']:.1f}%
⏱ 平均响应时间: {stats['average_response_time']:.2f}秒
 总输入字符: {stats['total_input_tokens']:,}
 总输出字符: {stats['total_output_tokens']:,}
{'='*60}
"""
        
        print(summary)
        self.logger.info(summary)
    
    def export_records_to_json(self, filename: str = None) -> str:
        """导出调用记录到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_records_{timestamp}.json"
        
        records_data = {
            "session_stats": self.get_session_stats(),
            "call_records": [asdict(record) for record in self.call_records]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(records_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"调用记录已导出到: {filename}")
        return filename

# 全局监控器实例
_global_monitor: Optional[LLMMonitor] = None

def get_global_monitor() -> LLMMonitor:
    """获取全局LLM监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        # 使用绝对路径确保日志文件能够正确创建
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(current_dir, "logs", "llm_monitor.log")
        _global_monitor = LLMMonitor(
            log_file=log_file_path,
            enable_console=True
        )
    return _global_monitor

def initialize_monitor(log_file: str = None, enable_console: bool = True) -> LLMMonitor:
    """初始化全局监控器"""
    global _global_monitor
    if log_file is None:
        # 使用绝对路径确保日志文件能够正确创建
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(current_dir, "logs", "llm_monitor.log")
    _global_monitor = LLMMonitor(log_file, enable_console)
    return _global_monitor