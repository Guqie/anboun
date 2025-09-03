# workflow.py

import asyncio
import os
import re
import csv
import json
import ast
import traceback
import logging
from datetime import datetime
from utils import convert_to_cn_term, convert_to_date, clean_stock_codes, get_prompt, call_llm, extract_with_xml, split_to_sentences, save_as_txt, replace_year_with_2025, remove_year_at_start
from embedding import get_similar_tags
from prompts.k_adjust_length import validate_and_instruct_title_length
from utils import validate_and_instruct_content_length
from length_tracker import get_global_length_tracker
from iteration_manager import get_global_iteration_manager, WorkflowStage, IterationType
from llm_monitor import get_global_monitor, LogLevel

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('workflow_errors.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_exception(func_name: str, attempt: int, max_attempts: int, error: Exception, context: dict = None):
    """
    记录异常信息的统一函数
    
    Args:
        func_name (str): 发生异常的函数名
        attempt (int): 当前尝试次数
        max_attempts (int): 最大尝试次数
        error (Exception): 异常对象
        context (dict): 额外的上下文信息
    """
    error_info = {
        'function': func_name,
        'attempt': f'{attempt}/{max_attempts}',
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat(),
        'traceback': traceback.format_exc()
    }
    
    if context:
        error_info.update(context)
    
    logger.error(f"异常详情: {json.dumps(error_info, ensure_ascii=False, indent=2)}")
    print(f"[{func_name}] 第{attempt}次尝试失败: {type(error).__name__}: {str(error)}")

def validate_llm_input(system_message: str, user_message: str, func_name: str) -> tuple[bool, str]:
    """
    验证LLM输入的质量和格式
    
    Args:
        system_message: 系统消息
        user_message: 用户消息
        func_name: 调用函数名
    
    Returns:
        tuple[bool, str]: (是否有效, 错误信息)
    """
    errors = []
    
    # 检查消息是否为空
    if not system_message or not system_message.strip():
        errors.append("系统消息为空")
    
    if not user_message or not user_message.strip():
        errors.append("用户消息为空")
    
    # 检查消息长度是否合理
    if len(system_message) > 50000:
        errors.append(f"系统消息过长: {len(system_message)} 字符")
    
    if len(user_message) > 100000:
        errors.append(f"用户消息过长: {len(user_message)} 字符")
    
    # 检查是否包含必要的XML标签（根据不同函数）
    if func_name == "draft_brief_content":
        if "<source_text>" not in user_message:
            errors.append("缺少必要的<source_text>标签")
    
    if func_name == "refine_brief_content":
        # refine_brief_content的source_text不是必需的，因为它主要基于反馈进行修改
        pass
    
    if func_name == "review_brief_content":
        if "<brief_content>" not in user_message:
            errors.append("缺少必要的<brief_content>标签")
    
    if func_name == "review_brief_sentences":
        if "<brief_sentences>" not in user_message:
            errors.append("缺少必要的<brief_sentences>标签")
    
    # 记录验证结果
    if errors:
        error_msg = f"LLM输入验证失败 ({func_name}): {'; '.join(errors)}"
        logger.warning(error_msg)
        return False, error_msg
    
    return True, ""

def validate_llm_output(llm_result: str, expected_tags: list, func_name: str) -> tuple[bool, str]:
    """
    验证LLM输出的质量和格式
    
    Args:
        llm_result: LLM返回结果
        expected_tags: 期望的XML标签列表
        func_name: 调用函数名
    
    Returns:
        tuple[bool, str]: (是否有效, 错误信息)
    """
    errors = []
    
    # 检查输出是否为空
    if not llm_result or not llm_result.strip():
        errors.append("LLM输出为空")
        return False, f"LLM输出验证失败 ({func_name}): {'; '.join(errors)}"
    
    # 检查输出长度是否合理
    if len(llm_result) < 10:
        errors.append(f"LLM输出过短: {len(llm_result)} 字符")
    
    if len(llm_result) > 200000:
        errors.append(f"LLM输出过长: {len(llm_result)} 字符")
    
    # 检查是否包含期望的XML标签
    for tag in expected_tags:
        if f"<{tag}>" not in llm_result or f"</{tag}>" not in llm_result:
            errors.append(f"缺少期望的<{tag}>标签")
    
    # 检查XML格式是否正确（基本检查）
    for tag in expected_tags:
        start_count = llm_result.count(f"<{tag}>")
        end_count = llm_result.count(f"</{tag}>")
        if start_count != end_count:
            errors.append(f"<{tag}>标签不匹配: 开始{start_count}个，结束{end_count}个")
    
    # 记录验证结果
    if errors:
        error_msg = f"LLM输出验证失败 ({func_name}): {'; '.join(errors)}"
        logger.warning(error_msg)
        return False, error_msg
    
    return True, ""
def get_source_files() -> list[str]:
    """
    获取inputs文件夹中所有txt文件的文件名（不包含扩展名）
    
    Returns:
        list[str]: 文件名列表
    """
    file_names = []
    for filename in os.listdir("inputs"):
        if filename.endswith('.txt'):
            file_names.append(filename[:-4])
    return file_names

def get_source_text(file_name: str) -> str:
    """
    从inputs文件夹中读取指定的txt文件内容
    
    Args:
        file_name: 文件名（不包含.txt扩展名）
    Returns:
        str: 文件内容
    """
    with open(f"inputs/{file_name}.txt", 'r', encoding='utf-8') as file:
        return file.read()

async def interpret_source_text(source_text: str) -> str:
    """
    解读文章
    Args:
        source_text (str): 文章内容
    Returns:
        tuple: (解读结果的JSON字符串, 关键要点字符串, 发布日期, 分析师观点或要求)
    """
    print(f"\n[DEBUG] 开始解读文章，文章长度: {len(source_text)}")
    
    import importlib
    a_module = importlib.import_module('prompts.a_interpret_source_text')
    json_format = a_module.json_format
    json_dumps_result = json.dumps(json_format, indent=4, ensure_ascii=False)
    
    system_message = get_prompt('a_interpret_source_text',
                               json=json,
                               json_format=json_format,
                               json_dumps_result=json_dumps_result)
    user_message = f"<source_text>\n{source_text}\n</source_text>"
    
    print(f"[DEBUG] 系统消息长度: {len(system_message)}, 用户消息长度: {len(user_message)}")
    
    # 验证LLM输入
    is_valid_input, input_error = validate_llm_input(system_message, user_message, "interpret_source_text")
    if not is_valid_input:
        logger.error(f"interpret_source_text输入验证失败: {input_error}")
        print(f"[DEBUG] 输入验证失败: {input_error}")
        return None, None, None
    
    print(f"[DEBUG] 输入验证通过，准备调用LLM")
    
    for attempt in range(3):
        try:
            llm_result = await call_llm(
                system_message, user_message, "openai/gpt-5-mini", 0.3, 0.8, 0, 0,
                stage="interpret_source",
                iteration_info={"attempt": attempt + 1, "max_attempts": 3}
            )
            
            # 验证LLM输出
            is_valid_output, output_error = validate_llm_output(llm_result, ["interpretation"], "interpret_source_text")
            if not is_valid_output:
                logger.warning(f"interpret_source_text输出验证失败: {output_error}")
                continue
            
            if interpretation := json.loads(extract_with_xml(llm_result, "interpretation")):
                key_points = "\n".join(interpretation["关键要点提炼"])
                published_date = replace_year_with_2025(interpretation.pop("新闻文章发布日期"))
                return json.dumps(interpretation, ensure_ascii=False, indent=4), key_points, published_date
        except Exception as e:
            log_exception(
                "interpret_source_text", 
                attempt + 1, 
                3, 
                e, 
                {
                    "source_text_length": len(source_text),
                    "llm_result_available": 'llm_result' in locals() and bool(llm_result),
                    "system_message_length": len(system_message) if 'system_message' in locals() else 0
                }
            )
    return None, None, None

async def draft_brief_content(source_text: str, interpretation: str, article_contents: str) -> tuple[str, str]:
    """
    生成文章的事实段落和设计上下文摘要
    Args:
        source_text (str): 文章内容
        interpretation (str): 对文章内容的解读
        article_contents (str): 参考文章内容
    Returns:
        tuple[str, str]: (事实段落, 设计上下文摘要)
    """
    # 获取全局长度跟踪器
    tracker = get_global_length_tracker()
    
    system_message = get_prompt('b_draft_brief_content').format(article_contents=article_contents)
    user_message = f"<source_text>\n{source_text}\n</source_text>\n<interpretation>\n{interpretation}\n</interpretation>"
    
    # 验证LLM输入
    is_valid_input, input_error = validate_llm_input(system_message, user_message, "draft_brief_content")
    if not is_valid_input:
        logger.error(f"draft_brief_content输入验证失败: {input_error}")
        return "输入验证失败，无法生成简报内容", ""
    
    for attempt in range(5):
        try:
            llm_result = await call_llm(
                system_message, user_message, "openai/gpt-oss-120b", 0.6, 0.95, 0, 0,
                stage="draft_content",
                iteration_info={"attempt": attempt + 1, "max_attempts": 5}
            )
            
            # 增加空值检测
            if not llm_result:
                print(f"LLM返回空结果（第{attempt+1}次）")
                continue  # 重试
            
            # 验证LLM输出
            is_valid_output, output_error = validate_llm_output(llm_result, ["brief_content", "draft_context_summary"], "draft_brief_content")
            if not is_valid_output:
                logger.warning(f"draft_brief_content输出验证失败: {output_error}")
                continue
                
            brief_content = extract_with_xml(llm_result, "brief_content")
            draft_context_summary = extract_with_xml(llm_result, "draft_context_summary")
            
            if not brief_content:
                print(f"未找到brief_content标签（第{attempt+1}次）")
                continue  # 重试
            
            # 使用长度跟踪器记录初始生成的内容
            tracker.validate_content_length(
                brief_content, "draft", 1, 
                {"source_length": len(source_text), "has_context_summary": bool(draft_context_summary)}
            )
                
            return brief_content.replace('\n', ''), draft_context_summary or ""
            
        except Exception as e:
            log_exception(
                "draft_brief_content", 
                attempt + 1, 
                5, 
                e, 
                {
                    "source_text_length": len(source_text),
                    "interpretation_length": len(interpretation),
                    "article_contents_length": len(article_contents),
                    "llm_result_available": 'llm_result' in locals() and bool(llm_result)
                }
            )
            continue
            
    # 所有重试失败后返回空字符串
    print("警告: 使用空段落代替LLM失败结果")
    return "", ""
async def review_brief_content(source_text: str, brief_content: str, draft_context_summary: str = None) -> str | None:
    """
    校对文章的事实段落
    Args:
        source_text (str): 文章内容
        brief_content (str): 事实段落
        draft_context_summary (str, optional): 原始设计上下文信息，包含选定开篇方式、参考风格范式、结构组织逻辑、篇幅控制策略
    Returns:
        str | None: 如果需要修正则返回对事实段落的反馈，否则返回None
    """
    # 获取全局长度跟踪器
    tracker = get_global_length_tracker()
    
    # 1. 使用独立长度跟踪器进行校验（但仍保留原有逻辑用于迭代控制）
    is_length_valid, length_feedback_tracker, dynamic_length_info = tracker.validate_content_length(
        brief_content, "macro_review", tracker.macro_iteration_count,
        {"has_context_summary": bool(draft_context_summary)}
    )
    
    # 2. 保持原有的程序化长度校验逻辑（用于迭代控制）
    length_feedback = validate_and_instruct_content_length(brief_content)

    # 计算当前内容长度用于显示
    from utils import calculate_length_by_mode, LengthMode
    current_length = calculate_length_by_mode(brief_content, LengthMode.SEMANTIC_WITH_PUNCT)
    
    # 统一终端反馈显示格式
    print(f"\n=== 长度校验结果（574字符标准） ===")
    print(f"当前内容长度: {current_length} 字符")
    if length_feedback:
        print(f"校验反馈: {length_feedback.strip()}")
    else:
        print("校验状态: 长度符合要求（574字符标准）")
    print("=" * 40)
    
    # 3. 计算准确字符数用于动态提示（统一使用字符标准）
    actual_length = calculate_length_by_mode(brief_content, LengthMode.SEMANTIC_WITH_PUNCT)
    length_status = "符合要求" if length_feedback is None else "需要调整"
    
    # 4. 构建动态长度信息（使用574字符标准）
    if not dynamic_length_info:
        dynamic_length_info = f"""
### 当前内容长度信息 ###
- **精确字符数**: {actual_length} 字符
- **目标范围**: 400-550 字符（理想目标：574字符）
- **长度状态**: {length_status}
- **574标准**: 使用字符计算（包含标点符号）
"""
    
    if length_feedback:
        dynamic_length_info += f"\n- **程序化校验结果**: {length_feedback.strip()}"

    # 5. 执行LLM语义校验（传入动态长度信息和上下文）
    system_message = get_prompt('c_review_brief_content', dynamic_length_info=dynamic_length_info, draft_context_summary=draft_context_summary or "")
    user_message = f"<source_text>\n{source_text}\n</source_text>\n<brief_content>\n{brief_content}\n</brief_content>"
    if draft_context_summary:
        user_message += f"\n<draft_context_summary>\n{draft_context_summary}\n</draft_context_summary>"
    # 验证LLM输入
    is_valid_input, input_error = validate_llm_input(system_message, user_message, "review_brief_content")
    if not is_valid_input:
        logger.error(f"review_brief_content输入验证失败: {input_error}")
        return "输入验证失败，无法进行宏观审查"
    
    for attempt in range(3):
        try:
            llm_result = await call_llm(
                system_message, user_message, "openai/gpt-oss-120b", 0, 0.85, 0, 0,
                stage="macro_review",
                iteration_info={
                    "macro_iteration": tracker.macro_iteration_count,
                    "attempt": attempt + 1,
                    "max_attempts": 3
                }
            )
            
            # 验证LLM输出
            is_valid_output, output_error = validate_llm_output(llm_result, ["feedback_on_brief_content", "corrections_required"], "review_brief_content")
            if not is_valid_output:
                logger.warning(f"review_brief_content输出验证失败: {output_error}")
                continue
            
            feedback_on_brief_content, corrections_required = extract_with_xml(llm_result, ["feedback_on_brief_content", "corrections_required"])
            if feedback_on_brief_content and corrections_required is not None:
                # 5. 合并反馈信息
                final_feedback = ""
                needs_refinement = False

                # 判断是否需要修正：LLM明确指出需要修正 OR 存在长度问题
                llm_needs_correction = corrections_required.strip().lower() == "true"
                has_length_issue = length_feedback is not None
                
                if llm_needs_correction or has_length_issue:
                    needs_refinement = True
                    
                    # 优先处理长度问题
                    if has_length_issue:
                        final_feedback += length_feedback
                    
                    # 添加语义反馈（仅当LLM明确指出需要修正时）
                    if llm_needs_correction:
                        if final_feedback:
                            final_feedback += "\n\n--- 语义与事实核查反馈 ---\n"
                        final_feedback += feedback_on_brief_content

                # 关键修复：只有当确实需要修正时才返回反馈
                return final_feedback if needs_refinement else None
            # 如果提取失败，记录错误并重试
            log_exception(
                "review_brief_content", 
                attempt + 1, 
                3, 
                Exception("XML提取失败"), 
                {
                    "source_text_length": len(source_text),
                    "brief_content_length": len(brief_content),
                    "llm_result_available": bool(llm_result),
                    "llm_result_length": len(llm_result) if llm_result else 0
                }
            )
        except Exception as e:
            log_exception(
                "review_brief_content", 
                attempt + 1, 
                3, 
                e, 
                {
                    "source_text_length": len(source_text),
                    "brief_content_length": len(brief_content),
                    "system_message_length": len(system_message),
                    "user_message_length": len(user_message)
                }
            )
            # 可选：增加指数退避
            import asyncio
            await asyncio.sleep(2 ** attempt)
    
    # 如果LLM调用完全失败，但存在长度问题，仍需返回长度反馈
    return length_feedback if length_feedback else None
async def review_brief_sentences(source_text: str, brief_content: str, draft_context_summary: str = None) -> str | None:
    """
    校对文章的语句表达
    Args:
        source_text (str): 文章内容
        brief_content (str): 事实段落
        draft_context_summary (str, optional): 原始设计上下文信息，包含选定开篇方式、参考风格范式、结构组织逻辑、篇幅控制策略
    Returns:
        str | None: 如果需要修正则返回对语句表达的反馈，否则返回None
    """
    # 获取全局长度跟踪器并执行长度校验（微观审查阶段）
    tracker = get_global_length_tracker()
    is_valid, length_feedback, dynamic_length_info = tracker.validate_content_length(
        brief_content, "micro_review", tracker.micro_iteration_count + 1
    )
    
    # 优化终端反馈显示
    print(f"\n=== 微观审查长度校验 ===")
    print(f"当前内容长度: {tracker.current_length} 字符")
    if length_feedback:
        print(f"长度反馈: {length_feedback.strip()}")
    else:
        print("长度状态: 符合574字符标准")
    print(f"阶段信息: 微观第{tracker.micro_iteration_count}轮")
    print("=" * 40)
    
    # 优化长度校验逻辑：只有在强制修正区才返回长度反馈
    # 如果是建议优化区（中等优先级），则允许通过微观审查
    if not is_valid and length_feedback:
        # 检查是否为强制修正区（最高优先级）
        if "最高优先级" in length_feedback and "必须修正" in length_feedback:
            print(f"[调试] 长度严重不符合要求（强制修正区），直接返回长度反馈")
            print(f"微观审查发现严重长度问题: {length_feedback}")
            return length_feedback
        elif "中等优先级" in length_feedback and "建议优化" in length_feedback:
            print(f"[调试] 长度在建议优化区，允许通过微观审查，避免无限迭代")
            print(f"微观审查发现轻微长度问题，但允许通过: {length_feedback}")
            # 在建议优化区时，不返回长度反馈，允许微观审查通过
        else:
            # 其他情况，为了安全起见，返回长度反馈
            print(f"[调试] 长度校验不通过，返回长度反馈")
            print(f"微观审查发现长度问题: {length_feedback}")
        return length_feedback
    
    # 导入json_format并预计算json.dumps结果
    import importlib
    d_module = importlib.import_module('prompts.d_review_brief_sentences')
    json_format = d_module.json_format
    json_dumps_result = json.dumps(json_format, indent=4, ensure_ascii=False)
    
    system_message = get_prompt('d_review_brief_sentences', 
                               draft_context_summary=draft_context_summary or "",
                               json=json,
                               json_format=json_format,
                               json_dumps_result=json_dumps_result)
    user_message = f"<source_text>\n{source_text}\n</source_text>\n<brief_sentences>\n{split_to_sentences(brief_content)}\n</brief_sentences>"
    if draft_context_summary:
        user_message += f"\n<draft_context_summary>\n{draft_context_summary}\n</draft_context_summary>"
    
    # 验证LLM输入
    is_valid_input, input_error = validate_llm_input(system_message, user_message, "review_brief_sentences")
    if not is_valid_input:
        logger.error(f"review_brief_sentences输入验证失败: {input_error}")
        return "输入验证失败，无法进行微观审查"
    
    for attempt in range(3):
        try:
            llm_result = await call_llm(
                system_message, user_message, "openai/gpt-oss-120b", 0, 0.85, 0, 0,
                stage="micro_review",
                iteration_info={
                    "micro_iteration": tracker.micro_iteration_count,
                    "attempt": attempt + 1,
                    "max_attempts": 3
                }
            )
            
            # 验证LLM输出
            is_valid_output, output_error = validate_llm_output(llm_result, ["feedback_on_brief_sentences", "corrections_required"], "review_brief_sentences")
            if not is_valid_output:
                logger.warning(f"review_brief_sentences输出验证失败: {output_error}")
                continue
            
            feedback_on_brief_sentences, corrections_required = extract_with_xml(llm_result, ["feedback_on_brief_sentences", "corrections_required"])
            if feedback_on_brief_sentences is not None and corrections_required is not None:
                # 增强布尔值判断逻辑，支持多种格式
                corrections_required_clean = corrections_required.strip().lower()
                will_refine = corrections_required_clean in ["true", "是", "需要", "1", "yes"]
                
                # 添加调试日志
                print(f"微观审查决策: corrections_required='{corrections_required}' -> will_refine={will_refine}")
                
                # 如果布尔值格式异常，记录警告但继续处理
                if corrections_required_clean not in ["true", "false", "是", "否", "需要", "不需要", "1", "0", "yes", "no"]:
                    logger.warning(f"微观审查中corrections_required格式异常: '{corrections_required}'，默认为需要修正")
                    will_refine = True
                try:
                    monitor = get_global_monitor()
                    monitor.log_stage_context(
                        stage="micro_review.decision",
                        context={
                            "micro_iteration": tracker.micro_iteration_count + 1,
                            "will_refine": will_refine,
                            "feedback_preview": (feedback_on_brief_sentences or "")[:200]
                        },
                        level=LogLevel.DEBUG
                    )
                except Exception:
                    pass
                
                if will_refine:
                    print(f"微观审查发现问题: {feedback_on_brief_sentences}")
                    return feedback_on_brief_sentences
                else:
                    print("微观审查通过: 语句表达符合要求")
                    return None
        except Exception as e:
            log_exception(
                "review_brief_sentences", 
                attempt + 1, 
                3, 
                e, 
                {
                    "source_text_length": len(source_text),
                    "brief_content_length": len(brief_content),
                    "system_message_length": len(system_message),
                    "user_message_length": len(user_message),
                    "llm_result_available": 'llm_result' in locals() and bool(llm_result)
                }
            )
            # 如果是最后一次尝试，直接返回None终止迭代
            if attempt == 2:  # 第3次尝试（索引为2）
                print("微观审查异常: 多次尝试后仍无法获取有效结果，终止迭代")
                return None
    
    # 所有尝试都失败，返回None终止迭代
    print("微观审查异常: 多次尝试后仍无法获取有效结果，终止迭代")
    return None

async def refine_brief_content(brief_content: str, feedback_on_brief_content: str | None, feedback_on_brief_sentences: str | None, draft_context_summary: str = None) -> tuple[str, str]:
    """
    根据反馈修改简报内容
    Args:
        brief_content (str): 简报内容
        feedback_on_brief_content (str | None): 对事实段落的反馈
        feedback_on_brief_sentences (str | None): 对语句表达的反馈
        draft_context_summary (str, optional): 原始设计上下文信息，包含选定开篇方式、参考风格范式、结构组织逻辑、篇幅控制策略
    Returns:
        tuple[str, str]: (修改后的简报内容, 更新后的上下文信息)
    """
    if not feedback_on_brief_content and not feedback_on_brief_sentences:
        return brief_content, draft_context_summary or ""
    
    # 获取全局长度跟踪器
    tracker = get_global_length_tracker()
    
    # 确定润色类型并显示开始信息
    refine_type = "宏观" if feedback_on_brief_content else "微观"
    # 修复：定义当前迭代计数
    current_iteration = tracker.macro_iteration_count if feedback_on_brief_content else tracker.micro_iteration_count
    print(f"\n 开始{refine_type}润色处理...")

    # 埋点：润色开始
    try:
        monitor = get_global_monitor()
        monitor.log_stage_context(
            stage="refine.start",
            context={
                "refine_type": refine_type,
                "current_iteration": current_iteration,
                "brief_length": len(brief_content or ""),
                "has_macro_feedback": bool((feedback_on_brief_content or "").strip()),
                "has_micro_feedback": bool((feedback_on_brief_sentences or "").strip()),
                "macro_feedback_preview": (feedback_on_brief_content or "")[:200],
                "micro_feedback_preview": (feedback_on_brief_sentences or "")[:200],
                "context_summary_preview": (draft_context_summary or "")[:200]
            },
            level=LogLevel.DEBUG
        )
    except Exception:
        pass
    
    system_message = get_prompt('e_refine_brief_content', 
                                brief_content=brief_content,
                                feedback_on_brief_content=feedback_on_brief_content or "",
                                feedback_on_brief_sentences=feedback_on_brief_sentences or "",
                                draft_context_summary=draft_context_summary or "")
    
    # 构建结构化的user_message，提供完整上下文信息
    user_message = f"""<brief_content>
{brief_content or ""}
</brief_content>

<feedback_on_brief_content>
{feedback_on_brief_content or ""}
</feedback_on_brief_content>

<feedback_on_brief_sentences>
{feedback_on_brief_sentences or ""}
</feedback_on_brief_sentences>

<draft_context_summary>
{draft_context_summary or ""}
</draft_context_summary>

<constraints>
- 必须保持简报的核心事实准确性
- 字符数控制在574字符左右
- 保持专业的财经简报风格
</constraints>

<iteration_info>
当前{refine_type}润色第{current_iteration}轮，请根据上述反馈进行精准修改。
</iteration_info>"""
    
    # 验证LLM输入
    is_valid_input, input_error = validate_llm_input(system_message, user_message, "refine_brief_content")
    if not is_valid_input:
        logger.error(f"refine_brief_content输入验证失败: {input_error}")
        return brief_content, draft_context_summary or ""
    
    for attempt in range(3):
        try:
            llm_result = await call_llm(
                system_message, user_message, "openai/gpt-oss-120b", 0.6, 0.85, 0, 0,
                stage="refine_content",
                iteration_info={
                    "refine_type": refine_type,
                    "current_iteration": current_iteration,
                    "attempt": attempt + 1,
                    "max_attempts": 3
                }
            )
            
            # 验证LLM输出
            is_valid_output, output_error = validate_llm_output(llm_result, ["refined_brief_content", "updated_draft_context_summary"], "refine_brief_content")
            if not is_valid_output:
                logger.warning(f"refine_brief_content输出验证失败: {output_error}")
                # 失败可见性：埋点输出校验失败
                try:
                    monitor = get_global_monitor()
                    monitor.log_stage_context(
                        stage="refine.validation_fail",
                        context={
                            "refine_type": refine_type,
                            "current_iteration": current_iteration,
                            "attempt": attempt + 1,
                            "error": str(output_error)[:300]
                        },
                        level=LogLevel.WARNING
                    )
                except Exception:
                    pass
                continue
            
            refined_brief_content = extract_with_xml(llm_result, "refined_brief_content")
            updated_context_summary = extract_with_xml(llm_result, "updated_draft_context_summary")
            
            if refined_brief_content:
                # 无变化保护：若仅为空白差异或完全一致，则视为无实质修改，进行下一次重试
                try:
                    norm_old = re.sub(r"\s+", " ", (brief_content or "")).strip()
                    norm_new = re.sub(r"\s+", " ", (refined_brief_content or "")).strip()
                    if norm_new == norm_old:
                        logger.info("refine_brief_content: 模型输出与原文无实质变化，尝试下一次重试")
                        try:
                            monitor = get_global_monitor()
                            monitor.log_stage_context(
                                stage="refine.no_change",
                                context={
                                    "refine_type": refine_type,
                                    "current_iteration": current_iteration,
                                    "attempt": attempt + 1,
                                    "brief_length": len(brief_content or ""),
                                },
                                level=LogLevel.INFO
                            )
                        except Exception:
                            pass
                        continue
                except Exception:
                    # 归一化失败不阻断流程
                    pass
                # 计算当前迭代计数
                current_iteration = tracker.macro_iteration_count if feedback_on_brief_content else tracker.micro_iteration_count
                
                # 使用长度跟踪器记录修正后的内容
                is_valid, length_feedback, dynamic_info = tracker.validate_content_length(
                    refined_brief_content, "refine", current_iteration,
                    {
                        "original_length": len(brief_content), 
                        "feedback_provided": bool((feedback_on_brief_content or "").strip() or (feedback_on_brief_sentences or "").strip()),
                        "refine_type": refine_type
                    }
                )
                
                # 显示润色完成信息
                original_length = len(brief_content)
                new_length = len(refined_brief_content)
                length_change = new_length - original_length
                change_indicator = "增加" if length_change > 0 else "减少" if length_change < 0 else "不变"
                
                print(f"\n {refine_type}润色完成")
                print(f"长度变化: {original_length} → {new_length} 字符 {change_indicator} ({length_change:+d})")
                
                # 埋点：润色结束
                try:
                    monitor = get_global_monitor()
                    monitor.log_stage_context(
                        stage="refine.end",
                        context={
                            "refine_type": refine_type,
                            "current_iteration": current_iteration,
                            "original_length": original_length,
                            "new_length": new_length,
                            "length_change": length_change,
                            "length_valid": bool(is_valid),
                            "length_feedback": (length_feedback or "").strip()[:200] if length_feedback else None,
                            "dynamic_info": (dynamic_info or "")[:200] if dynamic_info else None,
                            "updated_context_summary_preview": (updated_context_summary or "")[:200]
                        },
                        level=LogLevel.DEBUG
                    )
                except Exception:
                    pass
                
                return refined_brief_content, updated_context_summary or draft_context_summary or ""
        except Exception as e:
            log_exception(
                "refine_brief_content", 
                attempt + 1, 
                3, 
                e, 
                {
                    "brief_content_length": len(brief_content),
                    "feedback_on_brief_content_length": len(feedback_on_brief_content or ""),
                    "feedback_on_brief_sentences_length": len(feedback_on_brief_sentences or ""),
                    "system_message_length": len(system_message),
                    "refine_type": refine_type,
                    "llm_result_available": 'llm_result' in locals() and bool(llm_result)
                }
            )
    
    print(f"\n  {refine_type}润色失败，返回原始内容")
    try:
        monitor = get_global_monitor()
        monitor.log_stage_context(
            stage="refine.fail",
            context={
                "refine_type": refine_type,
                "current_iteration": current_iteration,
                "brief_length": len(brief_content or ""),
                "context_summary_preview": (draft_context_summary or "")[:200]
            },
            level=LogLevel.WARNING
        )
    except Exception:
        pass
    return brief_content, draft_context_summary or ""

async def draft_and_refine_brief_content(source_text: str, interpretation: str, article_contents: str, draft_context_summary: str = "") -> tuple[str, str]:
    """
    通过"撰写-校对-修改"的结构化迭代循环来创建并完善简报内容。
    该函数首先生成一个草稿，然后进入一个两阶段的串行审查流程：
    1. 宏观迭代（外循环）：专注于事实准确性、结构完整性和长度合规性。
    2. 微观迭代（内循环）：在宏观稳定的基础上，专注于语言逻辑、措辞和文体风格。
    只有当两个阶段的审查都通过后，才会返回最终的简报和更新后的上下文摘要。

    Args:
        source_text (str): 解读后的原文。
        interpretation (str): 对原文的解读。
        article_contents (str): 匹配到的历史文章内容。
        draft_context_summary (str, optional): 初始设计上下文摘要，默认为空字符串。

    Returns:
        tuple[str, str]: (最终完善后的简报内容, 更新后的上下文摘要)
    """
    # 获取全局长度跟踪器并重置迭代计数
    tracker = get_global_length_tracker()
    tracker.reset_tracker()
    
    # 获取全局迭代管理器并初始化
    iteration_manager = get_global_iteration_manager()
    iteration_manager.reset()
    iteration_manager.set_stage(WorkflowStage.REFINE_CONTENT)
    
    brief_content, current_context_summary = await draft_brief_content(source_text, interpretation, article_contents)
    
    # 初始化上下文摘要，如果未提供则使用空字符串
    current_context_summary = current_context_summary or draft_context_summary or ""
    
    # 埋点：进入宏观审查主循环之前
    try:
        get_global_monitor().log_stage_context(
            stage="macro.loop.enter",
            context={
                "max_iterations": 3,
                "initial_length": len(brief_content or ""),
                "has_context_summary": bool(current_context_summary),
                "context_summary_preview": (current_context_summary or "")[:200]
            },
            level=LogLevel.INFO
        )
    except Exception:
        pass
    
    # --- 外循环：宏观结构与事实审查 ---
    max_iterations_macro = 3 # 设置宏观迭代上限，防止无限循环
    macro_review_passed = False  # 添加标志位跟踪宏观审查状态
    
    # 验证迭代管理器状态
    is_valid, errors = iteration_manager.validate_state()
    if not is_valid:
        print("\n[警告] 迭代管理器状态异常，重置后继续")
        for error in errors:
            print(f"  错误: {error}")
        iteration_manager.reset()
        iteration_manager.set_stage(WorkflowStage.REFINE_CONTENT)
    
    for i in range(max_iterations_macro):
        # 开始新的宏观迭代
        iteration_count, can_continue = iteration_manager.start_macro_iteration()
        
        # 同步更新tracker的计数（保持兼容性）
        tracker.macro_iteration_count = iteration_count
        
        print(f"\n{'='*50}")
        print(f" 开始宏观审查（第 {iteration_count}/{max_iterations_macro} 轮）")
        print(f"{'='*50}")
        
        # 埋点：每轮宏观开始
        try:
            get_global_monitor().log_stage_context(
                stage="macro.loop.round_start",
                context={
                    "round": tracker.macro_iteration_count,
                    "max_iterations": max_iterations_macro,
                    "current_length": len(brief_content or ""),
                    "context_summary_preview": (current_context_summary or "")[:200]
                },
                level=LogLevel.DEBUG
            )
        except Exception:
            pass
        
        feedback_on_content = await review_brief_content(source_text, brief_content, current_context_summary)
        
        if feedback_on_content is None:
            print("\n 宏观审查通过，内容结构和事实准确性达标。")
            iteration_manager.complete_macro_iteration(is_passed=True)
            macro_review_passed = True
            break  # 退出外循环，进入微观审查
        
        print(f"\n  宏观审查发现问题，正在进行修订...")
        print(f"反馈摘要: {feedback_on_content[:100]}...")
        
        # 调用refine时，只传入宏观反馈，微观反馈为None
        refined_brief_content, current_context_summary = await refine_brief_content(brief_content, feedback_on_content, None, current_context_summary)
        
        print("\n 宏观结构已根据反馈进行修改，将进行新一轮宏观校对...")
        brief_content = refined_brief_content
        
        # 标记当前迭代完成但需要继续
        iteration_manager.complete_macro_iteration(is_passed=False)
    
    if not macro_review_passed:
        print("\n  已达到宏观审查最大迭代次数，强制进入微观审查。")

    # --- 内循环：微观语言与逻辑审查 ---
    max_iterations_micro = 3 # 设置微观迭代上限，防止无限循环
    micro_review_passed = False  # 添加标志位跟踪微观审查状态
    
    # 验证迭代管理器状态（微观审查前）
    if iteration_manager.micro_iteration_count >= iteration_manager.max_micro_iterations:
        print("\n[警告] 微观迭代已达上限，重置微观计数")
        iteration_manager.micro_iteration_count = 0
    
    for i in range(max_iterations_micro):
        # 开始新的微观迭代
        iteration_count, can_continue = iteration_manager.start_micro_iteration()
        
        # 同步更新tracker的计数（保持兼容性）
        tracker.micro_iteration_count = iteration_count
        
        print(f"\n{'='*50}")
        print(f" 开始微观审查（第 {iteration_count}/{max_iterations_micro} 轮）")
        print(f"{'='*50}")
        print(f"[调试] 当前简报内容长度: {len(brief_content)}")
        print(f"[调试] 当前上下文摘要长度: {len(current_context_summary or '')}")
        
        # 修正了review_brief_sentences的调用，传入了source_text
        print(f"[调试] 调用review_brief_sentences函数...")
        feedback_on_sentences = await review_brief_sentences(source_text, brief_content, current_context_summary)
        
        print(f"[调试] review_brief_sentences返回结果: {type(feedback_on_sentences)}")
        if feedback_on_sentences is not None:
            print(f"[调试] 反馈内容长度: {len(feedback_on_sentences)}")
            print(f"[调试] 反馈内容预览: {feedback_on_sentences[:200]}...")
        else:
            print(f"[调试] 返回None，微观审查通过")
        
        if feedback_on_sentences is None:
            print("\n 微观审查通过，语言和逻辑无误。简报最终版本确认。")
            iteration_manager.complete_micro_iteration(is_passed=True)
            micro_review_passed = True
            break  # 退出微观审查循环
            
        print(f"\n  微观审查发现问题，正在进行润色...")
        print(f"反馈摘要: {feedback_on_sentences[:100]}...")
        
        # 调用refine时，只传入微观反馈，宏观反馈为None
        print(f"[调试] 调用refine_brief_content进行润色...")
        refined_brief_content, current_context_summary = await refine_brief_content(brief_content, None, feedback_on_sentences, current_context_summary)
        
        print(f"[调试] 润色后内容长度: {len(refined_brief_content)}")
        print("\n简报文本已根据反馈进行润色，将进行一轮微观校对...")
        brief_content = refined_brief_content
        
        # 标记当前迭代完成但需要继续
        iteration_manager.complete_micro_iteration(is_passed=False)
    
    if not micro_review_passed:
        print("\n  已达到微观审查最大迭代次数，返回当前最终版本。")
    
    # 获取最终迭代统计信息
    final_stats = iteration_manager.get_statistics()
    print(f"\n{'='*60}")
    print(" 迭代统计摘要")
    print(f"{'='*60}")
    print(f"宏观审查: {final_stats['iteration_counts']['macro_iterations']}/{final_stats['iteration_limits']['max_macro_iterations']} 轮")
    print(f"微观审查: {final_stats['iteration_counts']['micro_iterations']}/{final_stats['iteration_limits']['max_micro_iterations']} 轮")
    total_passed = final_stats['history_summary']['passed_iterations']
    total_records = final_stats['history_summary']['total_records']
    success_rate = (total_passed / total_records * 100) if total_records > 0 else 0
    print(f"总成功率: {success_rate:.1f}%")
    
    # 验证最终状态
    is_valid, errors = iteration_manager.validate_state()
    if not is_valid:
        print("\n[警告] 迭代管理器最终状态验证失败")
        for error in errors:
            print(f"  错误: {error}")
    
    # 埋点：流程结束
    try:
        get_global_monitor().log_stage_context(
            stage="workflow.summary",
            context={
                "macro_review_passed": macro_review_passed,
                "micro_review_passed": micro_review_passed,
                "final_length": len(brief_content or ""),
                "context_summary_preview": (current_context_summary or "")[:200],
                "iteration_stats": final_stats
            },
            level=LogLevel.INFO
        )
    except Exception:
        pass
    
    return brief_content, current_context_summary

async def draft_brief_title(brief_content: str, article_titles: str) -> str:
    """
    生成文章的标题
    Args:
        brief_content (str): 简报内容
    Returns:
        str: 标题
    """
    system_message = get_prompt('g_draft_brief_title').format(article_titles=article_titles)
    print(system_message)
    user_message = f"<brief_content>\n{brief_content}\n</brief_content>"
    for attempt in range(3):
        try:
            llm_result = await call_llm(
                system_message, user_message, "google/gemini-2.5-pro", 0.5, 0.95, 0, 0.5,
                stage="draft_title",
                iteration_info={"attempt": attempt + 1, "max_attempts": 3}
            )
            if brief_title := extract_with_xml(llm_result, "brief_title"):
                return brief_title
        except Exception:
            pass

async def review_brief_title(brief_content: str, brief_title: str) -> str | None:
    """
    校对文章的标题
    Args:
        brief_content (str): 简报内容
        brief_title (str): 简报标题
    Returns:
        str | None: 如果需要修正则返回对标题的反馈，否则返回None
    """
    system_message = get_prompt('h_review_brief_title')
    user_message = f"<brief_content>\n{brief_content}\n</brief_content>\n<brief_title>\n{brief_title}\n</brief_title>"
    adjust_length_prompt = get_prompt('k_adjust_length', text=brief_title)
    for attempt in range(3):
        try:
            llm_result = await call_llm(
                system_message, user_message, "qwen/qwen3-235b-a22b-2507", 0.1, 0.95, 0, 0,
                stage="review_title",
                iteration_info={"attempt": attempt + 1, "max_attempts": 3}
            )
            feedback_on_brief_title, corrections_required = extract_with_xml(llm_result, ["feedback_on_brief_title", "corrections_required"])
            if feedback_on_brief_title and corrections_required is not None:
                # 统一布尔值判断逻辑，确保与其他review函数一致
                if corrections_required.strip().lower() == "true":
                   return feedback_on_brief_title + adjust_length_prompt
                elif adjust_length_prompt:
                    return adjust_length_prompt
                else:
                    return None
        except Exception:
            pass
    return None

async def refine_brief_title(brief_title: str, feedback_on_brief_title: str | None) -> str | None:
    """
    修改文章的标题
    Args:
        brief_title (str): 标题
        feedback_on_brief_title (str | None): 对标题的反馈
    Returns:                    
        str | None: 修改后的标题，如果无需修改则返回None
    """
    if not feedback_on_brief_title:
        return None
    system_message = get_prompt('i_refine_brief_title')
    user_message = f"<brief_title>\n{brief_title}\n</brief_title>\n<feedback_on_brief_title>\n{feedback_on_brief_title}\n</feedback_on_brief_title>"
    for attempt in range(3):
        try:
            llm_result = await call_llm(
                system_message, user_message, "qwen/qwen3-235b-a22b-2507", 0.3, 0.5, 0, 0,
                stage="refine_title",
                iteration_info={"attempt": attempt + 1, "max_attempts": 3}
            )
            if refined_brief_title := extract_with_xml(llm_result, "refined_brief_title"):
                return refined_brief_title
        except Exception:
            pass
    return None

async def draft_and_refine_brief_title(brief_content: str, article_titles: str) -> str:
    """
    通过"撰写-校对-修改"的循环来创建并完善标题。
    该函数首先生成一个标题草稿，然后反复进行校对和修改，
    直到校对反馈表明无需任何修改为止。
    Args:
        brief_content (str): 最终的简报内容。
    Returns:
        str: 最终完善后的标题。
    """
    brief_title = await draft_brief_title(brief_content, article_titles)
    while True:
        feedback_on_brief_title = await review_brief_title(brief_content, brief_title)
        refined_brief_title = await refine_brief_title(brief_title, feedback_on_brief_title)
        if refined_brief_title is None:
            if get_prompt('k_adjust_length', text=brief_title):
                continue        
            print("标题校对完成，无需进一步修改。")
            return brief_title
        else:
            print("标题已根据反馈进行修改，将进行一轮校对...")
            brief_title = refined_brief_title

def get_all_keywords_and_tags(file_name: str) -> list:
    """
    从CSV文件中提取关键词和标签数据
    
    Args:
        file_name (str): 文件名（包含.csv扩展名）
    Returns:
        list: 包含每行数据字典的列表
    """
    results = []
    with open(f"{file_name}.csv", 'r', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            results.append({
                "DataID": row["DataID"],
                "political_and_economic_terms": ast.literal_eval(row["political_and_economic_terms"]),
                "technical_terms": ast.literal_eval(row["technical_terms"]),
                "other_abstract_concepts": ast.literal_eval(row["other_abstract_concepts"]),
                "organizations": ast.literal_eval(row["organizations"]),
                "persons": ast.literal_eval(row["persons"]),
                "cities_or_districts": ast.literal_eval(row["cities_or_districts"]),
                "other_concrete_entities": ast.literal_eval(row["other_concrete_entities"]),
                "other_tags_of_topic_or_points": ast.literal_eval(row["other_tags_of_topic_or_points"])
            })
    return results

def get_matched_articles(new_keywords_and_tags: set, all_keywords_and_tags: list, file_name: str) -> list:
    """
    计算新文章关键词与所有文章关键词的匹配程度，并返回匹配文章的详细信息
    
    Args:
        new_keywords_and_tags (set): 新文章的关键词和标签集合
        all_keywords_and_tags (list): 所有文章的关键词和标签字典列表
        file_name (str): 文件名（不包含.csv扩展名）
    
    Returns:
        list: 按匹配分数从高到低排序的字典列表，包含DataID、InfoTitle、InfoContent
    """
    weights = {
        "political_and_economic_terms": 1,
        "technical_terms": 1,
        "other_abstract_concepts": 1,
        "organizations": 1,
        "persons": 1,
        "cities_or_districts": 0.5,
        "other_concrete_entities": 1,
        "other_tags_of_topic_or_points": 1
    }
    id_and_score_tuples = [(keywords_and_tags["DataID"], sum(len(new_keywords_and_tags.intersection(set(keywords_and_tags[aspect]))) * weight for aspect, weight in weights.items())) for keywords_and_tags in all_keywords_and_tags]
    id_to_score = {id: score for id, score in id_and_score_tuples if score > 1}
    results = []
    with open(f"{file_name}.csv", 'r', encoding='utf-8') as file:
        for row in csv.DictReader(file):
            if row["DataID"] in id_to_score:
                results.append({
                    "DataID": row["DataID"],
                    "InfoTitle": row["InfoTitle"],
                    "InfoContent": row["InfoContent"],
                    "ProductDate": row["ProductDate"],
                    "score": id_to_score[row["DataID"]]
                })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]

async def match_articles(article: str, file_name: str) -> list:
    """
    匹配文章函数，传入文章字符串，返回匹配的文章结果
    
    Args:
        article (str): 输入的文章内容
        file_name (str): 要匹配的CSV文件名（不包含.csv扩展名），默认为"reference_text"
    
    Returns:
        list: 按匹配分数从高到低排序的字典列表，包含DataID、InfoTitle、InfoContent、score
    """
    # 步骤1: 使用embedding获取与新文章相似的标签
    new_keywords_and_tags = get_similar_tags(article, top_n=100)
    # 步骤2: 获取所有文章的关键词和标签数据
    all_keywords_and_tags = get_all_keywords_and_tags(file_name)
    # 步骤3: 计算匹配度并返回结果
    matched_articles = get_matched_articles(new_keywords_and_tags, all_keywords_and_tags, file_name)
    return matched_articles

async def translate_to_other_languages(brief_title: str, brief_content: str) -> tuple[str, str, str, str, str, str, str, str]:
    """
    翻译为其它语言
    Args:
        brief_title (str): 简报标题
        brief_content (str): 简报内容
    Returns:
        tuple: 其它语言的简报的标题和内容
    """
    system_message = get_prompt('l_translate_to_other_languages')
    user_message = f"<chinese_title>\n{brief_title}\n</chinese_title>\n<chinese_content>\n{brief_content}\n</chinese_content>"
    for attempt in range(3):
        try:
            llm_result = await call_llm(
                system_message, user_message, "openai/gpt-oss-120b", 0.1, 0.5, 0, 0,
                stage="translate",
                iteration_info={"attempt": attempt + 1, "max_attempts": 3}
            )
            if brief_in_other_languages := extract_with_xml(llm_result, ["english_title", "english_content", "german_title", "german_content", "french_title", "french_content", "japanese_title", "japanese_content"]):
                return tuple(brief_in_other_languages)
        except Exception:
            pass

async def generate_briefs() -> None:
    """
    执行完整的并行简报生成工作流，处理inputs文件夹中的所有txt文件
    Returns:
        None
    """
    async def generate_brief(file_name: str) -> None:
        """
        处理单个文件的完整简报生成工作流
        Args:
            file_name (str): 输入文件名 (不包含.txt扩展名)
        Returns:
            None
        """
        # 导入LLM监控器并记录工作流开始
        from llm_monitor import LLMMonitor, LogLevel
        llm_monitor = LLMMonitor()
        llm_monitor.log_workflow_stage(f"generate_brief_{file_name}", f"开始处理文件: {file_name}", LogLevel.INFO)
        
        print(f"开始处理文件: {file_name}")
        print(f"[{file_name}] 1/6: 读取并解读原文")
        source_text = get_source_text(file_name)
        interpretation, key_points, published_date = await interpret_source_text(source_text)
        
        # 检查解读结果是否有效
        if interpretation is None or key_points is None or published_date is None:
            logger.error(f"文件 {file_name} 的源文本解读失败，跳过处理")
            return
            
        interpretation = convert_to_date(interpretation, published_date)
        source_text = convert_to_cn_term(source_text)
        source_text = clean_stock_codes(source_text)
        source_text = convert_to_date(source_text, published_date)
        print(f"[{file_name}] 2/6: 匹配历史文章")
        articles = await match_articles(key_points, "reference_text")
        article_contents = "\n\n".join([re.sub(r'（[A-Za-z]+）$', '', article['InfoContent']).strip() for article in articles[:3]])
        article_titles = "\n\n".join([(article['InfoTitle'].split('：', 1)[1].strip() if '：' in article['InfoTitle'] else article['InfoTitle']) for article in articles])
        print(f"[{file_name}] 3/6: 创建并完善简报内容")
        brief_content, final_context_summary = await draft_and_refine_brief_content(source_text, interpretation, article_contents)
        brief_content = remove_year_at_start(brief_content)
        print(f"[{file_name}] 5/6: 创建并完善简报标题")
        brief_title = await draft_and_refine_brief_title(brief_content, article_titles)
        save_as_txt(f"{brief_title}\n{brief_content}\n\n{"\n\n".join([f"{article['InfoTitle']}\n{article['InfoContent']}\n{article['ProductDate']}" for article in articles])}", f"{file_name} with matched briefs")
        # print(f"[{file_name}] 6/6: 翻译为其它语言")
        # english_title, english_content, german_title, german_content, french_title, french_content, japanese_title, japanese_content = await translate_to_other_languages(brief_title, brief_content)
        # save_as_txt(f"{brief_title}\n{brief_content}\n\n{english_title}\n{english_content}\n\n{german_title}\n{german_content}\n\n{french_title}\n{french_content}\n\n{japanese_title}\n{japanese_content}", f"{file_name} with other language versions")
        
        # 输出LLM监控会话摘要
        print(f"\n{'='*60}")
        print(f" LLM调用监控摘要 - {file_name}")
        print(f"{'='*60}")
        llm_monitor.print_session_summary()
    if file_names := get_source_files():
        print(f"发现 {len(file_names)} 个文件，开始并行处理: {file_names}")
        await asyncio.gather(*[generate_brief(file_name) for file_name in file_names])
        print(f"所有 {len(file_names)} 个文件处理完成")
    return None

if __name__ == "__main__":
    asyncio.run(generate_briefs())
"""
    # 定义测试用的简报内容
    query = "中国的国内生产总值GDP保持快速增长，但居民感受到的获得感并不明显，原因分析"
    # 测试文件名
    file_name = "分析专栏2021-07至2025-06"
    print("开始测试match_articles函数...")
    print(f"测试简报内容长度: {len(query)} 字符")
    print(f"匹配数据文件: {file_name}.csv")
    # 使用asyncio.run()运行异步函数
    articles = asyncio.run(match_articles(query, file_name))
    print(f"\n匹配到 {len(articles)} 篇相关文章")
    # 显示前5篇匹配文章的信息
    for i, article in enumerate(articles[:5], 1):
        print(f"\n第{i}篇匹配文章 (得分: {article['score']}):")
        print(f"{article['InfoTitle']}")
        print(f"{article['InfoContent']}")
    csv_path = "outputs/GDP与体感差异分析文章2021-07至2025-06.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["InfoTitle", "InfoContent", "ProductDate"])
        for article in articles:
            writer.writerow([article["InfoTitle"], article["InfoContent"], article["ProductDate"]])
    print(f"\n测试完成！匹配结果已保存到: {csv_path}")
"""