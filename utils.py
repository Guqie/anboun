# utils.py

import aiohttp
import asyncio
import os
import re
import json
from datetime import datetime, timedelta
import random
from typing import Optional, Tuple, Dict
import importlib
from enum import Enum
import time
import inspect
from llm_monitor import get_global_monitor, LLMCallStatus

OPENROUTER_API_KEY = ""

async def call_llm(system_message: str, user_message: str, model="openai/gpt-oss-120b", temperature=0.5, top_p=0.95, frequency_penalty=0, presence_penalty=0, stage: str = None, iteration_info: Dict = None) -> Optional[str]:
    """
    调用LLM API并记录详细的监控日志
    
    Args:
        system_message (str): 系统消息
        user_message (str): 用户消息
        model (str): 模型名称
        temperature (float): 温度参数
        top_p (float): top_p参数
        frequency_penalty (float): 频率惩罚
        presence_penalty (float): 存在惩罚
        stage (str): 当前处理阶段
        iteration_info (Dict): 迭代信息
    
    Returns:
        Optional[str]: LLM响应内容
    """
    # 获取监控器和调用信息
    monitor = get_global_monitor()
    call_id = monitor.generate_call_id()
    
    # 获取调用函数名
    caller_frame = inspect.currentframe().f_back
    function_name = caller_frame.f_code.co_name if caller_frame else "unknown"
    
    # 记录调用开始
    monitor.log_call_start(
        call_id=call_id,
        function_name=function_name,
        model=model,
        system_message=system_message,
        user_message=user_message,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stage=stage,
        iteration_info=iteration_info
    )
    
    api_key = OPENROUTER_API_KEY
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
        "model": model,
        "max_tokens": 8192,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty
    }
    
    start_time = time.time()
    response_content = None
    error_message = None
    attempt_count = 0
    
    try:
        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                attempt_count = attempt + 1
                try:
                    # 兼容测试中的 AsyncMock：不使用异步上下文管理器包裹 post
                    response = await session.post(url, headers=headers, json=data, ssl=False, timeout=aiohttp.ClientTimeout(total=120))
                    response_data = await response.json()
                    if content := response_data["choices"][0]["message"]["content"]:
                        response_content = content
                        print(content)  # 保持原有的控制台输出
                        break
                except Exception as e:
                    error_message = f"尝试 {attempt_count}: {str(e)}"
                    if attempt < 2:  # 不是最后一次尝试
                        await asyncio.sleep(1)  # 重试前等待
                    continue
    except Exception as e:
        error_message = f"整体调用异常: {str(e)}"
    
    # 计算响应时间
    response_time = time.time() - start_time
    
    # 确定调用状态
    if response_content:
        status = LLMCallStatus.SUCCESS
    elif response_time >= 120:  # 超时
        status = LLMCallStatus.TIMEOUT
        error_message = error_message or "请求超时"
    else:
        status = LLMCallStatus.FAILED
        error_message = error_message or "未知错误"
    
    # 记录调用结束
    monitor.log_call_end(
        call_id=call_id,
        function_name=function_name,
        model=model,
        system_message=system_message,
        user_message=user_message,
        response=response_content,
        response_time=response_time,
        attempt_count=attempt_count,
        status=status,
        error_message=error_message,
        stage=stage,
        iteration_info=iteration_info,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    
    return response_content

async def call_local_llm(system_message: str, user_message: str, model="qwen3:32b", temperature=0.5, top_p=0.95, frequency_penalty=0, presence_penalty=0) -> Optional[str]:
    """
    调用本地LLM（例如Ollama）的聊天接口，采用流式读取返回内容。
    说明：为兼容测试中的 AsyncMock，本函数不使用 `async with session.post(...)`，而是通过 `await session.post(...)` 获取响应对象。
    
    Args:
        system_message (str): 系统消息
        user_message (str): 用户消息
        model (str): 模型名称（本地模型标识）
        temperature (float): 采样温度
        top_p (float): nucleus sampling 参数
        frequency_penalty (float): 频率惩罚
        presence_penalty (float): 存在惩罚
    
    Returns:
        Optional[str]: 拼接后的完整响应文本
    """
    url = "http://localhost:11434/api/chat"
    data = {
        "messages": [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
        "model": model,
        "max_tokens": 8192,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": True
    }
    try:
        async with aiohttp.ClientSession() as session:
            # 兼容测试中的 AsyncMock：不使用异步上下文管理器包裹 post
            response = await session.post(url, headers={}, json=data, timeout=aiohttp.ClientTimeout(total=150))
            content = ""
            async for line in response.content:
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        content += chunk["message"]["content"]
                        print(chunk["message"]["content"], end="", flush=True)
                    except json.JSONDecodeError:
                        continue
            if content:
                print()
                return content
    except Exception:
        pass
    return None

def get_prompt(name: str, **arguments) -> str:
    """
    动态加载并返回指定名称的提示模板（prompt）。
    安全替换策略：
    - 如果模块中导出的同名对象是可调用（函数），则直接调用并传入参数，避免对模板进行字符串格式化，
      以防止模板中包含的花括号（例如JSON示例、表达式片段如"{json.dumps(...)}"）被误解析。
    - 如果导出的是字符串模板，则仅对提供的参数键执行“按键名的直接文本替换”（将"{key}"替换为对应的值），
      而不使用 str.format/format_map，以避免 KeyError/AttributeError 以及对非占位花括号的误处理。
    参数:
        name (str): 提示模板模块中同名对象的名称（例如 prompts.a_interpret_source_text 模块中的 a_interpret_source_text）
        **arguments: 需要在模板中替换的占位键值对，仅会替换形如"{key}"的占位符
    返回:
        str: 组装后的完整提示词
    """
    prompt_obj = getattr(importlib.import_module(f"prompts.{name}"), name)
    # 如果是函数，则直接调用（函数内部应自行处理参数拼接与花括号）
    if callable(prompt_obj):
        return prompt_obj(**arguments) if arguments else prompt_obj()

    # 否则是字符串模板，仅对提供的键执行安全替换
    prompt_text = str(prompt_obj)
    if arguments:
        for k, v in arguments.items():
            placeholder = "{" + str(k) + "}"
            prompt_text = prompt_text.replace(placeholder, str(v))
    return prompt_text

def extract_with_xml(text, tags):
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    tags = tags if isinstance(tags, list) else [tags]
    results = []
    for tag in tags:
        if matched := re.search(rf".*<{tag}>(?P<result>[\s\S]*)</{tag}>", text, re.DOTALL):
            results.append(matched.group("result").strip())
        else:
            return None
    return tuple(results) if len(results) > 1 else results[0]

def extract_from_json(data, keys):
    data = json.loads(data)
    keys = keys if isinstance(keys, list) else [keys]
    results = []
    for key in keys:
        if key in data:
            results.append(data[key])
        else:
            return None
    return tuple(results) if len(results) > 1 else results[0]

def save_as_txt(text, file_name):
    """
    统一的文本文件保存函数
    Args:
        text: 要保存的文本内容
        file_name: 文件名（不包含.txt扩展名）
    """
    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/{f"{file_name}_{random.randint(100000, 999999)}.txt"}"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"文本已保存到: {file_path}")

def replace_year_with_2025(text: str) -> str:
    return re.sub(r'^\d+-', '2025-', text)

def remove_year_at_start(text: str) -> str:
    return re.sub(r"^\d+年", "", text)

def split_to_sentences(text):
    if not text:
        return "" #返回空字符串

    positions = [i for i, char in enumerate(text) if char == '。'] + [len(text) - 1]
    return "\n\n".join(f"{i + 1}. {text[start:end + 1].strip()}" for i, (start, end) in enumerate(zip([0] + [position + 1 for position in positions[:-1]], positions)) if text[start:end + 1].strip())

word_mapping={
    "捷运":"地铁",
    "晶片": "芯片",
    "讯息": "消息",
    "网路": "网络",
    "资安": "网安",
    "行动装置": "移动设备",
    "行动电话": "手机",
    "阵列": "数组",
    "萤幕": "屏幕",
    "程式": "程序",
    "资料库": "数据库",
    "伺服器": "服务器",
    "硬碟": "硬盘",
    "光碟": "光盘",
    "软体": "软件",
    "硬体": "硬件",
    "记忆体": "内存",
    "滑鼠": "鼠标",
    "人工智慧": "人工智能",
    "巨量资料": "大数据",
    "云端运算": "云计算",
    "穿戴式装置": "可穿戴设备",
    "衍生性金融商品": "金融衍生品",
    "避险": "对冲",
    "本益比": "市盈率",
    "殖利率": "收益率",
    "投资报酬率": "投资回报率",
    "风险控管": "风险管理",
    "信用评等": "信用评级",
    "金融风暴": "金融危机",
    "经济成长": "经济增长",
    "国内生产毛额": "国内生产总值",
    "通膨": "通胀",
    "汇率机制": "汇率制度",
    "利率自由化": "利率市场化",
    "财政短绌": "财政赤字",
    "预算法短绌": "财政赤字",
    "国家发展委员会": "国家发展和改革委员会",
    "马克宏": "马克龙",
    "川普": "特朗普",
    "欧巴马": "奥巴马",
    "普丁": "普京",
    "梅伊": "特雷莎·梅",
    "梅尔茨":"默茨",
    "北韩":"朝鲜",
    "纽西兰":"新西兰",
    "沙乌地阿拉伯":"沙特阿拉伯",
    "实质经济": "实体经济",
    "三大动能": "三驾马车",
    "中共三中全会": "三中全会",
    "美国与中国共产党战略竞争特别委员会": "美中战略竞争特别委员会",
    "中华民国":"台湾当局"
}

WEEKDAY_PATTERN = re.compile(r'(上|这|本|下)(?:个)?周([一二三四五六日天])|(?:上|这|本|下)(?:个)?星期([一二三四五六日天])|周([一二三四五六日天])|星期([一二三四五六日天])')

weekday_mapping = {
    '一': 0,  # 星期一
    '二': 1,  # 星期二
    '三': 2,  # 星期三
    '四': 3,  # 星期四
    '五': 4,  # 星期五
    '六': 5,  # 星期六
    '日': 6,  # 星期日
    '天': 6   # 星期天（星期日的另一种表达）
}

# 时间限定词映射
time_modifier_mapping = {
    '上': -1,  # 上周
    '这': 0,   # 这周/本周
    '本': 0,   # 本周
    '下': 1    # 下周
}

DATE_REFERENCE_PATTERN = re.compile(r'(今日|今天|明日|明天|昨日|昨天|前日|前天|后日|后天|大前天|大后天)')

# 时间指代词映射
date_reference_mapping = {
    '今日': 0,
    '今天': 0,
    '明日': 1,
    '明天': 1,
    '昨日': -1,
    '昨天': -1,
    '前日': -1,
    '前天': -2,
    '后日': 1,
    '后天': 2,
    '大前天': -3,
    '大后天': 3
}

def convert_to_cn_term(text):
    """
    将台湾媒体用语转换为大陆用语
    参数:
        text (str): 需要处理的文本
    返回:
        str: 转换后的文本
    """
    for tw_term, cn_term in word_mapping.items():
        if tw_term in text:
            text = text.replace(tw_term, cn_term)
    return text

def calculate_target_date(reference_date, target_weekday_num, time_modifier=None):
    """
    根据参考日期、目标周几和时间限定词计算目标日期
    参数:
        reference_date: 参考日期
        target_weekday_num: 目标周几的数字（0-6，0为周一）
        time_modifier: 时间限定词（'上'、'这'/'本'、'下'，或None）
    返回:
        datetime: 计算出的目标日期
    """
    current_weekday = reference_date.weekday()
    if time_modifier is None:
        # 没有时间限定词时，始终指向本周的对应日期（即使已经过去）
        days_diff = target_weekday_num - current_weekday
        return reference_date + timedelta(days=days_diff)
    else:
        # 有时间限定词
        week_offset = time_modifier_mapping.get(time_modifier, 0)
        if time_modifier in ['这', '本']:
            # 本周：找到本周内的目标日期
            days_diff = target_weekday_num - current_weekday
            return reference_date + timedelta(days=days_diff)
        else:
            # 上周或下周：先计算到目标周，再找到目标日期
            # 计算到本周目标日期的天数差
            days_to_target_in_current_week = target_weekday_num - current_weekday
            # 加上周偏移
            total_days = days_to_target_in_current_week + (week_offset * 7)
            return reference_date + timedelta(days=total_days)

def convert_to_date(text, reference_date=None):
    """
    将文本中的日期表达式替换为具体日期
    参数:
        text (str): 需要处理的文本
        reference_date (str or datetime, optional): 参考日期，可以是ISO格式字符串(如"2022-11-30")或datetime对象，默认为当前日期
    返回:
        str: 替换后的文本
    """
    # 统一处理参考日期设置
    if reference_date is None:
        reference_date = datetime.now()
    elif isinstance(reference_date, str):
        try:
            reference_date = datetime.fromisoformat(reference_date)
        except Exception:
            reference_date = datetime.now()
    # 处理周几表达式
    weekday_matches = list(WEEKDAY_PATTERN.finditer(text))
    # 从后往前处理，避免位置偏移问题
    for match in reversed(weekday_matches):
        # 提取时间限定词和周几字符
        time_modifier = None
        weekday_char = None
        # 检查不同的捕获组
        if match.group(1) and match.group(2):  # (上|这|本|下)周([一二三四五六日天])
            time_modifier = match.group(1)
            weekday_char = match.group(2)
        elif match.group(1) and match.group(3):  # (上|这|本|下)星期([一二三四五六日天])
            time_modifier = match.group(1)
            weekday_char = match.group(3)
        elif match.group(4):  # 周([一二三四五六日天])
            weekday_char = match.group(4)
        elif match.group(5):  # 星期([一二三四五六日天])
            weekday_char = match.group(5)
        if weekday_char:
            weekday_num = weekday_mapping.get(weekday_char)
            if weekday_num is not None:
                # 计算目标日期
                target_date = calculate_target_date(reference_date, weekday_num, time_modifier)
                # 格式化日期，如"5月22日"
                date_str = f"{target_date.month}月{target_date.day}日"
                # 直接替换原文本
                original_text = match.group(0)
                text = text.replace(original_text, date_str, 1)
    # 处理日期指代词
    date_ref_matches = list(DATE_REFERENCE_PATTERN.finditer(text))
    # 从后往前处理，避免位置偏移问题
    for match in reversed(date_ref_matches):
        date_ref_word = match.group(1)
        if date_ref_word in date_reference_mapping:
            days_offset = date_reference_mapping[date_ref_word]
            target_date = reference_date + timedelta(days=days_offset)
            # 格式化日期，如"5月22日"
            date_str = f"{target_date.month}月{target_date.day}日"
            # 直接替换原文本
            original_text = match.group(0)
            text = text[:match.start()] + date_str + text[match.end():]
    return text

def clean_stock_codes(text):
    """
    清理文本中的股票代码和空括号
    参数:
        text (str): 需要处理的文本
    返回:
        str: 清理后的文本
    """
    # 股票代码检测和移除
    stock_code_pattern = r'(?:\d{4,6}\.(?:SH|SZ|HK)|[A-Z]{1,4}\.[A-Z]{1,3})'
    stock_matches = list(re.finditer(stock_code_pattern, text))
    # 反向处理避免索引问题，移除股票代码及其前后的逗号和空格
    for match in reversed(stock_matches):
        start = match.start()
        end = match.end()
        # 向前查找逗号和空格
        while start > 0 and text[start-1] in '，, ':
            start -= 1
        # 向后查找逗号和空格
        while end < len(text) and text[end] in '，, ':
            end += 1
        text = text[:start] + text[end:]
    # 移除空括号和仅包含无意义符号的括号
    meaningless_chars = r'[\s，,、；;：:！!？?。.\-—_/\\|\*\+\=\~\`\^\&\%\$\#\@]'
    empty_brackets_pattern = rf'[\(（]({meaningless_chars}*)[\)）]'
    text = re.sub(empty_brackets_pattern, '', text)
    return text

# 测试 call_local_llm 函数
async def test_call_local_llm():
    """
    测试本地LLM调用是否正常工作
    """
    print("开始测试 call_local_llm 函数...")
    
    system_message = "你是一个有用的助手。"
    user_message = "请回答：1+1等于几？"
    
    try:
        result = await call_local_llm(system_message, user_message, "phi4:latest", 0.1, 0.5, 0, 0)
        if result:
            print(f"✅ LLM调用成功！返回内容长度: {len(result)} 字符")
            print(f"返回内容预览: {result[:100]}...")
        else:
            print("❌ LLM调用失败：返回None")
    except Exception as e:
        print(f"❌ LLM调用出错: {e}")

# ==================== 字符长度计算系统 ====================

class LengthMode(Enum):
    """长度计算模式枚举"""
    CHARACTER = "character"  # 字符模式（610标准）
    WORD = "word"           # 字数模式（574标准，不含标点）
    SEMANTIC_WITH_PUNCT = "semantic_with_punct"  # 字符模式（包含标点符号）

def calculate_characters_with_punctuation(content: str) -> Dict[str, int]:
    """
    计算字符（包含标点符号）的详细统计
    
    Args:
        content (str): 待计算的内容
        
    Returns:
        Dict[str, int]: 各类字符的统计结果
    """
    # 1. 中文字符（包括中文标点符号）
    chinese_chars = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', content)
    chinese_count = len(chinese_chars)
    
    # 2. 英文单词（连续的英文字母）
    english_words = re.findall(r'[a-zA-Z]+', content)
    english_count = len(english_words)
    
    # 3. 数字组（连续的数字）
    number_groups = re.findall(r'\d+', content)
    number_count = len(number_groups)
    
    # 4. 英文标点符号和其他符号
    # 排除空白字符，排除已经统计的中文字符、英文字母、数字
    remaining_content = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', content)  # 移除中文字符
    remaining_content = re.sub(r'[a-zA-Z]', '', remaining_content)  # 移除英文字母
    remaining_content = re.sub(r'\d', '', remaining_content)  # 移除数字
    remaining_content = re.sub(r'\s', '', remaining_content)  # 移除空白字符
    
    english_punct_count = len(remaining_content)
    
    # 总字符数
    total_semantic_units = chinese_count + english_count + number_count + english_punct_count
    
    return {
        'chinese_chars': chinese_count,
        'english_words': english_count,
        'number_groups': number_count,
        'english_punctuation': english_punct_count,
        'total_semantic_units': total_semantic_units
    }

def calculate_length_by_mode(content: str, mode: LengthMode) -> int:
    """
    根据指定模式计算内容长度
    
    Args:
        content (str): 待计算的内容
        mode (LengthMode): 计算模式
        
    Returns:
        int: 计算得到的长度
    """
    if mode == LengthMode.CHARACTER:
        # 字符模式：移除空白字符后的字符数
        return len(re.sub(r'\s', '', content))
    elif mode == LengthMode.WORD:
        # 字数模式：移除标点符号后的字符数（更接近574字标准）
        content_no_punct = re.sub(r'[，。！？；：""（）【】《》、·—…％%]', '', content)
        return len(re.sub(r'\s', '', content_no_punct))
    elif mode == LengthMode.SEMANTIC_WITH_PUNCT:
        # 字符模式（包含标点符号）
        semantic_stats = calculate_characters_with_punctuation(content)
        return semantic_stats['total_semantic_units']
    else:
        raise ValueError(f"不支持的长度计算模式: {mode}")

def validate_content_length_characters(
    content: str,
    target_length: int = 475,
    mode: LengthMode = LengthMode.SEMANTIC_WITH_PUNCT,
    tolerance: int = 50,
    min_threshold: int = 75,
    max_threshold: int = 75
) -> Tuple[Optional[str], int, Dict]:
    """
    字符模式的内容长度校验函数（推荐使用）
    
    Args:
        content (str): 待校验的内容
        target_length (int): 目标长度
        mode (LengthMode): 长度计算模式
        tolerance (int): 理想区间的容忍度
        min_threshold (int): 最小长度阈值（相对于目标长度的偏差）
        max_threshold (int): 最大长度阈值（相对于目标长度的偏差）
        
    Returns:
        Tuple[Optional[str], int, Dict]: (校验结果消息, 实际长度, 详细信息)
    """
    actual_length = calculate_length_by_mode(content, mode)
    
    # 计算所有模式的长度用于对比
    char_length = calculate_length_by_mode(content, LengthMode.CHARACTER)
    word_length = calculate_length_by_mode(content, LengthMode.WORD)
    semantic_length = calculate_length_by_mode(content, LengthMode.SEMANTIC_WITH_PUNCT)
    
    # 获取详细的字符统计
    semantic_stats = calculate_characters_with_punctuation(content)
    
    detail_info = {
        'actual_length': actual_length,
        'character_length': char_length,
        'word_length': word_length,
        'semantic_length': semantic_length,
        'semantic_breakdown': semantic_stats,
        'target_length': target_length,
        'mode': mode.value,
        'difference': actual_length - target_length
    }
    
    # 定义长度区间
    min_len = target_length - min_threshold
    max_len = target_length + max_threshold
    optimal_min = target_length - tolerance
    optimal_max = target_length + tolerance
    
    # 根据模式确定单位描述
    if mode == LengthMode.CHARACTER:
        unit_desc = "字符"
    elif mode == LengthMode.WORD:
        unit_desc = "字"
    else:  # SEMANTIC_WITH_PUNCT
        unit_desc = "字符"
    
    # 1. 强制修正区 (Hard Fail)
    if not (min_len <= actual_length <= max_len):
        if actual_length > max_len:
            diff = actual_length - max_len
            instruction = f"""
### 内容长度修改指令 (最高优先级 - 必须修正) ###
- **问题**: 简报内容**严重超长**。
- **当前长度**: {actual_length} {unit_desc}（{mode.value}模式）。
- **目标长度**: {target_length} {unit_desc}。
- **修改要求**: 必须缩减至少 {diff} 个{unit_desc}。
- **缩减策略**: 删除具体案例细节、修饰性词语和次要背景信息。
"""
        else:  # actual_length < min_len
            diff = min_len - actual_length
            instruction = f"""
### 内容长度修改指令 (最高优先级 - 必须修正) ###
- **问题**: 简报内容**严重过短**。
- **当前长度**: {actual_length} {unit_desc}（{mode.value}模式）。
- **目标长度**: {target_length} {unit_desc}。
- **修改要求**: 必须扩充至少 {diff} 个{unit_desc}。
- **扩充策略**: 补充关键细节和背景信息。
"""
        return instruction, actual_length, detail_info
    
    # 2. 建议优化区 (Soft Fail)
    elif not (optimal_min <= actual_length <= optimal_max):
        diff = target_length - actual_length
        if diff > 0:  # 内容偏短
            instruction = f"""
### 内容长度优化建议 (中等优先级 - 建议优化) ###
- **状态**: 简报长度可接受，但**略短**。
- **当前长度**: {actual_length} {unit_desc}（{mode.value}模式）。
- **目标长度**: {target_length} {unit_desc}。
- **优化建议**: 建议扩充约 {diff} 个{unit_desc}。
"""
        else:  # 内容偏长
            instruction = f"""
### 内容长度优化建议 (中等优先级 - 建议优化) ###
- **状态**: 简报长度可接受，但**略长**。
- **当前长度**: {actual_length} {unit_desc}（{mode.value}模式）。
- **目标长度**: {target_length} {unit_desc}。
- **优化建议**: 建议缩减约 {-diff} 个{unit_desc}。
"""
        return instruction, actual_length, detail_info
    
    # 3. 理想区 (Success)
    else:
        return None, actual_length, detail_info

def validate_574_character_standard(content: str) -> Tuple[Optional[str], int, Dict]:
    """
    574字符标准的专用校验函数（推荐使用，作为默认标准）
    
    Args:
        content (str): 待校验的内容
        
    Returns:
        Tuple[Optional[str], int, Dict]: (校验结果消息, 实际长度, 详细信息)
    """
    return validate_content_length_characters(
        content=content,
        target_length=475,
        mode=LengthMode.SEMANTIC_WITH_PUNCT,
        tolerance=50,     # 理想区间: 425-525
        min_threshold=75, # 强制区间: 400-550
        max_threshold=100
    )

# ==================== 向后兼容的原始函数 ====================
        
def validate_and_instruct_content_length(
    content: str,
    min_len: int = 400,
    max_len: int = 550,
    optimal_target: int = 475,
    tolerance: int = 50
) -> Optional[str]:
    """
    验证简报内容的长度，并在不合规时生成具体的、分级的修改指令。
    该函数统一使用574字符标准进行长度校验。
    
    该函数实现了三级梯度校验：
    1.  **强制修正区 (Hard Fail)**: 长度在 [min_len, max_len] 范围之外，必须修正。
    2.  **建议优化区 (Soft Fail)**: 长度在 [min_len, max_len] 范围之内，但未达到以 optimal_target 为中心的理想区间，建议优化。
    3.  **理想区 (Success)**: 长度在以 optimal_target 为中心的理想区间内，无需修改。

    Args:
        content (str): 待验证的简报内容。
        min_len (int): 最小长度硬性下限（默认400）。
        max_len (int): 最大长度硬性上限（默认550）。
        optimal_target (int): 理想目标长度（默认475字符）。
        tolerance (int): 围绕理想目标的容忍度，定义了理想区间 (optimal_target ± tolerance)。

    Returns:
        Optional[str]: 如果需要修改或优化，返回指令字符串；如果长度在理想区，返回None。
    """
    # 统一使用574字符标准
    result, actual_length, detail_info = validate_574_character_standard(content)
    return result

if __name__ == "__main__":
    # asyncio.run(generate_briefs())
    # asyncio.run(label_articles("安邦历史文章测试"))
    asyncio.run(test_call_local_llm())