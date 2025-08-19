# utils.py

import aiohttp
import asyncio
import os
import re
import json
from datetime import datetime, timedelta
import random
from typing import Optional
import importlib

OPENROUTER_API_KEY = "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"

async def call_llm(system_message: str, user_message: str, model="openai/gpt-oss-120b", temperature=0.5, top_p=0.95, frequency_penalty=0, presence_penalty=0) -> Optional[str]:
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
    try:
        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                async with session.post(url, headers=headers, json=data, ssl=False, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    response_data = await response.json()
                    if content := response_data["choices"][0]["message"]["content"]:
                        print(content)
                        return content
    except Exception:
        pass
    return None

async def call_local_llm(system_message: str, user_message: str, model="qwen3:32b", temperature=0.5, top_p=0.95, frequency_penalty=0, presence_penalty=0) -> Optional[str]:
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
            async with session.post(url, headers={}, json=data, timeout=aiohttp.ClientTimeout(total=150)) as response:
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
    prompt = getattr(importlib.import_module(f"prompts.{name}"), name)
    if arguments:
        if callable(prompt):
            return prompt(**arguments)
        else:
            return prompt.format(**arguments)
    else:
        if callable(prompt):
            return prompt()
        else:
            return prompt

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
        
def validate_and_instruct_content_length(content: str, min_len: int = 300, max_len: int = 500) -> Optional[str]:
    """
    验证简报内容的长度，并在不合规时生成具体的修改指令。

    Args:
        content (str): 待验证的简报内容。
        min_len (int): 最小长度。
        max_len (int): 最大长度。

    Returns:
        Optional[str]: 如果长度不合规，返回修改指令字符串；如果合规，返回None。
    """
    actual_length = len(content)

    if not (min_len <= actual_length <= max_len):
        if actual_length > max_len:
            # 内容过长
            diff = actual_length - max_len
            instruction = f"""
### 内容长度修改指令 (最高优先级) ###
- **问题**: 简报内容过长。
- **当前精确长度**: {actual_length} 字符。
- **目标范围**: {min_len}-{max_len} 字符。
- **修改要求**: 你必须 (MUST) 将内容**缩减至少 {diff} 个字符**。
- **缩减策略**:
  1.  **优先删减**: 首先删除具体的案例细节、修饰性词语和次要的背景信息。
  2.  **合并句子**: 将逻辑相关的短句合并，使用更凝练的表达。
  3.  **保留核心**: 确保在缩减后，最重要的事实、数据和最终的分析结论得以保留。
"""
        else: # 内容过短
            diff = min_len - actual_length
            instruction = f"""
### 内容长度修改指令 (最高优先级) ###
- **问题**: 简报内容过短。
- **当前精确长度**: {actual_length} 字符。
- **目标范围**: {min_len}-{max_len} 字符。
- **修改要求**: 你必须 (MUST) 为内容**扩充至少 {diff} 个字符**。
- **扩充策略**:
  1.  **补充细节**: 从`<source_text>`中寻找被省略的关键细节或背景信息进行补充。
  2.  **展开论述**: 适度展开最终的分析结论，使其更具说服力。
  3.  **避免空话**: 补充的内容必须是言之有物的事实或观点，禁止添加无意义的填充性文字。
"""
        return instruction
    else:
        # 长度合规
        return None

if __name__ == "__main__":
    # asyncio.run(generate_briefs())
    # asyncio.run(label_articles("安邦历史文章测试"))
    asyncio.run(test_call_local_llm())