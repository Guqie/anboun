import regex
from typing import Optional

def validate_and_instruct_title_length(title: str, max_length: int = 20) -> Optional[str]:
    """
    验证标题长度，并在超长时生成具体的修改指令。

    Args:
        title (str): 待验证的标题。
        max_length (int): 最大长度限制。

    Returns:
        Optional[str]: 如果超长，返回修改指令字符串；如果长度合规，返回None。
    """
    # 精确计算视觉长度
    length = round(len(regex.findall(r'\p{Han}', title)) + len(regex.findall(r'[a-zA-Z0-9]', title)) / 2)

    if length > max_length:
        # 生成高质量的修复指令
        instruction = f"""
### 长度修改指令 (最高优先级) ###
- **问题**: 标题过长。
- **当前长度**: {length} 字符 (已精确计算)。
- **长度限制**: {max_length} 字符。
- **修改要求**: 你必须 (MUST) 将标题缩减到{max_length}字符以内。
- **修改策略 (请参考)**:
  1.  **删减次要信息**: 优先移除修饰性词语或次要细节。
  2.  **保留核心**: 确保缩减后仍能体现简报最核心的事实或观点。
  3.  **同义词替换**: 使用更短的词语替换，例如“世界卫生组织”替换为“世卫组织”。
- **示例**:
  - **过长**: “中国稳就业新政预计可缓解结构性失业但面临落地挑战” (26字符)
  - **修改后**: “中国稳就业新政面临落地挑战” (13字符)
"""
        return instruction
    else:
        # 长度合规，无需操作，返回None
        return None
def k_adjust_length(text: str):
    length = round(len(regex.findall(r'\p{Han}', text)) + len(regex.findall(r'[a-zA-Z0-9]', text)) / 2)
    if length > 20:
        return f"""简报标题长度不能超过20个字符，目前的标题长度为{length}个字符，多出了{length - 20}个字符。
请尝试删减细节或者相对非重点的信息，以及替换其中一些词语为更短的同义词，从而将标题长度控制在20个字符以内。
例如，“OpenAI计划发布AI Agent驱动的网络浏览器挑战谷歌Chrome”，简化为“OpenAI计划发布网络浏览器挑战谷歌Chrome”，因为“AI Agent驱动的”属于细节信息。
再如，“中国稳就业新政预计可缓解结构性失业但面临落地挑战”，简化为“中国稳就业新政面临落地挑战”，因为“但”字之前的“预计可缓解结构性失业”属于相对非重点信息。
还可以替换其中一些词语为更短的同义词，确保保持原意，例如“外交部长”简化为“外长”；“外交部副部长”简化为“副外长”；“世界卫生组织”简化为“世卫组织”；“股票市场”简化为“股市”。
注意避免过度简化，仅仅减少多出的字符数量即可。必须体现简报内容中最值得注意的事实或观点，以及包括必要的信息。例如，如果出现人名，保留其所在组织和职业头衔作为前缀。"""
    else:
        return ""


if __name__ == "__main__":
    print(k_adjust_length("二季度信贷投向显示金融资源持续向绿色经济小微企业和科技创新领域倾斜"))
