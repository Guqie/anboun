# workflow.py

import asyncio
import os
import re
import csv
import json
import ast
from utils import convert_to_cn_term, convert_to_date, clean_stock_codes, get_prompt, call_llm, extract_with_xml, split_to_sentences, save_as_txt, replace_year_with_2025, remove_year_at_start
from embedding import get_similar_tags
from prompts.k_adjust_length import validate_and_instruct_title_length
from utils import validate_and_instruct_content_length
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
    system_message = get_prompt('a_interpret_source_text')
    user_message = f"<source_text>\n{source_text}\n</source_text>"
    for attempt in range(3):
        try:
            llm_result = await call_llm(system_message, user_message, "openai/gpt-5-mini", 0.1, 0.5, 0, 0)
            if interpretation := json.loads(extract_with_xml(llm_result, "interpretation")):
                key_points = "\n".join(interpretation["关键要点提炼"])
                published_date = replace_year_with_2025(interpretation.pop("新闻文章发布日期"))
                return json.dumps(interpretation, ensure_ascii=False, indent=4), key_points, published_date
        except Exception:
            pass
    return None, None, None

async def draft_brief_content(source_text: str, interpretation: str, article_contents: str) -> str:
    """
    生成文章的事实段落
    Args:
        source_text (str): 文章内容
        interpretation (str): 对文章内容的解读
    Returns:
        str: 事实段落
    """
    system_message = get_prompt('b_draft_brief_content').format(article_contents=article_contents)
    user_message = f"<source_text>\n{source_text}\n</source_text>\n<interpretation>\n{interpretation}\n</interpretation>"
    
    for attempt in range(5):
        try:
            llm_result = await call_llm(system_message, user_message, "deepseek/deepseek-chat-v3-0324", 0.6, 0.95, 0, 0)
            
            # 增加空值检测
            if not llm_result:
                print(f"LLM返回空结果（第{attempt+1}次）")
                continue  # 重试
                
            if not (brief_content := extract_with_xml(llm_result, "brief_content")):
                print(f"未找到brief_content标签（第{attempt+1}次）")
                continue  # 重试
                
            return brief_content.replace('\n', '')
            
        except Exception as e:
            print(f"段落生成失败（第{attempt+1}次）: {str(e)}")
            continue
            
    # 所有重试失败后返回空字符串
    print("警告: 使用空段落代替LLM失败结果")
    return ""
async def review_brief_content(source_text: str, brief_content: str) -> str | None:
    """
    校对文章的事实段落
    Args:
        source_text (str): 文章内容
        brief_content (str): 事实段落
    Returns:
        str | None: 如果需要修正则返回对事实段落的反馈，否则返回None
    """
    # 1. 执行内容长度校验
    length_feedback = validate_and_instruct_content_length(brief_content)

    # 2. 执行LLM语义校验
    system_message = get_prompt('c_review_brief_content')
    user_message = f"<source_text>\n{source_text}\n</source_text>\n<brief_content>\n{brief_content}\n</brief_content>"
    for attempt in range(3):
        try:
            llm_result = await call_llm(system_message, user_message, "qwen/qwen3-30b-a3b-instruct-2507", 0.3, 0.95, 0, 0)
            feedback_on_brief_content, corrections_required = extract_with_xml(llm_result, ["feedback_on_brief_content", "corrections_required"])
            if feedback_on_brief_content and corrections_required:
                # 3. 合并反馈信息
                final_feedback = ""
                needs_refinement = False

                if length_feedback:
                    final_feedback += length_feedback
                    needs_refinement = True
                
                if corrections_required.strip().lower() == "true":
                    if final_feedback:
                        final_feedback += "\n\n--- 语义与事实核查反馈 ---\n"
                    final_feedback += feedback_on_brief_content
                    needs_refinement = True

                return final_feedback if needs_refinement else None
            # 如果提取失败，记录错误并重试
            print(f"提取失败，重试次数: {attempt + 1}")
        except Exception as e:
            print(f"调用LLM异常: {str(e)}")
            # 可选：增加指数退避
            import asyncio
            await asyncio.sleep(2 ** attempt)
    return None
async def review_brief_sentences(source_text: str, brief_content: str) -> str | None:
    """
    校对文章的事实句子
    Args:
        source_text (str): 文章内容
        brief_content (str): 事实段落
    Returns:
        str | None: 如果需要修正则返回对事实句子的反馈，否则返回None
    """
    system_message = get_prompt('d_review_brief_sentences')
    user_message = f"<source_text>\n{source_text}\n</source_text>\n<brief_sentences>\n{split_to_sentences(brief_content)}\n</brief_sentences>"
    for attempt in range(3):
        try:
            llm_result = await call_llm(system_message, user_message, "qwen/qwen3-30b-a3b-instruct-2507", 0.3, 0.95, 0, 0)
            feedback_on_brief_sentences, corrections_required = extract_with_xml(llm_result, ["feedback_on_brief_sentences", "corrections_required"])
            if feedback_on_brief_sentences and corrections_required:
                if "True" in corrections_required or "true" in corrections_required:
                    return feedback_on_brief_sentences
                else:
                    return None
        except Exception:
            pass
    return None

async def refine_brief_content(brief_content: str, feedback_on_brief_content: str | None, feedback_on_brief_sentences: str | None) -> str:
    """
    修改文章的事实段落
    Args:
        brief_content (str): 事实段落
        feedback_on_brief_content (str): 对事实段落的反馈
        feedback_on_brief_sentences (str): 对事实句子的反馈
    Returns:                    
        str: 修改后的段落，如果无需修改则返回None
    """
    if not feedback_on_brief_content and not feedback_on_brief_sentences:
        return None
    system_message = get_prompt('e_refine_brief_content')
    user_message = f"<brief_content>\n{brief_content}\n</brief_content>" + (f"\n<feedback_on_brief_content>\n{feedback_on_brief_content}\n</feedback_on_brief_content>" if feedback_on_brief_content else "") + (f"\n<feedback_on_brief_sentences>\n{feedback_on_brief_sentences}\n</feedback_on_brief_sentences>" if feedback_on_brief_sentences else "")
    for attempt in range(3):
        try:
            llm_result = await call_llm(system_message, user_message, "google/gemini-2.5-pro-preview", 0.1, 0.8, 0, 0)
            if refined_brief_content := extract_with_xml(llm_result, "refined_brief_content"):
                print(f"Refined brief content: {refined_brief_content}")
                return refined_brief_content
        except Exception:
            pass
    return None

async def draft_and_refine_brief_content(source_text: str, interpretation: str, article_contents: str) -> str:
    """
    通过“撰写-校对-修改”的循环来创建并完善事实段落。
    该函数首先生成一个事实段落草稿，然后反复进行校对和修改，
    直到校对反馈表明无需任何修改为止。
    Args:
        source_text (str): 原始文章内容。
        interpretation (str): 对原始文章的解读。
    Returns:
        str: 最终完善后的事实段落。
    """
    brief_content = await draft_brief_content(source_text, interpretation, article_contents)
    while True:
        feedback_on_brief_content, feedback_on_brief_sentences = await asyncio.gather(review_brief_content(source_text, brief_content), review_brief_sentences(source_text, brief_content))
        refined_brief_content = await refine_brief_content(brief_content, feedback_on_brief_content, feedback_on_brief_sentences)
        if refined_brief_content is None:
            print("事实段落校对完成，无需进一步修改。")
            return brief_content
        else:
            print("事实段落已根据反馈进行修改，将进行新一轮校对...")
            brief_content = refined_brief_content

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
            llm_result = await call_llm(system_message, user_message, "qwen/qwen3-235b-a22b-2507", 0.5, 0.95, 0, 0.5)
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
            llm_result = await call_llm(system_message, user_message, "qwen/qwen3-235b-a22b-2507", 0.1, 0.95, 0, 0)
            feedback_on_brief_title, corrections_required = extract_with_xml(llm_result, ["feedback_on_brief_title", "corrections_required"])
            if feedback_on_brief_title and corrections_required:
                if "True" in corrections_required or "true" in corrections_required:
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
            llm_result = await call_llm(system_message, user_message, "qwen/qwen3-235b-a22b-2507", 0.3, 0.5, 0, 0)
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
            print("标题已根据反馈进行修改，将进行新一轮校对...")
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
        file_name (str): 要匹配的CSV文件名（不包含.csv扩展名），默认为"每日经济每日金融1-10"
    
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
            llm_result = await call_llm(system_message, user_message, "openai/gpt-oss-120b", 0.1, 0.5, 0, 0)
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
        print(f"开始处理文件: {file_name}")
        print(f"[{file_name}] 1/6: 读取并解读原文")
        source_text = get_source_text(file_name)
        interpretation, key_points, published_date = await interpret_source_text(source_text)
        interpretation = convert_to_date(interpretation, published_date)
        source_text = convert_to_cn_term(source_text)
        source_text = clean_stock_codes(source_text)
        source_text = convert_to_date(source_text, published_date)
        print(f"[{file_name}] 2/6: 匹配历史文章")
        articles = await match_articles(key_points, "reference_text")
        article_contents = "\n\n".join([re.sub(r'（[A-Za-z]+）$', '', article['InfoContent']).strip() for article in articles[:3]])
        article_titles = "\n\n".join([(article['InfoTitle'].split('：', 1)[1].strip() if '：' in article['InfoTitle'] else article['InfoTitle']) for article in articles])
        print(f"[{file_name}] 3/6: 创建并完善简报内容")
        brief_content = await draft_and_refine_brief_content(source_text, interpretation, article_contents)
        brief_content = remove_year_at_start(brief_content)
        print(f"[{file_name}] 5/6: 创建并完善简报标题")
        brief_title = await draft_and_refine_brief_title(brief_content, article_titles)
        save_as_txt(f"{brief_title}\n{brief_content}\n\n{"\n\n".join([f"{article['InfoTitle']}\n{article['InfoContent']}\n{article['ProductDate']}" for article in articles])}", f"{file_name} with matched briefs")
        # print(f"[{file_name}] 6/6: 翻译为其它语言")
        # english_title, english_content, german_title, german_content, french_title, french_content, japanese_title, japanese_content = await translate_to_other_languages(brief_title, brief_content)
        # save_as_txt(f"{brief_title}\n{brief_content}\n\n{english_title}\n{english_content}\n\n{german_title}\n{german_content}\n\n{french_title}\n{french_content}\n\n{japanese_title}\n{japanese_content}", f"{file_name} with other language versions")
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