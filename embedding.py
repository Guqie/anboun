# embedding.py

import os
import time
from datetime import datetime
import json
import numpy as np
from openai import OpenAI
from typing import List, Union, Dict
import pandas as pd
import faiss
import pickle
import ast


DASHSCOPE_API_KEY = "sk-684a3a134fbf49af8818a88260778df3"

# 全局复用 OpenAI 客户端
client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=60)

def vectorize(texts: Union[str, List[str]]) -> List[Dict]:
    """
    向量化输入文本（支持单个字符串或字符串列表）。
    包含重试机制和超时设置。
    
    参数:
    - texts (str or List[str]): 要向量化的文本内容
    
    返回:
    - List[Dict]: 包含向量信息的字典列表
    """
    for attempt in range(3):
        try:
            completion = client.embeddings.create(model="text-embedding-v4", input=texts, dimensions=1024, encoding_format="float")
            return json.loads(completion.model_dump_json())['data']
        except Exception as e:
            if attempt < 2:
                print(f"API调用失败，立即重试... (尝试 {attempt + 1}/3)")
            else:
                print(f"API调用失败，已达到最大重试次数: {e}")
                raise

def merge_tags_to_csv(csv_path: str) -> None:
    """
    处理CSV文件，将多个标签列合并成一个article_tags列
    
    参数:
    - csv_path (str): CSV文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    # 定义需要处理的列
    columns = [
        "political_and_economic_terms",
        "technical_terms", 
        "other_abstract_concepts",
        "organizations",
        "persons",
        "cities_or_districts",
        "other_concrete_entities",
        "other_tags_of_topic_or_points"
    ]
    # 定义需要过滤掉的标签
    filter_tags = {'安邦咨询', '安邦智库', 'ANBOUND', '陈功'}
    # 使用推导式创建article_tags列，并过滤掉指定标签
    df['article_tags'] = [
        [tag for column in columns 
         if isinstance(row[column], str) 
         for tag in ast.literal_eval(row[column]) 
         if tag not in filter_tags] 
        for i, row in df.iterrows()
    ]
    # 将更新后的数据框保存回CSV文件
    df.to_csv(csv_path, index=False)

def vectorize_tags_to_csv(csv_path: str) -> None:
    """
    处理CSV文件，将每一行的article_tags列表向量化并转换为字典格式
    
    参数:
    - csv_path (str): CSV文件路径
    """
    df = pd.read_csv(csv_path)
    tag_vecs_list = []
    
    # 添加进度条
    try:
        from tqdm import tqdm
        progress_func = tqdm
    except ImportError:
        # 如果没有安装 tqdm，则使用简单的进度显示
        def progress_func(iterable, total=None, desc=""):
            current = 0
            for item in iterable:
                current += 1
                if total:
                    print(f"{desc}: {current}/{total} ({current/total*100:.1f}%)")
                yield item
        pass
    
    print(f"开始处理 {len(df)} 行数据...")
    
    # 使用并发处理提高向量化速度
    import concurrent.futures
    from functools import partial
    
    def process_row(row):
        article_tags = ast.literal_eval(row["article_tags"])
        if article_tags:
            embeddings = []
            # 分批处理，每批最多10条
            for start in range(0, len(article_tags), 10):
                batch_tags = article_tags[start:start + 10]
                try:
                    batch_embeddings = vectorize(batch_tags)
                    embeddings.extend([item["embedding"] for item in batch_embeddings])
                    # 减少等待时间以提高速度
                    time.sleep(0.05)
                except Exception as e:
                    print(f"处理批次 {start//10+1} 时出错: {e}")
                    # 出错时使用零向量填充
                    embeddings.extend([[0.0] * 1024 for _ in range(len(batch_tags))])
            
            return {tag: vec for tag, vec in zip(article_tags, embeddings)}
        else:
            return {}
    
    # 使用线程池并发处理行数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 包装进度条显示
        results = list(progress_func(
            executor.map(process_row, [row for _, row in df.iterrows()]), 
            total=len(df), 
            desc="处理行"
        ))
    
    df["article_tags"] = results
    df.to_csv(csv_path, index=False)
    print("标签向量化完成!")

def merge_and_vectorize_tags_to_csv(data_folder: str = "/Users/siyu/Documents/briefgenerator/测试数据") -> None:
    """
    从指定文件夹筛选出修改日期在2025年8月6日及以后的CSV文件，
    先调用merge_tags_to_csv函数，再调用vectorize_tags_to_csv函数
    
    参数:
    - data_folder (str): 数据文件夹路径
    """
    # 获取文件夹中所有CSV文件
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    # 筛选出修改日期在阈值日期及以后的文件
    recent_files = []
    for csv_file in csv_files:
        file_path = os.path.join(data_folder, csv_file)
        # 获取文件修改时间
        modification_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if modification_time >= datetime(2021, 10, 11): # 调整日期阈值以匹配当前文件
            recent_files.append(file_path)
    print(f"找到 {len(recent_files)} 个符合条件的CSV文件（修改日期在2021年10月11日及以后）")
    # 对每个符合条件的文件执行处理
    for i, file_path in enumerate(recent_files, 1):
        print(f"正在处理文件 {i}/{len(recent_files)}: {os.path.basename(file_path)}")
        # 步骤1：合并标签
        print(f"  - 合并标签...")
        merge_tags_to_csv(file_path)
        # 步骤2：向量化标签
        print(f"  - 向量化标签...")
        vectorize_tags_to_csv(file_path)
        print(f"  - 文件处理完成")
    print(f"所有 {len(recent_files)} 个文件处理完成")

def build_ann_index(data_folder: str = ".") -> dict:
    """从CSV文件中提取标签向量，构建FAISS索引并保存到文件"""
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    # 调整为当前项目路径
    recent_files = [os.path.join(data_folder, f) for f in csv_files if f == "reference_text.csv"]
    
    all_tag_vecs = {}
    total_rows = 0
    
    # 计算总行数用于进度显示
    for file_path in recent_files:
        df = pd.read_csv(file_path)
        total_rows += len(df)
    
    print(f"开始处理 {total_rows} 行数据...")
    processed_rows = 0
    
    for file_path in recent_files:
        print(f"处理文件: {file_path}")
        df = pd.read_csv(file_path)
        
        # 添加进度显示
        for idx, row in df.iterrows():
            tag_dict = ast.literal_eval(row['article_tags'])
            # 检查 tag_dict 是否为字典并且不为空
            if isinstance(tag_dict, dict) and tag_dict:
                for tag, vec in tag_dict.items():
                    # 检查向量是否有效
                    if vec is not None and isinstance(vec, list) and len(vec) > 0:
                        all_tag_vecs[tag] = vec
            # 如果 tag_dict 是列表，则需要向量化
            elif isinstance(tag_dict, list) and tag_dict:
                # 这种情况下需要重新向量化标签
                try:
                    embeddings = vectorize(tag_dict)
                    for tag, emb in zip(tag_dict, embeddings):
                        all_tag_vecs[tag] = emb["embedding"]
                except Exception as e:
                    print(f"向量化标签时出错: {e}")
            
            processed_rows += 1
            if processed_rows % 1000 == 0 or processed_rows == total_rows:
                print(f"索引构建进度: {processed_rows}/{total_rows} ({processed_rows/total_rows*100:.1f}%)")
    
    if not all_tag_vecs:
        print("警告: 没有找到标签向量数据")
        return None
        
    tags = list(all_tag_vecs.keys())
    vectors = np.array([all_tag_vecs[tag] for tag in tags], dtype='float32')
    
    # 检查向量维度
    if vectors.shape[1] == 0:
        print("错误: 向量维度为0，无法构建索引")
        return None
        
    print(f"正在构建FAISS索引，包含 {len(tags)} 个标签向量，向量维度: {vectors.shape[1]}...")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    ann_data = {"index": index, "tags": tags}
    
    with open('all_tag_vecs.txt', 'wb') as f:
        pickle.dump(ann_data, f)
    
    print(f"成功构建索引，包含 {len(tags)} 个标签向量，向量维度: {vectors.shape[1]}")
    return ann_data

def visualize_index(index_file='all_tag_vecs.txt'):
    """可视化索引中的标签向量"""
    # 导入可视化所需的库
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import seaborn as sns
    
    # 读取索引文件
    with open(index_file, 'rb') as f:
        ann_data = pickle.load(f)
    
    index = ann_data["index"]
    tags = ann_data["tags"]
    
    # 获取向量数据
    vecs = index.reconstruct_n(0, index.ntotal)
    
    print(f"索引中包含 {len(tags)} 个标签")
    print(f"向量维度: {vecs.shape[1]}")
    
    # 检查向量维度
    if vecs.shape[1] == 0:
        print("错误: 向量维度为0，无法进行可视化")
        return None
    
    # 如果标签数量太多，只显示前1000个
    display_count = min(1000, len(tags))
    display_vecs = vecs[:display_count]
    display_tags = tags[:display_count]
    
    # 使用PCA降维到2D用于可视化
    print("正在进行PCA降维...")
    pca = PCA(n_components=2)
    vecs_2d = pca.fit_transform(display_vecs)
    
    # 创建PCA可视化图表
    plt.figure(figsize=(12, 8))
    plt.scatter(vecs_2d[:, 0], vecs_2d[:, 1], alpha=0.7)
    plt.title(f'标签向量PCA可视化 (前{display_count}个标签)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # 标注一些关键点
    if display_count <= 100:  # 只有在标签较少时才标注
        for i, tag in enumerate(display_tags):
            plt.annotate(tag, (vecs_2d[i, 0], vecs_2d[i, 1]), 
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('tags_pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 如果标签数量适中，也尝试t-SNE可视化
    if display_count <= 500:
        print("正在进行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, display_count-1))
        vecs_tsne = tsne.fit_transform(display_vecs)
        
        # 创建t-SNE可视化图表
        plt.figure(figsize=(12, 8))
        plt.scatter(vecs_tsne[:, 0], vecs_tsne[:, 1], alpha=0.7)
        plt.title(f'标签向量t-SNE可视化 (前{display_count}个标签)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        if display_count <= 100:
            for i, tag in enumerate(display_tags):
                plt.annotate(tag, (vecs_tsne[i, 0], vecs_tsne[i, 1]), 
                            fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('tags_tsne_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 显示向量统计信息
    print(f"\n向量统计信息:")
    print(f"  总标签数: {len(tags)}")
    print(f"  向量维度: {vecs.shape[1]}")
    print(f"  向量模长 (应接近1): {np.linalg.norm(vecs[0]):.4f}")
    print(f"  最小值: {vecs.min():.4f}")
    print(f"  最大值: {vecs.max():.4f}")
    print(f"  平均值: {vecs.mean():.4f}")
    
    return vecs_2d if display_count <= 500 else None

def analyze_existing_index(index_file='all_tag_vecs.txt'):
    """分析已有的 all_tag_vecs.txt 文件中的索引信息"""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import seaborn as sns
    
    # 读取索引文件
    with open(index_file, 'rb') as f:
        ann_data = pickle.load(f)
    
    index = ann_data["index"]
    tags = ann_data["tags"]
    
    # 获取向量数据
    vecs = index.reconstruct_n(0, index.ntotal)
    
    print(f"索引分析报告:")
    print(f"  总标签数: {len(tags)}")
    print(f"  向量维度: {vecs.shape[1]}")
    print(f"  索引类型: {type(index)}")
    
    # 显示前20个标签
    print(f"\n前20个标签:")
    for i, tag in enumerate(tags[:20]):
        print(f"  {i+1:2d}. {tag}")
    
    # 向量统计信息
    print(f"\n向量统计信息:")
    print(f"  向量模长 (应接近1): {np.linalg.norm(vecs[0]):.4f}")
    print(f"  最小值: {vecs.min():.4f}")
    print(f"  最大值: {vecs.max():.4f}")
    print(f"  平均值: {vecs.mean():.4f}")
    print(f"  标准差: {vecs.std():.4f}")
    
    return ann_data

def remove_filter_tags_from_csv(data_folder: str = "/Users/siyu/Documents/briefgenerator/测试数据") -> None:
    """
    批量处理CSV文件，从article_tags字典中去除指定的键值对
    仅处理修改日期为2025年8月5日的文件
    """
    filter_tags = {'安邦咨询', '安邦智库', 'ANBOUND', '陈功'}
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    target_files = [
        os.path.join(data_folder, csv_file) 
        for csv_file in csv_files 
        if datetime.fromtimestamp(os.path.getmtime(os.path.join(data_folder, csv_file))).date() == datetime(2025, 8, 5).date()
    ]
    
    for file_path in target_files:
        df = pd.read_csv(file_path)
        df['article_tags'] = [
            {k: v for k, v in ast.literal_eval(row['article_tags']).items() if k not in filter_tags}
            for _, row in df.iterrows()
        ]
        df.to_csv(file_path, index=False)

def get_similar_tags(query: str, top_n: int = 100) -> set:
    """对查询文本进行向量化，并从预先构建的索引中查找最相似的标签，返回标签集合"""
    query_vec = vectorize([query])[0]
    with open('all_tag_vecs.txt', 'rb') as f:
        ann_data = pickle.load(f)
    
    # 获取索引的维度
    index = ann_data["index"]
    tags = ann_data["tags"]
    
    # 通过重构第一个向量来获取实际维度
    try:
        sample_vec = index.reconstruct(0)
        index_dimension = len(sample_vec)
    except:
        # 如果无法重构，则使用默认维度1024
        index_dimension = 1024
    
    print(f"索引维度: {index_dimension}")
    
    # 确保查询向量维度与索引维度匹配
    query_embedding = query_vec['embedding']
    if len(query_embedding) != index_dimension:
        print(f"警告: 查询向量维度({len(query_embedding)})与索引维度({index_dimension})不匹配")
        # 如果维度不匹配，截断或填充向量
        if len(query_embedding) > index_dimension:
            query_embedding = query_embedding[:index_dimension]
        else:
            query_embedding = query_embedding + [0.0] * (index_dimension - len(query_embedding))
    
    query_array = np.asarray(query_embedding, dtype='float32').reshape(1, -1)
    faiss.normalize_L2(query_array)
    distances, indices = index.search(query_array, top_n)
    similar_tags = [tags[i] for i in indices[0]]
    print(similar_tags)
    return set(similar_tags)

if __name__ == "__main__":
    # 分析已有的索引
    try:
        ann_data = analyze_existing_index()
    except Exception as e:
        print(f"分析索引时出错: {e}")
        print("可能需要重新生成索引文件")
    
    # 运行主函数进行标签相似度查询
    # query = """
# 近日，比亚迪股份在港交所公告称，公司成功完成了一项大规模的港股配售。这也是近四年来香港股市规模最大的一次融资活动。比亚迪公告称，假设配售股份全数配售，配售所得款项总额预计约为435.09亿港元（约合人民币407.48亿元），于扣除佣金和估计费用后，配售所得款项净额预计约为433.83亿港元（约合人民币406.30亿元）。而最新数据显示，截至2024年三季度，比亚迪的所有者权益总额为1688亿元，足可以看出本次融资的规模之大。比亚迪计划将新筹集的资金用于多个关键领域，包括扩大海外业务、投资研发、补充营运资金以及一般企业用途。花旗集团分析师Jeff Chung表示：“我们视此次股权融资为积极之举。”彭博情报分析师Joanna Chen也指出，此次配售将有助于比亚迪加速海外工厂的建设，这在关税风险不断增加的背景下显得更加迫切。目前比亚迪依然在扩大本地化生产来对抗关税，其位于匈牙利的工厂计划在今年晚些时候投产，另一家在土耳其的工厂正在筹建中，同时还在考虑设立第三家欧洲工厂。值得注意的是，安邦智库（ANBOUND）的研究人员曾提到，目前以比亚迪为代表的新能源车企负债率整体处于较高水平，这也凸显了本次股权融资的重要性。市场对本次配股的反应偏中性，配售定价较3月3日的收盘价折价7.8%，而3月4日港股收盘比亚迪股份下跌约6.8%。（PMS)
# """
    # 运行主函数进行标签相似度查询
    # get_similar_tags(query)
