#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新构建索引的脚本
"""

import pandas as pd
import numpy as np
import faiss
import pickle
import ast
from embedding import vectorize


def rebuild_index(csv_file="reference_text.csv", output_file="all_tag_vecs.txt"):
    """重新构建FAISS索引"""
    print("开始重新构建索引...")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    print(f"读取到 {len(df)} 行数据")
    
    # 收集所有标签
    all_tags = set()
    for _, row in df.iterrows():
        tags_list = ast.literal_eval(row["article_tags"])
        if isinstance(tags_list, list):
            all_tags.update(tags_list)
        elif isinstance(tags_list, dict):
            all_tags.update(tags_list.keys())
    
    tags_list = list(all_tags)
    print(f"总共找到 {len(tags_list)} 个唯一标签")
    
    # 向量化所有标签
    print("开始向量化标签...")
    all_vectors = []
    
    # 分批处理标签
    batch_size = 10
    for i in range(0, len(tags_list), batch_size):
        batch = tags_list[i:i+batch_size]
        try:
            embeddings = vectorize(batch)
            for emb in embeddings:
                all_vectors.append(emb["embedding"])
            print(f"已处理 {min(i+batch_size, len(tags_list))}/{len(tags_list)} 个标签")
        except Exception as e:
            print(f"处理批次 {i//batch_size+1} 时出错: {e}")
            # 出错时使用零向量填充
            for _ in batch:
                all_vectors.append([0.0] * 1024)
    
    # 转换为numpy数组
    vectors = np.array(all_vectors, dtype='float32')
    print(f"向量形状: {vectors.shape}")
    
    # 归一化向量
    faiss.normalize_L2(vectors)
    
    # 构建FAISS索引
    print("构建FAISS索引...")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
    index.add(vectors)
    
    # 保存索引
    ann_data = {"index": index, "tags": tags_list}
    with open(output_file, 'wb') as f:
        pickle.dump(ann_data, f)
    
    print(f"索引构建完成，包含 {len(tags_list)} 个标签，维度: {dimension}")
    return ann_data


if __name__ == "__main__":
    rebuild_index()