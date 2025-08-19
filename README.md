# 简报生成器项目说明文档

## 一、代码结构分析

### 1.1 项目目录结构
```
├── __pycache__/          # Python缓存文件
├── inputs/              # 输入文件目录（txt格式）
├── outputs/             # 输出文件目录
├── prompt_panlin/       # 提示词模板目录
├── prompts/             # 提示词配置目录
├── venv/                # Python虚拟环境
├── workflow.py          # 核心工作流文件（21.8KB）
├── utils.py             # 工具函数文件（16.0KB）
├── embedding.py         # 向量化处理文件（17.1KB）
├── requirements.txt     # 依赖包列表
└── .env                 # 环境变量配置文件
```

### 1.2 核心文件说明
1. **workflow.py** - 主工作流文件
   - 包含文章解读、内容生成、校对和标题生成等核心流程
   - 使用async/await实现异步调用
2. **utils.py** - 工具模块
   - 包含LLM调用、文本处理、日期转换等通用工具函数
3. **embedding.py** - 向量化处理模块
   - 实现文本向量化、标签处理等功能
   - 使用OpenAI和FAISS进行向量计算

## 二、环境配置指南

### 2.1 环境要求
- Python 3.10+（推荐3.11-3.12）
- 操作系统：Windows 10+ / macOS 12+ / Linux（Ubuntu 20.04+）
- 硬件要求：8GB RAM，推荐GPU支持FAISS向量计算

### 2.2 依赖安装
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装FAISS（可选，根据硬件选择）
pip install faiss-cpu  # CPU版本（默认）
pip install faiss-gpu  # GPU版本（需CUDA支持）
```

### 2.3 环境变量配置
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑实际配置
notepad .env  # Windows
# 或使用文本编辑器修改
```

## 三、迭代机制说明

### 2.1 标准重试机制
- **3次重试策略**：
  - 适用于大多数API调用场景（如大模型调用）
  - 典型代码示例：
```python
for attempt in range(3):
    try:
        # 调用大语言模型
        llm_result = await call_llm(...)
    except Exception:
        pass
```

### 2.2 扩展重试机制
- **5次重试策略**：
  - 用于内容生成等关键任务
  - 典型代码示例：
```python
for attempt in range(5):
    try:
        # 生成简报内容
        llm_result = await call_llm(...)
    except Exception:
        pass
```

## 四、开发规范

### 3.1 注释要求
1. 函数级块注释
2. 重要代码块总结注释
3. 禁止行尾注释
4. 不改变原有代码内容

### 3.2 示例注释格式
```python
async def interpret_source_text(source_text: str) -> str:
    """
    解读文章内容并提取关键信息
    
    Args:
        source_text (str): 需要解读的原始文章内容
    
    Returns:
        str: 解读结果的JSON字符串
        str: 提炼后的关键要点字符串（换行符分隔）
        str: 处理后的文章发布日期（年份强制替换为2025）
    """
    # 构建系统提示词和用户输入内容
    system_message = get_prompt('a_interpret_source_text')
    user_message = f"<source_text>\n{source_text}\n</source_text>"
```

### 4.1 代码注释规范
1. 函数级块注释
2. 重要代码块总结注释
3. 禁止行尾注释
4. 不改变原有代码内容

### 4.2 开发工具推荐
- 测试：pytest, pytest-asyncio
- 代码质量：black, flake8, mypy
- 依赖管理：pip-tools（用于依赖升级）

## 五、快速启动
```bash
# 安装依赖（首次运行）
pip install -r requirements.txt

# 设置环境变量
set DASHSCOPE_API_KEY=your_api_key  # Windows
export DASHSCOPE_API_KEY=your_api_key  # Linux/macOS

# 运行主程序
python workflow.py
```