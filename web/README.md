# 多模态情感分析系统 - Web界面

基于Streamlit的多模态情感分析Web应用。

## 功能特性

### 输入模态
- **文本输入**: 直接输入中文或英文文本进行情感分析
- **音频上传**: 支持WAV、MP3、M4A格式音频文件
- **视频上传**: 支持MP4、AVI、MOV格式视频文件

### 结果展示
- 预测情感（中文标签 + Emoji）
- 置信度百分比和进度条
- 情感分布柱状图（可视化）
- 详细概率表格
- 使用的模态列表

### 技术特点
- ✅ 使用 `@st.cache_resource` 缓存模型，避免重复加载
- ✅ 临时文件自动清理，节省磁盘空间
- ✅ 完善的异常处理和用户提示
- ✅ 响应式布局，支持宽屏模式
- ✅ 中文界面，使用UTF-8编码
- ✅ 支持灵活的模态组合输入

## 安装依赖

```bash
# 使用uv（推荐）
uv pip install -r requirements.txt

# 或使用pip
pip install -r requirements.txt
```

主要依赖：
- streamlit >= 1.28.0
- torch >= 2.0.0
- transformers >= 4.30.0
- 其他依赖见 requirements.txt

## 运行方式

### 方法1: 使用streamlit命令（推荐）

```bash
# 在项目根目录执行
streamlit run web/app.py
```

应用将在浏览器中自动打开，默认地址为：
```
http://localhost:8501
```

### 方法2: 指定端口和主机

```bash
# 自定义端口
streamlit run web/app.py --server.port 8080

# 允许外部访问
streamlit run web/app.py --server.address 0.0.0.0

# 组合使用
streamlit run web/app.py --server.port 8080 --server.address 0.0.0.0
```

### 方法3: 使用Python模块

```bash
python -m streamlit run web/app.py
```

## 使用说明

### 1. 基本使用流程

1. 打开Web界面
2. 在侧边栏查看使用说明
3. 至少提供一种输入：
   - 文本：直接在文本框输入
   - 音频：点击"Browse files"上传
   - 视频：点击"Browse files"上传
4. 点击"开始分析"按钮
5. 查看分析结果

### 2. 输入示例

**文本输入示例**：
- "我今天非常开心，心情很棒！"
- "这个产品质量太差了，我很失望。"
- "I'm so excited about this opportunity!"

**音频/视频**：
- 建议时长：不超过10秒
- 建议格式：音频用WAV，视频用MP4
- 内容：清晰的语音或表情

### 3. 结果解读

**情感类别**：
- 强烈负面 😫 - 愤怒、极度悲伤等
- 负面 😞 - 失望、不满意等
- 弱负面 😕 - 轻微不满、担忧等
- 中性 😐 - 平静、无明显情感倾向
- 弱正面 🙂 - 轻微满意、平静愉悦
- 正面 😊 - 开心、满意等
- 强烈正面 🤩 - 兴奋、狂喜等

**置信度**：
- > 80%：高度可信
- 60%-80%：较为可信
- < 60%：不确定性较高

## 界面预览

```
┌─────────────────────────────────────────────────────┐
│  😊 多模态情感分析系统                                │
├──────────────┬──────────────────────────────────────┤
│  侧边栏       │  主区域                               │
│              │                                      │
│  ⚙️ 设置     │  📝 输入区域                         │
│              │  ┌────────────┬────────────┐        │
│  关于系统    │  │ 文本输入    │ 音频上传   │        │
│  使用说明    │  └────────────┴────────────┘        │
│  系统信息    │  ┌──────────────────────────┐       │
│              │  │     视频上传              │       │
│  [高级设置]  │  └──────────────────────────┘       │
│              │  ┌──────────────────────────┐       │
│              │  │   [🚀 开始分析]          │       │
│              │  └──────────────────────────┘       │
│              │                                      │
│              │  📊 分析结果                         │
│              │  - 预测情感: 😊 正面                 │
│              │  - 置信度: 85.50%                   │
│              │  - 情感分布图                        │
└──────────────┴──────────────────────────────────────┘
```

## 文件结构

```
web/
├── app.py           # Streamlit应用主文件
├── README.md        # 本文档
└── static/          # 静态资源目录（预留）
```

## 配置说明

### 端口配置

默认使用8501端口，可在 `config.py` 中修改：

```python
WEB_PORT = 8501
WEB_HOST = "localhost"
```

### 文件上传限制

默认最大上传200MB，可在 `config.py` 中修改：

```python
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB
```

### 临时文件目录

临时文件保存在项目根目录的 `temp/` 文件夹，分析完成后自动清理。

## 故障排除

### 问题1: 模型加载失败

**症状**: 界面显示"模型加载失败"

**解决方案**:
1. 检查是否安装了所有依赖
2. 确保有足够的磁盘空间下载模型
3. 检查网络连接（首次运行需要下载预训练模型）

```bash
# 重新安装依赖
uv pip install -r requirements.txt
```

### 问题2: 音频/视频分析失败

**症状**: 显示"特征提取失败"

**解决方案**:
1. 确认文件格式正确
2. 尝试用更短的文件（建议<10秒）
3. 检查文件是否损坏

### 问题3: 端口被占用

**症状**: 启动时显示"Address already in use"

**解决方案**:
```bash
# 使用其他端口
streamlit run web/app.py --server.port 8502
```

### 问题4: 中文显示乱码

**症状**: 界面或图表中中文显示为方块

**解决方案**:
1. 确保系统已安装中文字体
2. Windows通常会自动使用SimHei或Microsoft YaHei
3. Linux可能需要手动安装：

```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-zenhei

# CentOS/RHEL
sudo yum install wqy-zenhei-fonts
```

## 性能优化

### 模型缓存

应用使用 `@st.cache_resource` 缓存模型，首次加载后会保留在内存中，后续交互无需重新加载。

### GPU加速

如果系统有CUDA支持的GPU，模型会自动使用GPU加速：

```python
# 自动检测，无需配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 批量分析

当前版本支持单个输入分析，未来版本将支持批量文件上传和分析。

## 开发说明

### 代码结构

```python
# 页面配置
st.set_page_config(...)

# 模型加载（缓存）
@st.cache_resource
def load_predictor():
    ...

# 侧边栏
def render_sidebar():
    ...

# 输入区域
def render_input_area():
    ...

# 结果展示
def render_results(result):
    ...

# 主函数
def main():
    ...
```

### 自定义样式

可以在 `app.py` 中修改Streamlit的默认样式：

```python
# 修改主题
st.set_page_config(
    page_title="多模态情感分析系统",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### 扩展功能

**添加新的输入模态**：
1. 在 `render_input_area()` 添加上传组件
2. 在 `main()` 中处理文件保存
3. 调用 `predictor.predict()` 传入新模态

**自定义可视化**：
1. 修改 `plot_emotion_distribution()` 函数
2. 或添加新的绘图函数

**添加导出功能**：
```python
# 导出为CSV
import csv
csv = result.to_csv()

# 导出为JSON
import json
json_str = json.dumps(result, ensure_ascii=False)
```

## 相关文档

- [Streamlit官方文档](https://docs.streamlit.io/)
- [项目主文档](../README.md)
- [配置说明](../config.py)

## 许可证

本项目遵循项目根目录的许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 加入讨论组

---

**最后更新**: 2026-01-29
**版本**: 1.0.0
