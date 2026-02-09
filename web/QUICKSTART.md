# 快速开始指南

## 5分钟快速启动多模态情感分析系统

### Windows用户

**方法1: 双击运行（最简单）**
```
1. 双击 web/run.bat
2. 等待浏览器自动打开
3. 开始使用！
```

**方法2: 命令行**
```cmd
cd C:\Users\lenovo\Desktop\biyesheji
streamlit run web/app.py
```

### Linux/Mac用户

**方法1: 运行脚本**
```bash
cd /path/to/biyesheji
./web/run.sh
```

**方法2: 直接命令**
```bash
streamlit run web/app.py
```

## 测试应用

运行测试脚本验证配置：
```bash
python web/test_app.py
```

预期输出：
```
✅ 所有测试通过！Web应用已准备就绪。
```

## 使用示例

### 示例1: 文本情感分析
1. 在"文本输入"框输入：`我今天非常开心！`
2. 点击"开始分析"
3. 查看预测结果

### 示例2: 音频情感分析
1. 点击"音频上传"选择音频文件
2. 支持格式：WAV, MP3, M4A
3. 点击"开始分析"

### 示例3: 多模态分析
1. 输入文本
2. 上传音频或视频
3. 点击"开始分析"
4. 获得更准确的预测结果

## 支持的情感类别

| 中文 | 英文 | Emoji |
|------|------|-------|
| 强烈负面 | strong_negative | 😫 |
| 负面 | negative | 😞 |
| 弱负面 | weak_negative | 😕 |
| 中性 | neutral | 😐 |
| 弱正面 | weak_positive | 🙂 |
| 正面 | positive | 😊 |
| 强烈正面 | strong_positive | 🤩 |

## 常见问题

### Q: 如何修改端口号？
A: 运行时指定端口：
```bash
streamlit run web/app.py --server.port 8080
```

### Q: 如何允许外部访问？
A: 指定地址为0.0.0.0：
```bash
streamlit run web/app.py --server.address 0.0.0.0
```

### Q: 如何关闭应用？
A: 在命令行按 `Ctrl+C`

### Q: 模型加载很慢怎么办？
A: 首次加载需要下载预训练模型，之后会使用缓存自动加速。

### Q: 支持哪些文件格式？
A:
- 音频：WAV, MP3, M4A
- 视频：MP4, AVI, MOV

## 下一步

- 📖 阅读完整文档：`web/README.md`
- 🧪 运行测试：`python web/test_app.py`
- 🎨 自定义界面：编辑 `web/app.py`
- 📊 查看配置：查看 `config.py`

## 技术支持

遇到问题？
1. 查看 `web/README.md` 故障排除章节
2. 检查测试输出：`python web/test_app.py`
3. 查看错误日志

---

**版本**: 1.0.0
**最后更新**: 2026-01-29
