# MonkeyOCR Frontend Demo

基于 FastAPI 接口的独立前端演示项目。

## 架构说明

这是一个完全独立的前端项目，通过调用 FastAPI 接口实现 MonkeyOCR 的所有功能：

```
┌─────────────────┐    HTTP API    ┌──────────────────┐
│   Frontend      │ ◄─────────────► │   Enhanced API   │
│   (Gradio)      │                │   (FastAPI)      │
│   Port: 7860    │                │   Port: 8000     │
└─────────────────┘                └──────────────────┘
                                           │
                                           ▼
                                   ┌──────────────────┐
                                   │   MonkeyOCR      │
                                   │   Model          │
                                   └──────────────────┘
```

## 功能特性

### API 功能 (Enhanced FastAPI)
- ✅ **PDF 转 Markdown**: 完整的文档解析功能
- ✅ **图像对话**: 支持多种识别任务
- ✅ **文件预览**: 支持多页 PDF 预览
- ✅ **下载支持**: 生成 PDF 布局和 Markdown 压缩包
- ✅ **CORS 支持**: 支持跨域前端访问

### 前端功能 (Gradio Demo)
- 🎨 **现代界面**: 基于 Gradio 的响应式界面
- 📁 **文件管理**: 拖拽上传，多格式支持
- 👁️ **实时预览**: 支持翻页预览
- 💬 **智能对话**: 预定义和自定义指令
- 📊 **结果展示**: Markdown 渲染和原始文本
- ⬇️ **一键下载**: 直接下载处理结果

## 快速开始

### 方式一：使用 Docker Compose（推荐）

```bash
cd newdemo
docker-compose up -d
```

服务地址：
- 前端界面: http://localhost:7860
- API 服务: http://localhost:8000
- API 文档: http://localhost:8000/docs

### 方式二：分别启动

#### 1. 启动 API 服务

```bash
# 在主项目的 docker 目录中
cd ../docker
docker-compose up -d monkeyocr-enhanced-api
```

#### 2. 启动前端

```bash
# 本地启动
pip install -r requirements.txt
API_BASE_URL=http://localhost:8000 python app.py

# 或 Docker 启动
docker build -t monkeyocr-demo .
docker run -p 7860:7860 -e API_BASE_URL=http://host.docker.internal:8000 monkeyocr-demo
```

## 配置说明

### 环境变量

#### API 服务
- `FASTAPI_HOST`: API 主机地址 (默认: 0.0.0.0)
- `FASTAPI_PORT`: API 端口 (默认: 8000)
- `TMPDIR`: 临时文件目录

#### 前端服务
- `API_BASE_URL`: API 服务地址 (默认: http://localhost:8000)
- `DEMO_HOST`: 前端主机地址 (默认: 0.0.0.0)
- `DEMO_PORT`: 前端端口 (默认: 7860)

### API 接口说明

#### 1. 解析 PDF
```http
POST /parse-pdf
Content-Type: multipart/form-data

# 响应
{
  "success": bool,
  "message": str,
  "markdown_content": str,  # 渲染后的 Markdown
  "markdown_raw": str,      # 原始 Markdown
  "layout_pdf_url": str,    # PDF 布局下载链接
  "markdown_zip_url": str   # Markdown 压缩包下载链接
}
```

#### 2. 图像对话
```http
POST /chat-with-image
Content-Type: multipart/form-data

# 参数
file: 文件
instruction: 指令文本

# 响应
{
  "success": bool,
  "content": str,
  "message": str
}
```

#### 3. 文件预览
```http
POST /preview-file
Content-Type: multipart/form-data

# 参数
file: 文件
page: 页码 (默认: 1)

# 响应
{
  "success": bool,
  "total_pages": int,
  "current_page": int,
  "image_url": str  # 预览图片链接
}
```

## 开发说明

### 项目结构

```
newdemo/
├── app.py              # 前端应用主文件
├── requirements.txt    # Python 依赖
├── Dockerfile          # 前端 Docker 配置
├── docker-compose.yml  # 完整服务编排
└── README.md          # 本文档
```

### 自定义开发

1. **修改前端界面**: 编辑 `app.py` 中的 Gradio 界面定义
2. **添加新功能**: 在 `enhanced_api.py` 中添加新的 API 端点
3. **调整样式**: 修改 `app.py` 中的 CSS 样式

### 调试模式

```bash
# API 调试
cd ../docker
python enhanced_api.py

# 前端调试
API_BASE_URL=http://localhost:8000 python app.py
```

## 故障排除

### 常见问题

1. **前端无法连接 API**
   - 检查 `API_BASE_URL` 环境变量
   - 确认 API 服务正常运行
   - 检查防火墙和网络设置

2. **API 服务启动失败**
   - 检查 GPU 驱动和 CUDA 环境
   - 确认模型文件挂载正确
   - 查看容器日志: `docker logs monkeyocr-api`

3. **文件上传失败**
   - 检查文件格式（支持 PDF, JPG, PNG）
   - 确认文件大小限制
   - 查看 API 错误日志

### 日志查看

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f monkeyocr-api
docker-compose logs -f monkeyocr-demo
```

## 生产部署建议

1. **安全配置**
   - 配置 CORS 白名单
   - 添加 API 认证
   - 使用 HTTPS

2. **性能优化**
   - 使用 Nginx 反向代理
   - 配置文件缓存
   - 启用 Gzip 压缩

3. **监控告警**
   - 配置健康检查
   - 添加性能监控
   - 设置日志收集

## 技术栈

- **后端**: FastAPI, Uvicorn, MonkeyOCR
- **前端**: Gradio, Requests, Pillow
- **容器**: Docker, Docker Compose
- **GPU**: NVIDIA Docker Runtime