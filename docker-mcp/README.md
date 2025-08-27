# MonkeyOCR MCP 服务

## 📋 服务概述

MonkeyOCR MCP (Model Context Protocol) 服务是一个基于标准MCP协议的OCR服务，提供了与MonkeyOCR API完全兼容的接口。该服务允许通过MCP协议访问MonkeyOCR的所有功能，包括文本提取、公式识别、表格解析、文档分析等。

## 🛠️ 功能特性

### 核心OCR功能
- **extract_text** - 从图像或PDF中提取文本内容
- **extract_formula** - 识别数学公式并输出LaTeX格式
- **extract_table** - 提取表格并输出HTML格式

### 文档解析功能
- **parse_document** - 完整文档解析（PDF或图像）
- **parse_document_split** - 按页分割的文档解析
- **parse_pdf** - 完整的PDF转Markdown处理（包含布局分析）

### 增强功能
- **chat_with_image** - 基于图像的聊天功能（支持自定义指令）
- **preview_file** - 文件预览功能（支持分页）

## 📁 项目结构
docker-mcp/
├── docker-compose.yml      # Docker Compose编排配置
├── nginx.conf             # Nginx反向代理配置（可选）
├── start.sh               # 启动脚本
├── stop.sh                # 停止脚本
├── README.md              # 本文档
└── mcp-server/            # MCP服务器代码
├── Dockerfile         # Docker构建文件
├── main.py           # MCP服务器主程序
└── requirements.txt  # Python依赖


## 🚀 快速开始

### 前提条件
- Docker 和 Docker Compose
- MonkeyOCR API服务（运行在localhost:8000）

### 1. 构建和启动服务

```bash
# 进入docker-mcp目录
cd docker-mcp

# 启动所有服务（MCP服务器 + MonkeyOCR API）
./start.sh

# 或者使用Docker Compose直接启动
docker-compose up -d
```

### 2. 验证服务状态

```bash
# 检查服务状态
docker-compose ps

# 查看MCP服务器日志
docker-compose logs monkeyocr-mcp
```

### 3. 停止服务

```bash
# 停止所有服务
./stop.sh

# 或者使用Docker Compose
docker-compose down
```

## ⚙️ 配置说明

### 环境变量配置

在 `.env` 文件中配置以下参数：

```env
# MonkeyOCR API地址
MONKEYOCR_API_URL=http://monkeyocr-api:8000

# MCP服务器端口
MCP_PORT=8001

# 日志级别
LOG_LEVEL=INFO
```

### Docker Compose配置

服务包含两个主要容器：
1. **monkeyocr-api** - MonkeyOCR主API服务
2. **monkeyocr-mcp** - MCP协议适配器服务

## 🔧 自定义构建

### 1. 构建MCP服务器镜像

```bash
# 进入mcp-server目录
cd mcp-server

# 构建Docker镜像
docker build -t monkeyocr-mcp .
```

### 2. 手动运行MCP服务器

```bash
# 安装依赖
pip install -r requirements.txt

# 运行MCP服务器
python main.py
```

## 📊 使用示例

### 通过MCP客户端调用

```python
import as极速
from mcp.client import Client

async def demo():
    async with Client("monkeyocr-mcp") as client:
        # 列出所有可用工具
        tools = await client.list极速()
        print("Available tools:", [tool['name'] for tool in tools])
        
        # 调用文本提取工具
        result = await client.call_tool("extract_text", {
            "file": "base64_encoded_file_data"
        })
        print("Extracted text:", result)

asyncio.run(demo())
```

### cURL示例

```bash
# 通过HTTP调用MCP服务（如果配置了HTTP适配器）
curl -X POST http://localhost:8001/tools/call \
  -H "Content-Type: application/json" \
 极速 '{
    "name": "extract_text",
    "arguments": {
      "file": "base64_data_here"
    }
  }'
```

## 🐛 故障排除

### 常见问题

1. **连接失败**：确保MonkeyOCR API服务正在运行
2. **依赖问题**：检查Python包版本兼容性
3. **内存不足**：调整Docker内存限制

### 日志查看

```bash
# 查看详细日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs monkeyocr-mcp
```

## 📈 性能优化

- 启用GPU加速（如果可用）
- 调整Docker资源极速
- 配置适当的批处理大小

## 🔗 相关资源

- [MCP协议文档](https://modelcontextprotocol.io)
- [MonkeyOCR项目文档](https://github.com/Yuliang-Liu/MonkeyOCR)
- [Docker文档](https://docs.docker.com)

## 📝 更新日志

- **v1.0.0** - 初始版本，支持所有MonkeyOCR API功能
- **v1.1.0** - 添加增强功能（PDF解析、图像聊天、文件预览）

---

这个MCP服务现在完全覆盖了MonkeyOCR的所有API功能，可以通过标准的MCP协议进行调用，为AI应用提供了强大的OCR能力集成。