#!/usr/bin/env python3
"""
MonkeyOCR Frontend Demo
独立的前端应用，通过调用FastAPI接口实现功能
"""

import gradio as gr
import requests
import os
import json
from typing import Optional
from loguru import logger
import io
from PIL import Image

# API配置
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

class MonkeyOCRClient:
    """MonkeyOCR API客户端"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        
    def parse_pdf(self, file_path: str) -> dict:
        """调用API解析PDF"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                response = requests.post(f"{self.base_url}/parse-pdf", files=files, timeout=300)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "message": f"API错误: {response.status_code} - {response.text}"
                }
        except Exception as e:
            logger.error(f"Error calling parse API: {e}")
            return {
                "success": False,
                "message": f"网络错误: {str(e)}"
            }
    
    def chat_with_image(self, file_path: str, instruction: str) -> dict:
        """调用API进行图像对话"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                data = {'instruction': instruction}
                response = requests.post(f"{self.base_url}/chat-with-image", files=files, data=data, timeout=300)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "message": f"API错误: {response.status_code} - {response.text}"
                }
        except Exception as e:
            logger.error(f"Error calling chat API: {e}")
            return {
                "success": False,
                "message": f"网络错误: {str(e)}"
            }
    
    def preview_file(self, file_path: str, page: int = 1) -> dict:
        """调用API预览文件"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                data = {'page': page}
                response = requests.post(f"{self.base_url}/preview-file", files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "message": f"API错误: {response.status_code} - {response.text}"
                }
        except Exception as e:
            logger.error(f"Error calling preview API: {e}")
            return {
                "success": False,
                "message": f"网络错误: {str(e)}"
            }
    
    def health_check(self) -> bool:
        """检查API健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False

# 初始化客户端
client = MonkeyOCRClient()

# 全局状态
current_file = None
preview_cache = {
    "total_pages": 0,
    "current_page": 1
}

def check_api_connection():
    """检查API连接状态"""
    if client.health_check():
        return "✅ API连接正常"
    else:
        return f"❌ 无法连接到API服务 ({client.base_url})"

def load_and_preview_file(file):
    """加载并预览文件"""
    global current_file, preview_cache
    
    if file is None:
        return None, "请上传文件", "0 / 0"
    
    current_file = file
    
    # 调用API预览
    result = client.preview_file(file, page=1)
    
    if result["success"]:
        preview_cache["total_pages"] = result["total_pages"]
        preview_cache["current_page"] = 1
        
        # 获取预览图像
        if result["image_url"]:
            try:
                image_response = requests.get(f"{client.base_url}{result['image_url']}")
                if image_response.status_code == 200:
                    image = Image.open(io.BytesIO(image_response.content))
                    page_info = f"{result['current_page']} / {result['total_pages']}"
                    return image, "文件加载成功", page_info
            except Exception as e:
                logger.error(f"Error loading preview image: {e}")
        
        return None, "预览加载失败", f"1 / {result['total_pages']}"
    else:
        return None, f"文件加载失败: {result['message']}", "0 / 0"

def turn_page(direction):
    """翻页功能"""
    global current_file, preview_cache
    
    if not current_file or preview_cache["total_pages"] == 0:
        return None, "0 / 0"
    
    new_page = preview_cache["current_page"]
    if direction == "prev" and new_page > 1:
        new_page -= 1
    elif direction == "next" and new_page < preview_cache["total_pages"]:
        new_page += 1
    
    if new_page != preview_cache["current_page"]:
        result = client.preview_file(current_file, page=new_page)
        
        if result["success"]:
            preview_cache["current_page"] = new_page
            
            try:
                image_response = requests.get(f"{client.base_url}{result['image_url']}")
                if image_response.status_code == 200:
                    image = Image.open(io.BytesIO(image_response.content))
                    page_info = f"{result['current_page']} / {result['total_pages']}"
                    return image, page_info
            except Exception as e:
                logger.error(f"Error loading page image: {e}")
    
    page_info = f"{preview_cache['current_page']} / {preview_cache['total_pages']}"
    return None, page_info

def parse_document(file):
    """解析文档"""
    if file is None:
        return "请先上传文件", "请先上传文件", gr.update(value="❌ 请先上传文件", visible=True), gr.update(value="❌ 请先上传文件", visible=True)
    
    try:
        # 调用API解析
        result = client.parse_pdf(file)
        
        if result["success"]:
            markdown_content = result.get("markdown_content", "")
            markdown_raw = result.get("markdown_raw", "")
            
            # 处理下载链接
            layout_pdf_link = ""
            zip_link = ""
            
            if result.get("layout_pdf_url"):
                layout_pdf_link = f"{client.base_url}{result['layout_pdf_url']}"
            if result.get("markdown_zip_url"):
                zip_link = f"{client.base_url}{result['markdown_zip_url']}"
            
            return (
                markdown_content or "解析完成，但内容为空",
                markdown_raw or "解析完成，但内容为空",
                gr.update(value=layout_pdf_link, visible=bool(layout_pdf_link)),
                gr.update(value=zip_link, visible=bool(zip_link))
            )
        else:
            error_msg = f"解析失败: {result['message']}"
            return (
                error_msg,
                error_msg,
                gr.update(value="❌ 解析失败", visible=True),
                gr.update(value="❌ 解析失败", visible=True)
            )
            
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        logger.error(f"Error in parse_document: {e}")
        return (
            error_msg,
            error_msg,
            gr.update(value="❌ 处理失败", visible=True),
            gr.update(value="❌ 处理失败", visible=True)
        )

def chat_with_document(instruction, file):
    """与文档对话"""
    if file is None:
        return "请先上传文件", "请先上传文件"
    
    try:
        # 调用API对话
        result = client.chat_with_image(file, instruction)
        
        if result["success"]:
            content = result.get("content", "")
            return content, content
        else:
            error_msg = f"对话失败: {result['message']}"
            return error_msg, error_msg
            
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        logger.error(f"Error in chat_with_document: {e}")
        return error_msg, error_msg

def clear_all():
    """清空所有内容"""
    global current_file, preview_cache
    current_file = None
    preview_cache = {"total_pages": 0, "current_page": 1}
    
    return (
        None,  # file_input
        None,  # preview_image
        "## 请上传文件并点击解析",  # markdown_output
        "请上传文件并点击解析",  # raw_output
        "0 / 0",  # page_info
        gr.update(value="", visible=False),  # pdf_download
        gr.update(value="", visible=False),  # zip_download
        check_api_connection()  # status
    )

# 预定义指令
instructions = {
    "文本识别": "Please output the text content from the image.",
    "公式识别": "Please write out the expression of the formula in the image using LaTeX format.",
    "表格识别(HTML)": "This is the image of a table. Please output the table in html format.",
    "表格识别(LaTeX)": "Please output the table in the image in LaTeX format."
}

# 创建Gradio界面
def create_interface():
    css = """
    .status-box {
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .page-info {
        text-align: center;
        padding: 8px;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin: 5px;
    }
    """
    
    with gr.Blocks(theme="ocean", css=css, title="MonkeyOCR Demo") as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🐵 MonkeyOCR Demo</h1>
            <p><em>基于 FastAPI 的独立前端演示</em></p>
        </div>
        """)
        
        # API状态显示
        with gr.Row():
            api_status = gr.HTML(value=check_api_connection(), elem_classes="status-box")
            gr.Button("🔄 刷新状态", size="sm").click(
                fn=check_api_connection,
                outputs=api_status
            )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 文件上传")
                file_input = gr.File(
                    label="选择PDF或图像文件",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png"],
                    type="filepath"
                )
                
                gr.Markdown("### 💬 对话设置")
                instruction_dropdown = gr.Dropdown(
                    choices=list(instructions.keys()),
                    value="文本识别",
                    label="选择任务类型"
                )
                custom_instruction = gr.Textbox(
                    label="或输入自定义指令",
                    placeholder="输入自定义指令...",
                    lines=2
                )
                
                gr.Markdown("### 🎛️ 操作")
                with gr.Row():
                    parse_btn = gr.Button("📄 解析文档", variant="primary")
                    chat_btn = gr.Button("💬 对话", variant="secondary")
                    clear_btn = gr.Button("🗑️ 清空", variant="huggingface")
            
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 👁️ 文件预览")
                        preview_image = gr.Image(
                            label="预览",
                            height=400,
                            show_label=False
                        )
                        
                        with gr.Row():
                            prev_btn = gr.Button("⬅️ 上一页")
                            page_info = gr.HTML(value="0 / 0", elem_classes="page-info")
                            next_btn = gr.Button("下一页 ➡️")
                    
                    with gr.Column():
                        gr.Markdown("### 📊 结果显示")
                        with gr.Tabs():
                            with gr.TabItem("Markdown渲染"):
                                markdown_output = gr.Markdown(
                                    value="## 请上传文件并点击解析",
                                    height=400,
                                    latex_delimiters=[
                                        {"left": "$$", "right": "$$", "display": True},
                                        {"left": "$", "right": "$", "display": False},
                                    ]
                                )
                            with gr.TabItem("原始文本"):
                                raw_output = gr.Textbox(
                                    value="请上传文件并点击解析",
                                    lines=20,
                                    max_lines=30,
                                    show_copy_button=True
                                )
                
                with gr.Row():
                    pdf_download = gr.HTML(
                        value="",
                        visible=False,
                        label="下载PDF布局"
                    )
                    zip_download = gr.HTML(
                        value="",
                        visible=False,
                        label="下载Markdown"
                    )
        
        # 事件绑定
        file_input.upload(
            fn=load_and_preview_file,
            inputs=file_input,
            outputs=[preview_image, api_status, page_info]
        )
        
        prev_btn.click(
            fn=lambda: turn_page("prev"),
            outputs=[preview_image, page_info]
        )
        
        next_btn.click(
            fn=lambda: turn_page("next"),
            outputs=[preview_image, page_info]
        )
        
        # 解析按钮
        parse_btn.click(
            fn=parse_document,
            inputs=file_input,
            outputs=[markdown_output, raw_output, pdf_download, zip_download]
        )
        
        # 对话按钮
        def handle_chat(instruction_key, custom_instr, file):
            # 如果有自定义指令，使用自定义指令，否则使用预定义指令
            if custom_instr.strip():
                instruction = custom_instr.strip()
            else:
                instruction = instructions.get(instruction_key, instructions["文本识别"])
            
            return chat_with_document(instruction, file)
        
        chat_btn.click(
            fn=handle_chat,
            inputs=[instruction_dropdown, custom_instruction, file_input],
            outputs=[markdown_output, raw_output]
        )
        
        # 清空按钮
        clear_btn.click(
            fn=clear_all,
            outputs=[
                file_input, preview_image, markdown_output, 
                raw_output, page_info, pdf_download, zip_download, api_status
            ]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    
    # 启动参数
    host = os.getenv('DEMO_HOST', '0.0.0.0')
    port = int(os.getenv('DEMO_PORT', '8002'))
    
    logger.info(f"Starting MonkeyOCR Demo on {host}:{port}")
    logger.info(f"API Base URL: {API_BASE_URL}")
    
    demo.queue().launch(
        server_name=host,
        server_port=port,
        debug=True,
        share=False
    )