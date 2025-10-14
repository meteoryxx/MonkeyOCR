#!/usr/bin/env python3
"""
MonkeyOCR Enhanced FastAPI Application
包含完整的PDF转Markdown功能和原有的OCR功能
"""

import os
import io
import tempfile
from typing import Optional, List
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import base64
import re
import subprocess
import uuid
import shutil
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from tempfile import gettempdir
import zipfile
from loguru import logger
import time

from magic_pdf.model.custom_model import MonkeyOCR
from magic_pdf.utils.load_image import pdf_to_images
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
from PIL import Image
import uvicorn

# 添加定时清理任务相关的导入
import schedule
import threading
import gc  # 添加垃圾回收模块
import psutil  # 添加进程监控模块

# Response models
class TaskResponse(BaseModel):
    success: bool
    task_type: str
    content: str
    message: Optional[str] = None

class ParseResponse(BaseModel):
    success: bool
    message: str
    output_dir: Optional[str] = None
    files: Optional[List[str]] = None
    download_url: Optional[str] = None
    # Enhanced features
    markdown_content: Optional[str] = None
    markdown_raw: Optional[str] = None
    layout_pdf_url: Optional[str] = None
    markdown_zip_url: Optional[str] = None
    # 新增字段用于纯文本 Markdown 内容
    markdown_text: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    content: str
    message: Optional[str] = None

class PreviewResponse(BaseModel):
    success: bool
    total_pages: int
    current_page: int
    image_url: Optional[str] = None

# Task instructions for single task recognition
TASK_INSTRUCTIONS = {
    'text': 'Please output the text content from the image.',
    'formula': 'Please write out the expression of the formula in the image using LaTeX format.',
    'table': 'This is the image of a table. Please output the table in html format.'
}

# Global model instance and lock
monkey_ocr_model = None
supports_async = False
model_lock = asyncio.Lock()
executor = ThreadPoolExecutor(max_workers=4)

def initialize_model():
    """Initialize MonkeyOCR model"""
    global monkey_ocr_model
    global supports_async
    if monkey_ocr_model is None:
        config_path = os.getenv("MONKEYOCR_CONFIG", "model_configs.yaml")
        monkey_ocr_model = MonkeyOCR(config_path)
        supports_async = is_async_model(monkey_ocr_model)
    return monkey_ocr_model

def is_async_model(model: MonkeyOCR) -> bool:
    """Check if the model supports async concurrent calls"""
    if hasattr(model, 'chat_model'):
        chat_model = model.chat_model
        # More specific check for async models
        is_async = hasattr(chat_model, 'async_batch_inference')
        logger.info(f"Model {chat_model.__class__.__name__} supports async: {is_async}")
        return is_async
    return False

async def smart_model_call(func, *args, **kwargs):
    """
    Smart wrapper that automatically chooses between concurrent and blocking calls
    based on the model's capabilities
    """
    global monkey_ocr_model, model_lock
    
    if not monkey_ocr_model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if supports_async:
        # For async models, no need for model_lock, can run concurrently
        logger.info("Using concurrent execution (async model detected)")
        # Use asyncio's thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    else:
        # For sync models, use model_lock to prevent conflicts
        logger.info("Using blocking execution with lock (sync model detected)")
        async with model_lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

def render_latex_table_to_image(latex_content, temp_dir):
    """Render LaTeX table to image and return base64 encoding"""
    try:
        # Use regex to extract tabular environment content
        pattern = r"(\\begin\{tabular\}.*?\\end\{tabular\})"
        matches = re.findall(pattern, latex_content, re.DOTALL)
        
        if matches:
            table_content = matches[0]
        elif '\\begin{tabular}' in latex_content:
            if '\\end{tabular}' not in latex_content:
                table_content = latex_content + '\n\\end{tabular}'
            else:
                table_content = latex_content
        else:
            return latex_content
        
        # Build complete LaTeX document
        full_latex = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{bm}
\usepackage{multirow}
\usepackage{array}
\usepackage{colortbl}
\usepackage[table]{xcolor}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{makecell}
\usepackage[active,tightpage]{preview}
\PreviewEnvironment{tabular}
\begin{document}
""" + table_content + r"""
\end{document}
"""
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        tex_path = os.path.join(temp_dir, f"table_{unique_id}.tex")
        pdf_path = os.path.join(temp_dir, f"table_{unique_id}.pdf")
        png_path = os.path.join(temp_dir, f"table_{unique_id}.png")
        
        # Write tex file
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(full_latex)
        
        # Call pdflatex to generate PDF
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", temp_dir, tex_path], 
            timeout=20,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return f"<pre>{latex_content}</pre>"
        
        if not os.path.exists(pdf_path):
            return f"<pre>{latex_content}</pre>"
        
        # Convert PDF to PNG image
        images = pdf_to_images(pdf_path)
        images[0].save(png_path, "PNG")
        
        # Read image and convert to base64
        with open(png_path, "rb") as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        
        # Clean up temporary files
        for file_path in [tex_path, pdf_path, png_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;height:auto;">'
        
    except Exception as e:
        logger.error(f"LaTeX rendering error: {e}")
        return f"<pre>{latex_content}</pre>"

def process_pdf_to_markdown(file_path: str) -> tuple:
    """Process PDF to markdown with enhanced features"""
    try:
        parent_path = os.path.dirname(file_path)
        full_name = os.path.basename(file_path)
        name = '.'.join(full_name.split(".")[:-1])
        local_image_dir = os.path.join(parent_path, "markdown", "images")
        local_md_dir = os.path.join(parent_path, "markdown")
        image_dir = str(os.path.basename(local_image_dir))
        
        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_md_dir, exist_ok=True)
        
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)   
        reader1 = FileBasedDataReader(parent_path)
        data_bytes = reader1.read(full_name)
        
        if full_name.split(".")[-1] in ['jpg', 'jpeg', 'png']:
            ds = ImageDataset(data_bytes)
        else:
            ds = PymuDocDataset(data_bytes)
        
        # Use the model for inference
        infer_result = ds.apply(doc_analyze_llm, MonkeyOCR_model=monkey_ocr_model)
        pipe_result = infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=monkey_ocr_model)
        
        # Generate layout PDF
        layout_pdf_path = os.path.join(parent_path, f"{name}_layout.pdf")
        pipe_result.draw_layout(layout_pdf_path)
        
        # Generate markdown
        pipe_result.dump_md(md_writer, f"{name}.md", image_dir)
        md_content_ori = FileBasedDataReader(local_md_dir).read(f"{name}.md").decode("utf-8")
        
        # Create temporary directory for LaTeX rendering
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Process HTML-wrapped LaTeX tables
            def replace_html_latex_table(match):
                html_content = match.group(1)
                if '\\begin{tabular}' in html_content:
                    return render_latex_table_to_image(html_content, temp_dir)
                else:
                    return match.group(0)
            
            # Use regex to replace LaTeX tables wrapped in <html>...</html>
            md_content = re.sub(r'<html>(.*?)</html>', replace_html_latex_table, md_content_ori, flags=re.DOTALL)
            
            # Convert local image links in markdown to base64 encoded HTML
            def replace_image_with_base64(match):
                img_path = match.group(1)
                if not os.path.isabs(img_path):
                    full_img_path = os.path.join(local_md_dir, img_path)
                else:
                    full_img_path = img_path
                
                try:
                    if os.path.exists(full_img_path):
                        with open(full_img_path, "rb") as f:
                            img_data = f.read()
                        img_base64 = base64.b64encode(img_data).decode("utf-8")
                        ext = os.path.splitext(full_img_path)[1].lower()
                        mime_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else f"image/{ext[1:]}"
                        return f'<img src="data:{mime_type};base64,{img_base64}" style="max-width:100%;height:auto;">'
                    else:
                        return match.group(0)
                except Exception:
                    return match.group(0)
            
            # Use regex to replace markdown image syntax ![alt](path)
            md_content = re.sub(r'!\[.*?\]\(([^)]+)\)', replace_image_with_base64, md_content)
            
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Create zip file - 只包含当前处理的文件，而不是全部文件
        zip_path = os.path.join(parent_path, f"{name}_markdown.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 只添加当前处理的文件到 ZIP 中
            for root, dirs, files in os.walk(local_md_dir):
                for file in files:
                    file_path_inner = os.path.join(root, file)
                    # 只添加当前处理的文件，通过更精确的文件名匹配确保只包含相关文件
                    # 同时确保包含 images 目录中的相关图片文件
                    if file.startswith(name) or file == f"{name}.md" or (root.endswith('images') and name in file):
                        arc_name = os.path.relpath(file_path_inner, local_md_dir)
                        zipf.write(file_path_inner, arc_name)
        
        return md_content, md_content_ori, layout_pdf_path, zip_path
        
    except Exception as e:
        logger.error(f"Error processing PDF to markdown: {e}")
        raise

async def smart_batch_model_call(images_and_questions_list, batch_func):
    """
    Smart batch processing that can handle multiple requests efficiently
    """
    global monkey_ocr_model
    
    if not monkey_ocr_model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if supports_async and hasattr(monkey_ocr_model.chat_model, 'async_batch_inference'):
        # Use native async batch processing for maximum efficiency
        logger.info(f"Using native async batch processing for {len(images_and_questions_list)} requests")
        
        # Flatten all images and questions
        all_images = []
        all_questions = []
        request_indices = []
        
        for i, (images, questions) in enumerate(images_and_questions_list):
            for img, q in zip(images, questions):
                all_images.append(img)
                all_questions.append(q)
                request_indices.append(i)
        
        # Single batch call for all requests
        try:
            # Use the chat model's async batch inference method properly
            all_results = await monkey_ocr_model.chat_model.async_batch_inference(all_images, all_questions)
        except Exception as e:
            logger.error(f"Async batch inference failed: {e}, falling back to individual processing")
            # Fallback to individual processing using the corrected method
            results = []
            for images, questions in images_and_questions_list:
                try:
                    # Use the thread-safe smart_model_call wrapper
                    result = await smart_model_call(batch_func, images, questions)
                    results.append(result)
                except Exception as inner_e:
                    logger.error(f"Individual processing also failed: {inner_e}")
                    results.append([f"Error: {str(inner_e)}"] * len(images))
            return results
        
        # Reconstruct results for each original request
        results = []
        result_idx = 0
        for images, questions in images_and_questions_list:
            request_results = []
            for _ in range(len(images)):
                request_results.append(all_results[result_idx])
                result_idx += 1
            results.append(request_results)
        
        return results
    
    elif supports_async:
        # Concurrent processing for async models
        logger.info(f"Using concurrent batch processing for {len(images_and_questions_list)} requests")
        tasks = []
        for images, questions in images_and_questions_list:
            task = smart_model_call(batch_func, images, questions)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Sequential processing for sync models
        logger.info(f"Using sequential batch processing for {len(images_and_questions_list)} requests")
        results = []
        for images, questions in images_and_questions_list:
            result = await smart_model_call(batch_func, images, questions)
            results.append(result)
        return results

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Startup
    try:
        initialize_model()
        model_type = "async-capable" if supports_async else "sync-only"
        logger.info(f"✅ MonkeyOCR model initialized successfully ({model_type})")
        
        # 启动定时清理任务
        start_cleanup_scheduler()
    except Exception as e:
        logger.info(f"❌ Failed to initialize MonkeyOCR model: {e}")
        raise
    
    yield
    
    # Shutdown
    global executor
    executor.shutdown(wait=True)
    logger.info("🔄 Application shutdown complete")

app = FastAPI(
    title="MonkeyOCR Enhanced API",
    description="Enhanced OCR and Document Parsing API with full PDF to Markdown conversion",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

temp_dir = os.getenv("TMPDIR", gettempdir())
logger.info(f"Using temporary directory: {temp_dir}")
os.makedirs(temp_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=temp_dir), name="static")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MonkeyOCR Enhanced API is running", 
        "version": "2.0.0",
        "endpoints": {
            # Original OCR endpoints
            "ocr_text": "/ocr/text",
            "ocr_url": "/ocr/url",
            "ocr_formula": "/ocr/formula", 
            "ocr_table": "/ocr/table",
            "parse": "/parse",
            "parse_split": "/parse/split",
            # Enhanced endpoints
            "parse_pdf": "/parse-pdf",
            "chat_with_image": "/chat-with-image", 
            "preview_file": "/preview-file",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": monkey_ocr_model is not None}

@app.post("/ocr/text", response_model=TaskResponse)
async def extract_text(file: UploadFile = File(...)):
    """Extract text from image or PDF"""
    return await perform_ocr_task(file, "text")

@app.post("/ocr/formula", response_model=TaskResponse)
async def extract_formula(file: UploadFile = File(...)):
    """Extract formulas from image or PDF"""
    return await perform_ocr_task(file, "formula")

@app.post("/ocr/table", response_model=TaskResponse)
async def extract_table(file: UploadFile = File(...)):
    """Extract tables from image or PDF"""
    return await perform_ocr_task(file, "table")

@app.post("/ocr/url", response_model=TaskResponse)
async def extract_text_from_url(
    url: str = Form(...),
    start_page: Optional[int] = Form(None),
    end_page: Optional[int] = Form(None)
):
    """Extract text from PDF file via URL with optional page range"""
    return await perform_ocr_task_from_url(url, "text", start_page, end_page)

@app.post("/parse", response_model=ParseResponse)
async def parse_document(file: UploadFile = File(...)):
    """Parse complete document (PDF or image)"""
    return await parse_document_internal(file, split_pages=False)

@app.post("/parse/split", response_model=ParseResponse)
async def parse_document_split(file: UploadFile = File(...)):
    """Parse complete document and split result by pages (PDF or image)"""
    return await parse_document_internal(file, split_pages=True)

async def async_parse_file(input_file_path: str, output_dir: str, split_pages: bool = False):
    """
    Optimized async version of parse_file that breaks down processing into async chunks
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    import uuid
    
    if not monkey_ocr_model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Get filename with unique identifier to avoid conflicts
    name_without_suff = '.'.join(os.path.basename(input_file_path).split(".")[:-1])
    unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
    safe_name = f"{name_without_suff}_{unique_id}"
    
    # Prepare output directory with unique name
    local_image_dir = os.path.join(output_dir, safe_name, "images")
    local_md_dir = os.path.join(output_dir, safe_name)
    image_dir = os.path.basename(local_image_dir)
    
    # Create directories asynchronously with better error handling
    def create_dir_safe(path):
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            # Directory already exists, that's fine
            pass
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise
    
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, local_image_dir)
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, local_md_dir)
    
    logger.info(f"Output dir: {local_md_dir}")
    
    # Read file content in thread pool
    def read_file_sync():
        from magic_pdf.data.data_reader_writer import FileBasedDataReader
        reader = FileBasedDataReader()
        return reader.read(input_file_path)
    
    file_bytes = await asyncio.get_event_loop().run_in_executor(None, read_file_sync)
    
    # Create dataset instance in thread pool
    def create_dataset_sync():
        from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
        file_extension = input_file_path.split(".")[-1].lower()
        if file_extension == "pdf":
            return PymuDocDataset(file_bytes)
        else:
            return ImageDataset(file_bytes)
    
    ds = await asyncio.get_event_loop().run_in_executor(None, create_dataset_sync)
    
    # Run inference in thread pool
    def run_inference_sync():
        from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
        return ds.apply(doc_analyze_llm, MonkeyOCR_model=monkey_ocr_model, split_pages=split_pages)
    
    logger.info("Starting document parsing...")
    start_time = time.time()
    
    # Use smart model call for inference
    if supports_async:
        # For async models, run without lock
        infer_result = await asyncio.get_event_loop().run_in_executor(None, run_inference_sync)
    else:
        # For sync models, use lock
        async with model_lock:
            infer_result = await asyncio.get_event_loop().run_in_executor(None, run_inference_sync)
    
    parsing_time = time.time() - start_time
    logger.info(f"Parsing time: {parsing_time:.2f}s")
    
    # Process results asynchronously
    await process_inference_results_async(
        infer_result, output_dir, safe_name, 
        local_image_dir, local_md_dir, image_dir, split_pages
    )
    
    return local_md_dir

async def process_inference_results_async(infer_result, output_dir, name_without_suff, 
                                        local_image_dir, local_md_dir, image_dir, split_pages):
    """
    Process inference results asynchronously
    """
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter
    
    def create_writers():
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)
        return image_writer, md_writer
    
    # Check if infer_result is a list (split pages)
    if isinstance(infer_result, list):
        logger.info(f"Processing {len(infer_result)} pages separately...")
        
        # Process pages concurrently
        tasks = []
        for page_idx, page_infer_result in enumerate(infer_result):
            task = process_single_page_async(
                page_infer_result, page_idx, output_dir, name_without_suff
            )
            tasks.append(task)
        
        # Wait for all page processing to complete
        await asyncio.gather(*tasks)
        
        logger.info(f"All {len(infer_result)} pages processed and saved in separate subdirectories")
    else:
        # Process single result
        logger.info("Processing as single result...")
        await process_single_result_async(
            infer_result, name_without_suff, local_image_dir, local_md_dir, image_dir
        )

async def process_single_page_async(page_infer_result, page_idx, output_dir, name_without_suff):
    """
    Process a single page result asynchronously
    """
    import uuid
    
    page_dir_name = f"page_{page_idx}"
    page_local_image_dir = os.path.join(output_dir, name_without_suff, page_dir_name, "images")
    page_local_md_dir = os.path.join(output_dir, name_without_suff, page_dir_name)
    page_image_dir = os.path.basename(page_local_image_dir)
    
    # Create page-specific directories with better error handling
    def create_dir_safe(path):
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            # Directory already exists, that's fine
            pass
        except Exception as e:
            logger.error(f"Failed to create page directory {path}: {e}")
            raise
    
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, page_local_image_dir)
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, page_local_md_dir)
    
    def process_page_sync():
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        
        # Create page-specific writers
        page_image_writer = FileBasedDataWriter(page_local_image_dir)
        page_md_writer = FileBasedDataWriter(page_local_md_dir)
        
        logger.info(f"Processing page {page_idx} - Output dir: {page_local_md_dir}")
        
        # Pipeline processing for this page
        page_pipe_result = page_infer_result.pipe_ocr_mode(page_image_writer, MonkeyOCR_model=monkey_ocr_model)
        
        # Save page-specific results
        page_infer_result.draw_model(os.path.join(page_local_md_dir, f"{name_without_suff}_page_{page_idx}_model.pdf"))
        page_pipe_result.draw_layout(os.path.join(page_local_md_dir, f"{name_without_suff}_page_{page_idx}_layout.pdf"))
        page_pipe_result.draw_span(os.path.join(page_local_md_dir, f"{name_without_suff}_page_{page_idx}_spans.pdf"))
        page_pipe_result.dump_md(page_md_writer, f"{name_without_suff}_page_{page_idx}.md", page_image_dir)
        page_pipe_result.dump_content_list(page_md_writer, f"{name_without_suff}_page_{page_idx}_content_list.json", page_image_dir)
        page_pipe_result.dump_middle_json(page_md_writer, f'{name_without_suff}_page_{page_idx}_middle.json')
    
    # Run page processing in thread pool
    await asyncio.get_event_loop().run_in_executor(None, process_page_sync)

async def process_single_result_async(infer_result, name_without_suff, local_image_dir, local_md_dir, image_dir):
    """
    Process single result asynchronously
    """
    def process_single_sync():
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)
        
        # Pipeline processing for single result
        pipe_result = infer_result.pipe_ocr_mode(image_writer, MonkeyOCR_model=monkey_ocr_model)
        
        # Save single result
        infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))
        pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))
        pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)
        pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')
    
    # Run processing in thread pool
    await asyncio.get_event_loop().run_in_executor(None, process_single_sync)

async def async_single_task_recognition(input_file_path: str, output_dir: str, task: str):
    """
    Optimized async version of single_task_recognition
    """
    import uuid
    
    logger.info(f"Starting async single task recognition: {task}")
    
    # Get filename with unique identifier to avoid conflicts
    name_without_suff = '.'.join(os.path.basename(input_file_path).split(".")[:-1])
    unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
    safe_name = f"{name_without_suff}_{unique_id}"
    
    # Prepare output directory with unique name
    local_md_dir = os.path.join(output_dir, safe_name)
    
    def create_dir_safe(path):
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            # Directory already exists, that's fine
            pass
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise
    
    await asyncio.get_event_loop().run_in_executor(None, create_dir_safe, local_md_dir)
    
    # Get task instruction
    instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS['text'])
    
    # Load images asynchronously
    def load_images_sync():
        file_extension = input_file_path.split(".")[-1].lower()
        images = []
        
        if file_extension == 'pdf':
            from magic_pdf.utils.load_image import pdf_to_images
            images = pdf_to_images(input_file_path)
        elif file_extension in ['jpg', 'jpeg', 'png']:
            from PIL import Image
            images = [input_file_path]
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        return images, file_extension
    
    images, file_extension = await asyncio.get_event_loop().run_in_executor(None, load_images_sync)
    
    # Perform recognition
    logger.info(f"Performing {task} recognition on {len(images)} image(s)...")
    start_time = time.time()
    
    # Prepare instructions for all images
    instructions = [instruction] * len(images)
    
    # Use chat model for recognition
    if supports_async and hasattr(monkey_ocr_model.chat_model, 'async_batch_inference'):
        # Use async batch inference if available
        try:
            responses = await monkey_ocr_model.chat_model.async_batch_inference(images, instructions)
        except Exception as e:
            logger.warning(f"Async batch inference failed: {e}, falling back to sync")
            responses = await asyncio.get_event_loop().run_in_executor(
                None, monkey_ocr_model.chat_model.batch_inference, images, instructions
            )
    else:
        # Use sync batch inference in thread pool
        responses = await asyncio.get_event_loop().run_in_executor(
            None, monkey_ocr_model.chat_model.batch_inference, images, instructions
        )
    
    recognition_time = time.time() - start_time
    logger.info(f"Recognition time: {recognition_time:.2f}s")
    
    # Combine and save results
    def save_results_sync():
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter
        
        md_writer = FileBasedDataWriter(local_md_dir)
        
        # Combine results
        combined_result = responses[0]
        for i, response in enumerate(responses):
            if i > 0:
                combined_result = combined_result + "\n\n" + response
        
        # Save result with original name (without unique suffix)
        result_filename = f"{name_without_suff}_{task}_result.md"
        md_writer.write(result_filename, combined_result.encode('utf-8'))
        
        return result_filename
    
    result_filename = await asyncio.get_event_loop().run_in_executor(None, save_results_sync)
    
    logger.info(f"Single task recognition completed!")
    logger.info(f"Result saved to: {os.path.join(local_md_dir, result_filename)}")
    
    # Clean up images
    def cleanup_images():
        try:
            for img in images:
                if hasattr(img, 'close'):
                    img.close()
        except Exception as cleanup_error:
            logger.warning(f"Warning: Error during cleanup: {cleanup_error}")
    
    await asyncio.get_event_loop().run_in_executor(None, cleanup_images)
    
    return local_md_dir

async def parse_document_internal(file: UploadFile, split_pages: bool = False):
    """Internal function to parse document with optional page splitting"""
    try:
        if not monkey_ocr_model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Validate file type - support both PDF and image files
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
        file_ext_with_dot = os.path.splitext(file.filename)[1].lower() if file.filename else ''
        
        if file_ext_with_dot not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext_with_dot}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Get original filename without extension
        original_name = '.'.join(file.filename.split('.')[:-1])
        
        # Save uploaded file temporarily with unique name to avoid conflicts
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext_with_dot, prefix=f"upload_{unique_suffix}_") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Create output directory with unique name
            output_dir = tempfile.mkdtemp(prefix=f"monkeyocr_parse_{unique_suffix}_")
            
            # Use optimized async parse function
            result_dir = await async_parse_file(temp_file_path, output_dir, split_pages)
            
            # List generated files
            files = []
            if os.path.exists(result_dir):
                for root, dirs, filenames in os.walk(result_dir):
                    for filename in filenames:
                        rel_path = os.path.relpath(os.path.join(root, filename), result_dir)
                        files.append(rel_path)
            
            # Create download URL with original filename and timestamp
            suffix = "_split" if split_pages else "_parsed"
            timestamp = int(time.time() * 1000)  # Use milliseconds for better uniqueness
            zip_filename = f"{original_name}{suffix}_{timestamp}_{unique_suffix}.zip"
            zip_path = os.path.join(temp_dir, zip_filename)
            
            # Create ZIP file asynchronously
            await create_zip_file_async(result_dir, zip_path, original_name, split_pages)
            
            download_url = f"/static/{zip_filename}"
            
            # Determine file type for response message
            file_type = "PDF" if file_ext_with_dot == '.pdf' else "image"
            parse_type = "with page splitting" if split_pages else "standard"
            
            return ParseResponse(
                success=True,
                message=f"{file_type} parsing ({parse_type}) completed successfully",
                output_dir=result_dir,
                files=files,
                download_url=download_url
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"Parsing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")

async def create_zip_file_async(result_dir, zip_path, original_name, split_pages):
    """Create ZIP file asynchronously"""
    def create_zip_sync():
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, filenames in os.walk(result_dir):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, result_dir)
                    
                    if split_pages:
                        # For split pages, maintain the page directory structure
                        # but add original name prefix
                        if rel_path.startswith('page_'):
                            # Keep the page structure: page_0/filename -> page_0/original_name_filename
                            parts = rel_path.split('/', 1)
                            if len(parts) == 2:
                                page_dir, filename_part = parts
                                if filename_part.startswith('images/'):
                                    # Handle images: page_0/images/img.jpg -> page_0/images/original_name_img.jpg
                                    img_name = filename_part.replace('images/', '')
                                    new_filename = f"{page_dir}/images/{img_name}"
                                else:
                                    # Handle other files in page directories
                                    new_filename = f"{page_dir}/{original_name}_{filename_part}"
                            else:
                                new_filename = f"{original_name}_{rel_path}"
                        else:
                            new_filename = f"{original_name}_{rel_path}"
                    else:
                        # Handle different file types
                        if filename.endswith('.md'):
                            new_filename = f"{original_name}.md"
                        elif filename.endswith('_content_list.json'):
                            new_filename = f"{original_name}_content_list.json"
                        elif filename.endswith('_middle.json'):
                            new_filename = f"{original_name}_middle.json"
                        elif filename.endswith('_model.pdf'):
                            new_filename = f"{original_name}_model.pdf"
                        elif filename.endswith('_layout.pdf'):
                            new_filename = f"{original_name}_layout.pdf"
                        elif filename.endswith('_spans.pdf'):
                            new_filename = f"{original_name}_spans.pdf"
                        else:
                            # For images and other files, keep relative path structure but rename
                            if 'images/' in rel_path:
                                # Keep images in images subfolder with original name prefix
                                image_name = os.path.basename(rel_path)
                                new_filename = f"images/{image_name}"
                            else:
                                new_filename = f"{original_name}_{filename}"
                    
                    zipf.write(file_path, new_filename)
    
    # Run ZIP creation in thread pool to avoid blocking
    await asyncio.get_event_loop().run_in_executor(None, create_zip_sync)

async def perform_ocr_task(file: UploadFile, task_type: str) -> TaskResponse:
    """Perform OCR task on uploaded file"""
    try:
        if not monkey_ocr_model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file temporarily with unique name
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, prefix=f"ocr_{unique_suffix}_") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Create output directory with unique name
            output_dir = tempfile.mkdtemp(prefix=f"monkeyocr_{task_type}_{unique_suffix}_")
            
            # Use optimized async single task recognition
            result_dir = await async_single_task_recognition(temp_file_path, output_dir, task_type)
            
            # Read result file
            def read_result_sync():
                result_files = [f for f in os.listdir(result_dir) if f.endswith(f'_{task_type}_result.md')]
                if not result_files:
                    raise Exception("No result file generated")
                
                result_file_path = os.path.join(result_dir, result_files[0])
                with open(result_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            content = await asyncio.get_event_loop().run_in_executor(None, read_result_sync)
            
            return TaskResponse(
                success=True,
                task_type=task_type,
                content=content,
                message=f"{task_type.capitalize()} extraction completed successfully"
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"OCR task failed: {str(e)}")
        return TaskResponse(
            success=False,
            task_type=task_type,
            content="",
            message=f"OCR task failed: {str(e)}"
        )

async def perform_ocr_task_from_url(url: str, task_type: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> TaskResponse:
    """Perform OCR task on PDF file from URL with optional page range"""
    try:
        if not monkey_ocr_model:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid URL")
        
        # Download file from URL
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine file extension
        content_type = response.headers.get('content-type', '')
        if 'application/pdf' in content_type:
            file_ext = '.pdf'
        elif 'image/jpeg' in content_type:
            file_ext = '.jpg'
        elif 'image/png' in content_type:
            file_ext = '.png'
        else:
            # Try to determine from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            file_ext = os.path.splitext(parsed_url.path)[1].lower()
            if file_ext not in ['.pdf', '.jpg', '.jpeg', '.png']:
                file_ext = '.pdf'  # Default to PDF
        
        # Save downloaded file temporarily with unique name
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, prefix=f"ocr_url_{unique_suffix}_") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        # If it's a PDF and page range is specified, extract pages
        if file_ext == '.pdf' and (start_page is not None or end_page is not None):
            from magic_pdf.utils.load_image import pdf_to_images
            from PyPDF2 import PdfWriter, PdfReader
            
            # Read PDF
            reader = PdfReader(temp_file_path)
            total_pages = len(reader.pages)
            
            # Set default values for page range
            start = max(1, start_page or 1)
            end = min(total_pages, end_page or total_pages)
            
            # Validate page range
            if start > end or start > total_pages or end > total_pages:
                raise HTTPException(status_code=400, detail="Invalid page range")
            
            # Extract pages
            writer = PdfWriter()
            for i in range(start-1, end):
                writer.add_page(reader.pages[i])
            
            # Save extracted pages to a new temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', prefix=f"ocr_url_extracted_{unique_suffix}_") as extracted_file:
                writer.write(extracted_file)
                extracted_file_path = extracted_file.name
            
            # Clean up original file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {cleanup_error}")
            
            # Update file path to extracted file
            temp_file_path = extracted_file_path
        
        try:
            # Create output directory with unique name
            output_dir = tempfile.mkdtemp(prefix=f"monkeyocr_{task_type}_url_{unique_suffix}_")
            
            # Use optimized async single task recognition
            result_dir = await async_single_task_recognition(temp_file_path, output_dir, task_type)
            
            # Read result file
            def read_result_sync():
                result_files = [f for f in os.listdir(result_dir) if f.endswith(f'_{task_type}_result.md')]
                if not result_files:
                    raise Exception("No result file generated")
                
                result_file_path = os.path.join(result_dir, result_files[0])
                with open(result_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            content = await asyncio.get_event_loop().run_in_executor(None, read_result_sync)
            
            return TaskResponse(
                success=True,
                task_type=task_type,
                content=content,
                message=f"{task_type.capitalize()} extraction from URL completed successfully"
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {temp_file_path}: {cleanup_error}")
            
    except Exception as e:
        logger.error(f"OCR task from URL failed: {str(e)}")
        return TaskResponse(
            success=False,
            task_type=task_type,
            content="",
            message=f"OCR task from URL failed: {str(e)}"
        )

@app.post("/parse-pdf", response_model=ParseResponse)
async def parse_pdf(file: UploadFile = File(...)):
    """Parse PDF to Markdown with full processing"""
    if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the file
        def process_file():
            return process_pdf_to_markdown(file_path)
        
        md_content, md_content_ori, layout_pdf_path, zip_path = await smart_model_call(process_file)
        
        # Generate URLs for downloadable files
        layout_pdf_url = f"/static/{os.path.basename(layout_pdf_path)}" if layout_pdf_path and os.path.exists(layout_pdf_path) else None
        zip_url = f"/static/{os.path.basename(zip_path)}" if zip_path and os.path.exists(zip_path) else None
        
        # 保存 Markdown 内容到临时文件，以便后续通过 API 获取
        name = '.'.join(filename.split(".")[:-1])
        md_file_path = os.path.join(temp_dir, f"{name}.md")
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(md_content_ori)
        
        return ParseResponse(
            success=True,
            message="PDF parsed successfully",
            markdown_content=md_content,
            markdown_raw=md_content_ori,
            markdown_text=md_content_ori,  # 新增纯文本 Markdown 内容
            layout_pdf_url=layout_pdf_url,
            markdown_zip_url=zip_url,
            files=[layout_pdf_path, zip_path] if layout_pdf_path and zip_path else []
        )
        
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        return ParseResponse(
            success=False,
            message=f"Error parsing PDF: {str(e)}"
        )

@app.get("/parse-pdf/text/{file_id}")
async def get_markdown_text(file_id: str):
    """获取纯文本 Markdown 内容"""
    try:
        # 在临时目录中查找对应的 Markdown 文件
        md_file_path = os.path.join(temp_dir, f"{file_id}.md")
        if os.path.exists(md_file_path):
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return JSONResponse(content={"markdown_text": content})
        else:
            raise HTTPException(status_code=404, detail="Markdown file not found")
    except Exception as e:
        logger.error(f"Error retrieving markdown text: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving markdown text: {str(e)}")

@app.get("/parse-pdf/raw/{file_id}")
async def get_raw_markdown(file_id: str):
    """获取原始 Markdown 内容（不带图片的纯文本）"""
    try:
        # 在临时目录中查找对应的 Markdown 文件
        md_file_path = os.path.join(temp_dir, f"{file_id}.md")
        if os.path.exists(md_file_path):
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return JSONResponse(content={"raw_markdown": content})
        else:
            raise HTTPException(status_code=404, detail="Markdown file not found")
    except Exception as e:
        logger.error(f"Error retrieving raw markdown: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving raw markdown: {str(e)}")

@app.post("/parse-pdf/selective", response_model=ParseResponse)
async def parse_pdf_selective(
    file: UploadFile = File(...),
    output_format: str = Form("html", description="Output format: 'html', 'text', or 'zip'")
):
    """Parse PDF to Markdown with selectable output format"""
    if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
    
    if output_format not in ["html", "text", "zip"]:
        raise HTTPException(status_code=400, detail="Invalid output format. Use 'html', 'text', or 'zip'")
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the file
        def process_file():
            return process_pdf_to_markdown(file_path)
        
        md_content, md_content_ori, layout_pdf_path, zip_path = await smart_model_call(process_file)
        
        # Generate URLs for downloadable files
        layout_pdf_url = f"/static/{os.path.basename(layout_pdf_path)}" if layout_pdf_path and os.path.exists(layout_pdf_path) else None
        zip_url = f"/static/{os.path.basename(zip_path)}" if zip_path and os.path.exists(zip_path) else None
        
        # 保存 Markdown 内容到临时文件，以便后续通过 API 获取
        name = '.'.join(filename.split(".")[:-1])
        md_file_path = os.path.join(temp_dir, f"{name}.md")
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(md_content_ori)
        
        # 根据请求的格式返回不同的响应
        if output_format == "text":
            # 只返回纯文本 Markdown
            return ParseResponse(
                success=True,
                message="PDF parsed successfully (text only)",
                markdown_text=md_content_ori,
                files=[md_file_path] if md_file_path else []
            )
        elif output_format == "zip":
            # 只返回 ZIP 文件
            return ParseResponse(
                success=True,
                message="PDF parsed successfully (ZIP only)",
                markdown_zip_url=zip_url,
                files=[zip_path] if zip_path else []
            )
        else:
            # 返回完整 HTML 格式
            return ParseResponse(
                success=True,
                message="PDF parsed successfully",
                markdown_content=md_content,
                markdown_raw=md_content_ori,
                markdown_text=md_content_ori,
                layout_pdf_url=layout_pdf_url,
                markdown_zip_url=zip_url,
                files=[layout_pdf_path, zip_path] if layout_pdf_path and zip_path else []
            )
        
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        return ParseResponse(
            success=False,
            message=f"Error parsing PDF: {str(e)}"
        )

@app.post("/chat-with-image", response_model=ChatResponse)
async def chat_with_image(
    file: UploadFile = File(...),
    instruction: str = Form(default="Please output the text content from the image.")
):
    """Chat with image using MonkeyOCR"""
    if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the file
        def chat_process():
            if file.filename.lower().endswith('.pdf'):
                images = pdf_to_images(file_path)
                image = images[0]  # Use first page
            else:
                image = Image.open(file_path)
            
            # Use the model for chat inference
            result = monkey_ocr_model.chat_model.chat([image], [instruction])
            return result[0] if result else "No result returned"
        
        result = await smart_model_call(chat_process)
        
        return ChatResponse(
            success=True,
            content=result,
            message="Chat completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error in chat with image: {e}")
        return ChatResponse(
            success=False,
            content="",
            message=f"Error processing image: {str(e)}"
        )

@app.post("/preview-file", response_model=PreviewResponse)
async def preview_file(
    file: UploadFile = File(...),
    page: int = Form(default=1)
):
    """Preview file with pagination support"""
    if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only PDF and image files are supported")
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        def preview_process():
            if file.filename.lower().endswith('.pdf'):
                images = pdf_to_images(file_path)
                total_pages = len(images)
                current_page = min(max(1, page), total_pages)
                image = images[current_page - 1]
            else:
                image = Image.open(file_path)
                total_pages = 1
                current_page = 1
            
            # Save image as temporary file for serving
            preview_filename = f"preview_{timestamp}_{current_page}.png"
            preview_path = os.path.join(temp_dir, preview_filename)
            image.save(preview_path, "PNG")
            
            return total_pages, current_page, f"/static/{preview_filename}"
        
        total_pages, current_page, image_url = await smart_model_call(preview_process)
        
        return PreviewResponse(
            success=True,
            total_pages=total_pages,
            current_page=current_page,
            image_url=image_url
        )
        
    except Exception as e:
        logger.error(f"Error previewing file: {e}")
        return PreviewResponse(
            success=False,
            total_pages=0,
            current_page=0,
            image_url=None
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7861)

# 添加定时清理任务

def cleanup_temp_files():
    """清理超过1天的临时文件(从2天缩短到1天)"""
    try:
        current_time = time.time()
        one_day_ago = current_time - (1 * 24 * 60 * 60)  # 1天前的时间戳
        
        cleaned_files = 0
        cleaned_dirs = 0
        freed_space = 0
        
        # 清理临时目录中的文件
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                # 检查文件修改时间
                if os.path.isfile(file_path):
                    file_modified_time = os.path.getmtime(file_path)
                    if file_modified_time < one_day_ago:
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_files += 1
                            freed_space += file_size
                            logger.debug(f"Cleaned up old temporary file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary file {file_path}: {e}")
                # 也检查目录
                elif os.path.isdir(file_path):
                    dir_modified_time = os.path.getmtime(file_path)
                    if dir_modified_time < one_day_ago:
                        try:
                            dir_size = sum(os.path.getsize(os.path.join(dirpath, f)) 
                                         for dirpath, _, filenames in os.walk(file_path) 
                                         for f in filenames)
                            shutil.rmtree(file_path, ignore_errors=True)
                            cleaned_dirs += 1
                            freed_space += dir_size
                            logger.debug(f"Cleaned up old temporary directory: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove temporary directory {file_path}: {e}")
        
        # 清理其他可能的临时文件和目录
        temp_dirs = [tempfile.gettempdir()]
        for temp_dir_path in temp_dirs:
            if os.path.exists(temp_dir_path):
                for root, dirs, files in os.walk(temp_dir_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 检查是否是 monkeyocr 相关的临时文件
                        if 'monkeyocr' in file.lower() or 'upload' in file.lower() or 'preview' in file.lower() or 'markdown' in file.lower() or 'layout' in file.lower():
                            try:
                                file_modified_time = os.path.getmtime(file_path)
                                if file_modified_time < one_day_ago:
                                    try:
                                        file_size = os.path.getsize(file_path)
                                        os.remove(file_path)
                                        cleaned_files += 1
                                        freed_space += file_size
                                        logger.debug(f"Cleaned up old temporary file: {file_path}")
                                    except Exception as e:
                                        logger.warning(f"Failed to remove temporary file {file_path}: {e}")
                            except Exception:
                                pass
                    
                    # 清理与monkeyocr相关的临时目录
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if 'monkeyocr' in dir_name.lower() or 'markdown' in dir_name.lower() or 'layout' in dir_name.lower():
                            try:
                                # 检查目录中的文件修改时间
                                dir_modified_time = os.path.getmtime(dir_path)
                                if dir_modified_time < one_day_ago:
                                    try:
                                        dir_size = sum(os.path.getsize(os.path.join(dirpath, f)) 
                                                     for dirpath, _, filenames in os.walk(dir_path) 
                                                     for f in filenames)
                                        shutil.rmtree(dir_path, ignore_errors=True)
                                        cleaned_dirs += 1
                                        freed_space += dir_size
                                        logger.debug(f"Removed old temporary directory: {dir_path}")
                                    except Exception as e:
                                        logger.warning(f"Failed to remove temporary directory {dir_path}: {e}")
                            except Exception:
                                pass
        
        # 强制垃圾回收
        gc.collect()
        
        if cleaned_files > 0 or cleaned_dirs > 0:
            freed_mb = freed_space / (1024 * 1024)
            logger.info(f"Cleanup completed: {cleaned_files} files, {cleaned_dirs} directories removed, freed {freed_mb:.2f}MB")
        
        # 记录内存使用情况
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            logger.info(f"Memory usage: RSS={mem_info.rss / (1024**3):.2f}GB, VMS={mem_info.vms / (1024**3):.2f}GB")
        except Exception:
            pass
                                
    except Exception as e:
        logger.error(f"Error during temporary file cleanup: {e}")

def start_cleanup_scheduler():
    """启动定时清理任务"""
    # 每6小时执行一次清理任务(从每天一次改为每6小时一次,更积极)
    schedule.every(6).hours.do(cleanup_temp_files)
    
    # 启动时立即执行一次清理
    threading.Thread(target=cleanup_temp_files, daemon=True).start()
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    
    # 在后台线程中运行调度器
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Temporary file cleanup scheduler started (runs every 6 hours)")
