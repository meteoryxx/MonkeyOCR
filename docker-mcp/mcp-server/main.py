#!/usr/bin/env python3
"""
MonkeyOCR MCP Server
Model Context Protocol server for MonkeyOCR API
"""

import os
import io
import base64
import json
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, ImageContent, Embedding

# Initialize MCP server
server = Server("monkeyocr-mcp")

# Configuration
API_BASE_URL = os.getenv("MONKEYOCR_API_URL", "http://localhost:8000")

async def call_monkeyocr_api(endpoint: str, file_data: bytes, filename: str) -> Dict[str, Any]:
    """Call MonkeyOCR API with file upload"""
    async with httpx.AsyncClient() as client:
        files = {"file": (filename, file_data)}
        response = await client.post(f"{API_BASE_URL}{endpoint}", files=files)
        response.raise_for_status()
        return response.json()

@server.list_tools()
async def list_tools() -> List[Dict[str, Any]]:
    """List available OCR tools"""
    return [
        {
            "name": "extract_text",
            "description": "Extract text from image or PDF file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Base64 encoded file data or file path"
                    }
                },
                "required": ["file"]
            }
        },
        {
            "name": "extract_formula",
            "description": "Extract mathematical formulas from image or PDF",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Base64 encoded file data or file path"
                    }
                },
                "required": ["file"]
            }
        },
        {
            "name": "extract_table",
            "description": "Extract tables from image or PDF",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Base64 encoded file data or file path"
                    }
                },
                "required": ["file"]
            }
        },
        {
            "name": "parse_document",
            "description": "Parse complete document (PDF or image) with full analysis",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Base64 encoded file data or file path"
                    }
                },
                "required": ["file"]
            }
        },
        {
            "name": "parse_document_split",
            "description": "Parse document and split result by pages",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Base64 encoded file data or file path"
                    }
                },
                "required": ["file"]
            }
        },
        {
            "name": "parse_pdf",
            "description": "Parse PDF to Markdown with full processing including layout analysis",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Base64 encoded file data or file path"
                    }
                },
                "required": ["file"]
            }
        },
        {
            "name": "chat_with_image", 
            "description": "Chat with image using custom instructions",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Base64 encoded file data or file path"
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Custom instruction for the chat",
                        "default": "Please output the text content from the image."
                    }
                },
                "required": ["file"]
            }
        },
        {
            "name": "preview_file",
            "description": "Preview file with pagination support",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Base64 encoded file data or file path"
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number to preview (default: 1)",
                        "default": 1
                    }
                },
                "required": ["file"]
            }
        }
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute OCR tool"""
    
    # Map tool names to API endpoints
    endpoint_map = {
        "extract_text": "/ocr/text",
        "extract_formula": "/ocr/formula", 
        "extract_table": "/ocr/table",
        "parse_document": "/parse",
        "parse_document_split": "/parse/split",
        "parse_pdf": "/parse-pdf",
        "chat_with_image": "/chat-with-image",
        "preview_file": "/preview-file"
    }
    
    if name not in endpoint_map:
        raise ValueError(f"Unknown tool: {name}")
    
    # Get file data from arguments
    file_input = arguments.get("file")
    if not file_input:
        raise ValueError("File parameter is required")
    
    # Handle base64 encoded file or file path
    if file_input.startswith("data:"):
        # Base64 data URI
        header, data = file_input.split(",", 1)
        file_data = base64.b64decode(data)
        filename = "input_file"
    elif os.path.isfile(file_input):
        # File path
        with open(file_input, "rb") as f:
            file_data = f.read()
        filename = os.path.basename(file_input)
    else:
        # Assume it's base64 without data URI
        try:
            file_data = base64.b64decode(file_input)
            filename = "input_file"
        except:
            raise ValueError("Invalid file input. Provide base64 data or file path")
    
    # Call MonkeyOCR API
    endpoint = endpoint_map[name]
    if params:
        # For tools that require form data
        data = params
        files = {"file": (filename, file_data)}
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{API_BASE_URL}{endpoint}", data=data, files=files)
    else:
        # For simple file uploads
        result = await call_monkeyocr_api(endpoint, file_data, filename)
    
    # Format response
    if result.get("success"):
        content = result.get("content", "")
        if not content and result.get("files"):
            content = f"Processing completed. Files: {', '.join(result['files'])}"
        
        return [TextContent(type="text", text=content)]
    else:
        error_msg = result.get("message", "Unknown error")
        return [TextContent(type="text", text=f"Error: {error_msg}")]

@server.list_resources()
async def list_resources() -> List[Dict[str, Any]]:
    """List available resources"""
    return []

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content"""
    raise ValueError("Resources not supported")

async def main():
    """Main entry point"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())