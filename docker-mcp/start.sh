#!/bin/bash

echo "Starting MonkeyOCR MCP Stack..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start services
echo "Building and starting services..."
docker-compose up -d --build

echo "Services started!"
echo "- MonkeyOCR API: http://localhost:8000"
echo "- MCP Server: stdio mode (connect via MCP client)"
echo "- Health check: http://localhost:8000/health"

echo "To view logs: docker-compose logs -f"