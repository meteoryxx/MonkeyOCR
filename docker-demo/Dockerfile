# 使用 Debian 12 “bookworm” slim 镜像，避免 testing(trixie) 同步不完整的问题
FROM python:3.10-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1



# 3) Python 依赖使用阿里云 PyPI 镜像
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 4) 其余步骤
COPY . .
EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002 || exit 1
CMD ["python", "app.py"]