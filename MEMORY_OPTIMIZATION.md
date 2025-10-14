# MonkeyOCR 内存优化指南

## 🔍 问题分析

根据实际运行情况，发现内存占用过高的问题：
- **进程1**: 50.1GB虚拟内存, 13.5GB常驻内存, CPU 113.2%
- **进程2**: 33.9GB虚拟内存, 4.1GB常驻内存, CPU 90.1%

### 主要原因

1. **多Worker重复加载模型**: 每个uvicorn worker进程都会加载完整的模型到内存/GPU显存
2. **临时文件未及时清理**: 大量处理产生的临时文件和目录占用磁盘和内存
3. **批处理大小过大**: `batch_size=10` 在高并发时会占用大量内存
4. **无资源限制**: Docker容器和进程没有设置内存上限
5. **垃圾回收不及时**: Python垃圾回收机制在长时间运行时效率降低

## ✅ 已实施的优化措施

### 1. Docker层面优化 (`docker-compose8002.yml`)

```yaml
deploy:
  resources:
    limits:
      memory: 20G  # 限制容器最大内存使用20GB
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["1"]
          capabilities: [gpu]

environment:
  # CUDA内存优化
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  # 限制线程数，减少CPU内存占用
  - OMP_NUM_THREADS=2
  - MKL_NUM_THREADS=2
  # 单Worker模式，避免重复加载模型
  - UVICORN_WORKERS=1
  # 限制并发连接数
  - UVICORN_LIMIT_CONCURRENCY=10
  # 设置keep-alive超时
  - UVICORN_TIMEOUT_KEEP_ALIVE=30
```

**效果**: 
- 限制容器最大内存使用，防止OOM
- 减少CUDA内存碎片化
- 减少CPU线程争用
- **单Worker模式只加载一次模型，节省约13GB内存**

### 2. Uvicorn启动优化 (`entrypoint.sh`)

```bash
# 使用单worker模式，避免多进程重复加载模型
exec uvicorn api.main:app \
    --host ${FASTAPI_HOST:-0.0.0.0} \
    --port ${FASTAPI_PORT:-7861} \
    --workers ${UVICORN_WORKERS:-1} \
    --timeout-keep-alive ${UVICORN_TIMEOUT_KEEP_ALIVE:-30} \
    --limit-concurrency ${UVICORN_LIMIT_CONCURRENCY:-10} \
    --backlog 100
```

**效果**:
- 单worker模式，只加载一次模型
- 限制并发请求数，防止内存爆炸
- 合理的超时设置，释放空闲连接

### 3. 模型配置优化 (`model_configs.yaml`)

```yaml
chat_config:
  batch_size: 4  # 从10降低到4
  queue_config:
    max_batch_size: 128  # 从256降低到128
    max_queue_size: 1000  # 从2000降低到1000
```

**效果**:
- 减少每批次处理的内存占用约60%
- 降低队列内存占用
- 在保持吞吐量的同时减少峰值内存

### 4. 应用层优化 (`main.py`)

#### 4.1 增强的临时文件清理

```python
def cleanup_temp_files():
    """清理超过1天的临时文件(从2天缩短到1天)"""
    # - 清理时间从2天缩短到1天
    # - 记录清理的文件数量和释放的空间
    # - 主动执行垃圾回收
    # - 记录内存使用情况
```

**效果**:
- 更及时清理临时文件，释放磁盘空间
- 减少文件系统缓存占用
- 强制垃圾回收，释放Python对象内存

#### 4.2 更频繁的定时清理

```python
def start_cleanup_scheduler():
    # 每6小时执行一次清理(从每天1次改为每6小时1次)
    schedule.every(6).hours.do(cleanup_temp_files)
    # 启动时立即执行一次清理
    threading.Thread(target=cleanup_temp_files, daemon=True).start()
```

**效果**:
- 更积极的清理策略
- 启动时立即清理历史遗留文件
- 减少临时文件堆积

#### 4.3 内存监控

```python
# 记录内存使用情况
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
logger.info(f"Memory usage: RSS={mem_info.rss / (1024**3):.2f}GB, VMS={mem_info.vms / (1024**3):.2f}GB")
```

**效果**:
- 实时监控内存使用
- 便于定位内存泄漏问题

## 📊 预期优化效果

### 内存占用对比

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| 进程数 | 2个worker | 1个worker | -50% |
| 模型加载次数 | 2次 | 1次 | -50% |
| 常驻内存(RSS) | 17.6GB | ~8-10GB | **-43~57%** |
| 虚拟内存(VMS) | 84GB | ~35-40GB | **-52~58%** |
| 批处理内存峰值 | 高(batch=10) | 降低60%(batch=4) | -60% |
| 临时文件清理 | 2天/次 | 6小时/次 | +8x频率 |

### 并发性能

- **优化前**: 2个worker，但每个占用大量内存，容易OOM
- **优化后**: 1个worker + 异步处理 + 并发限制，稳定可靠
- **吞吐量**: 通过异步批处理保持合理吞吐量
- **响应时间**: 限制并发数后响应更稳定

## 🚀 部署步骤

### 1. 停止当前服务

```bash
cd /www/wwwroot/MonkeyOCR/docker-api
docker-compose -f docker-compose8002.yml down
```

### 2. 更新配置文件

确保以下文件已更新:
- ✅ `docker-compose8002.yml` - Docker资源限制和环境变量
- ✅ `entrypoint.sh` - Uvicorn启动参数
- ✅ `main.py` - 增强的清理和监控
- ✅ `model_configs.yaml` - 模型批处理配置

### 3. 安装依赖(如需要)

```bash
# 在容器中安装psutil用于内存监控
pip install psutil schedule
```

### 4. 启动优化后的服务

```bash
docker-compose -f docker-compose8002.yml up -d
```

### 5. 监控内存使用

```bash
# 查看容器资源使用
docker stats ocrdocker-gpu1

# 查看日志中的内存报告
docker logs -f ocrdocker-gpu1 | grep "Memory usage"

# 查看GPU显存使用
nvidia-smi
```

## 📋 监控检查清单

部署后需要监控以下指标：

### 每小时检查
- [ ] 容器内存使用 < 18GB (留2GB缓冲)
- [ ] GPU显存使用稳定
- [ ] 进程只有1个uvicorn worker

### 每天检查
- [ ] 临时文件目录大小 < 10GB
- [ ] 清理日志显示正常运行
- [ ] 无OOM错误

### 每周检查
- [ ] 服务响应时间正常
- [ ] 无内存泄漏迹象
- [ ] 错误率在可接受范围

## 🔧 进一步优化建议

### 1. 如果内存仍然偏高

```yaml
# docker-compose8002.yml
environment:
  - UVICORN_LIMIT_CONCURRENCY=5  # 进一步限制并发
```

```yaml
# model_configs.yaml
chat_config:
  batch_size: 2  # 进一步减小批处理大小
```

### 2. 启用模型量化(如果支持)

```yaml
# model_configs.yaml
chat_config:
  quantization: int4  # 或 int8
```

**效果**: 可以减少模型显存占用50-75%

### 3. 使用模型缓存服务

考虑使用Redis或专门的模型服务来共享模型实例，避免重复加载。

### 4. 添加请求队列

对于高并发场景，添加外部消息队列(如RabbitMQ)来缓冲请求。

## 🐛 故障排查

### 问题1: 容器启动失败，提示内存不足

**解决方案**:
```yaml
# 临时增加内存限制
deploy:
  resources:
    limits:
      memory: 24G  # 从20G增加到24G
```

### 问题2: 服务响应缓慢

**原因**: 并发限制过低
**解决方案**:
```bash
# 适当增加并发数
UVICORN_LIMIT_CONCURRENCY=15
```

### 问题3: 临时文件清理失败

**检查**:
```bash
# 进入容器检查临时目录权限
docker exec -it ocrdocker-gpu1 ls -la /app/tmp
```

### 问题4: 内存持续增长

**排查步骤**:
1. 检查是否有长时间运行的请求
2. 查看临时文件是否正常清理
3. 检查是否有内存泄漏(使用memory_profiler)

## 📈 性能基准测试

优化后建议进行以下测试：

```bash
# 1. 单请求测试
curl -X POST -F "file=@test.pdf" http://localhost:8002/parse

# 2. 并发测试(使用ab或wrk)
ab -n 20 -c 5 -p test.pdf -T application/pdf http://localhost:8002/parse

# 3. 持续压力测试
# 运行24小时，监控内存是否稳定
```

## 📝 更新日志

- **2025-10-14**: 初始优化版本
  - 添加Docker内存限制
  - 优化为单Worker模式
  - 降低批处理大小
  - 增强临时文件清理
  - 添加内存监控

---

## ⚠️ 重要提醒

1. **单Worker模式**: 虽然减少了内存占用，但也降低了最大并发能力。如果需要更高并发，考虑使用负载均衡部署多个独立容器。

2. **GPU显存**: 这些优化主要针对系统内存。GPU显存主要由模型本身决定，如需优化需要使用模型量化或更小的模型。

3. **定期重启**: 尽管已添加垃圾回收，仍建议每周重启一次容器以彻底释放内存。

4. **监控告警**: 建议配置Prometheus + Grafana监控内存使用，设置告警阈值。
