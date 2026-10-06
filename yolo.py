from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException
from pathlib import Path
import logging
import uuid

logger = logging.getLogger(__name__)

# 图片文件上传文件夹
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

async def upload_file(file: UploadFile = File(...)):
    """接收前端上传的图片，保存到本地，返回路径"""
    original_name = file.filename or ""
    logger.info(
        "收到图片文件: filename=%s content_type=%s",
        original_name,
        file.content_type,
    )

    # 1、生成唯一文件名，防止冲突
    ext = Path(original_name).suffix.lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        logger.warning("图片格式不支持: filename=%s extension=%s", original_name, ext)
        raise HTTPException(status_code=400, detail="不支持的图片格式")
    # 生成随即名，减少重名冲突
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename

    # 2、写入磁盘
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    logger.info(
        "图片写入完成: bytes=%d path=%s",
        len(content),
        save_path.resolve(),
    )

    # 3、返回绝对路径(因为YOLO模型需要绝对路径)
    response = {
        "image_path": str(save_path.resolve()),
        "file_name": original_name,
    }
    logger.info("上传处理完成，返回 image_path=%s", response["image_path"])
    return response
