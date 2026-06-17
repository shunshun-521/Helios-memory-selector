#!/bin/bash

ZIP_FILE="data/mova_audio_infer/t2a_train1_audio.zip"
TARGET_DIR="data/mova_audio_infer"

echo "开始解压文件..."

if [ -f "$ZIP_FILE" ]; then
    # 使用 -o 覆盖已有文件，-d 指定解压目录
    unzip -o "$ZIP_FILE" -d "$TARGET_DIR"
    
    if [ $? -eq 0 ]; then
        echo "解压成功！文件已存放在 $TARGET_DIR 目录下。"
    else
        echo "解压过程中出现错误。"
        exit 1
    fi
else
    echo "错误：找不到文件 $ZIP_FILE，请检查路径是否正确。"
    exit 1
fi
