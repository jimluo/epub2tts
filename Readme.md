# 电子书转有声书
用于epub格式的电子书转换有声书，方便自己听读

## 功能需求
- EPUB 解析
- 文本清洗与规范化
- 分段策略（长文本稳定性）
- TTS 引擎适配层
- Kokoros 引擎
- 音频拼接与编码
- 中间产物：统一 WAV，后续ogg编码

## 决策摘要
- 交付形态：仅 CLI；本地离线；CPU 推理
- 引擎：可插拔 TTS 引擎接口； Kokoros/fish tts
- 文本：针对中文电子书优化的清洗与分段策略
- 音频：中间产物统一 WAV/ Ogg，不依赖 ffmpeg。

## 使用方式
```
>epub2tts convert test.epub --engine kokoro --out out  --resume
[1/16   6%] 合成 Cover
[1/16   6%] 完成 Cover 0.0s -> ./out/test/chapters/001_Cover.wav
[2/16  13%] 合成 id5
[2/16  13%] 完成 id5 7.0s -> ./out/test/chapters/002_id5.wav
[3/16  19%] 合成 id6
```
