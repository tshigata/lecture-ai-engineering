import whisper
import os
from pathlib import Path
import re

def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def transcribe_audio(audio_path: str, output_path: str) -> None:
    """
    音声ファイルから文字起こしを行い、タイムコード付きでテキストファイルとして保存
    Args:
        audio_path: 入力音声ファイルのパス
        output_path: 出力テキストファイルのパス
    """
    try:
        # Whisperモデルの読み込み
        model = whisper.load_model("base")
        
        # 音声ファイルの文字起こし（セグメント付き）
        result = model.transcribe(audio_path, verbose=True)
        
        # 出力ディレクトリの作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # タイムコード付きで書き出し
        with open(output_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start = format_timestamp(segment["start"])
                end = format_timestamp(segment["end"])
                text = segment["text"].strip()
                f.write(f"[{start} --> {end}] {text}\n")
        
        print(f"タイムコード付き文字起こしが完了しました: {output_path}")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

def main():
    # 入力音声ファイルのパス
    audio_path = "data/audio/python_basic_guidance.mp3"
    
    # 出力テキストファイルのパス
    output_path = "data/output/python_basic_guidance.txt"
    
    # 文字起こしの実行
    transcribe_audio(audio_path, output_path)

if __name__ == "__main__":
    main() 