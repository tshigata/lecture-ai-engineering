from moviepy.editor import VideoFileClip
import os
from pathlib import Path

def extract_audio_from_video(video_path: str, output_path: str) -> None:
    """
    ビデオファイルから音声を抽出してMP3ファイルとして保存
    Args:
        video_path: 入力ビデオファイルのパス
        output_path: 出力音声ファイルのパス
    """
    try:
        # ビデオファイルを読み込み
        video = VideoFileClip(video_path)
        
        # 音声を抽出
        audio = video.audio
        
        # MP3として保存
        audio.write_audiofile(output_path)
        
        # リソースを解放
        video.close()
        audio.close()
        
        print(f"音声の抽出が完了しました: {output_path}")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

def main():
    # 入力と出力のパスを設定
    base_dir = Path("data")
    video_file = base_dir / "video" / "python_basic_guidance.mp4"
    output_file = base_dir / "audio" / "python_basic_guidance.mp3"
    
    # 出力ディレクトリが存在しない場合は作成
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 音声を抽出
    extract_audio_from_video(str(video_file), str(output_file))

if __name__ == "__main__":
    main() 