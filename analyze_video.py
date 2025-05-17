import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# 環境変数の読み込み
load_dotenv()

# APIキーの設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")

# Gemini APIの設定
genai.configure(api_key=GOOGLE_API_KEY)

def analyze_video(video_path: str) -> str:
    """
    動画ファイルを解析し、スライドの内容と音声を理解して文字起こしを行う
    Args:
        video_path: 動画ファイルのパス
    Returns:
        解析結果のテキスト
    """
    try:
        # 動画ファイルの読み込み
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        # Gemini Pro Visionモデルの設定
        model = genai.GenerativeModel('gemini-pro-vision')

        # プロンプトの設定
        prompt = """
        この動画は教育用のスライドプレゼンテーションです。
        以下の点に注意して解析してください：
        1. スライドに表示されている内容を正確に読み取る
        2. 話者の説明を音声から文字起こしする
        3. スライドの内容と音声の説明を組み合わせて、より正確な文字起こしを行う
        4. 各スライドごとに区切って出力する
        5. タイムスタンプを付けて、どの部分の説明かを明確にする

        出力形式：
        [スライド1 - 00:00:00]
        スライドの内容：
        [スライドの内容を箇条書きで]
        
        説明：
        [音声の文字起こし]
        """

        # 動画の解析
        response = model.generate_content([prompt, video_file])
        
        return response.text

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return None

def main():
    # 入力動画ファイルのパス
    video_path = "data/video/python_basic_guidance.mp4"
    
    # 出力テキストファイルのパス
    output_path = "data/output/video_analysis.txt"
    
    # 動画の解析
    result = analyze_video(video_path)
    
    if result:
        # 出力ディレクトリの作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 結果の保存
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        
        print(f"動画の解析が完了しました: {output_path}")

if __name__ == "__main__":
    main() 