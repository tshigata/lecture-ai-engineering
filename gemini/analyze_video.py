import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import mimetypes
import re

# 環境変数の読み込み
load_dotenv()

# APIキーの設定
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")

# Gemini APIの設定
genai.configure(api_key=GOOGLE_API_KEY)

def remove_fillers(text: str) -> str:
    """
    テキストからフィラーや不要な表現を除去し、自然な文章に整形する
    Args:
        text: 処理対象のテキスト
    Returns:
        フィラーを除去し、自然な文章に整形したテキスト
    """
    # フィラーと不要な表現のパターン
    patterns = [
        # 一般的なフィラー
        r'えー[と]?', r'あの[ー]?', r'その[ー]?', r'えっと', r'まあ', r'なんか', r'っていうか',
        # 繰り返し表現
        r'([、。])\1+',  # 連続する句読点
        r'([ぁ-んァ-ン])\1{2,}',  # 3回以上続くひらがな/カタカナ
        # 不要な接続詞
        r'で[、。]', r'が[、。]', r'けど[、。]', r'から[、。]',
        # 文末の不要な表現
        r'[、。]です[、。]', r'[、。]ます[、。]', r'[、。]でした[、。]', r'[、。]ました[、。]',
        # 余分な空白
        r'\s+',
        # 文頭の不要な表現
        r'^[、。]', r'^[ぁ-んァ-ン]{1,3}[、。]'
    ]
    
    # パターンの適用
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    # 文の整形
    # 1. 文末の句読点を統一
    text = re.sub(r'[、。]+$', '。', text)
    # 2. 文頭の不要な文字を削除
    text = re.sub(r'^[、。\s]+', '', text)
    # 3. 連続する句読点を1つに
    text = re.sub(r'[、。]+', '。', text)
    # 4. 文と文の間の空白を適切に
    text = re.sub(r'。\s*。', '。', text)
    # 5. 最後の文末に句点がない場合は追加
    if text and not text[-1] in ['。', '！', '？']:
        text += '。'
    
    return text.strip()

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

        # 動画ファイルのMIMEタイプを取得
        mime_type, _ = mimetypes.guess_type(video_path)
        if not mime_type:
            mime_type = "video/mp4"  # デフォルトのMIMEタイプ

        # 動画ファイルをバイナリで読み込み
        with open(video_file, "rb") as f:
            video_data = f.read()

        # Gemini 2.0 Flashモデルの設定
        model = genai.GenerativeModel('gemini-2.0-flash')

        # プロンプトの設定
        prompt = """
        この動画は教育用のスライドプレゼンテーションです。
        以下の点に注意して解析してください：
        1. スライドに表示されている内容を正確に読み取る
        2. 話者の説明を音声から文字起こしする
        3. スライドの内容と音声の説明を組み合わせて、より正確な文字起こしを行う
        4. 各スライドごとに区切って出力する
        5. タイムスタンプを付けて、どの部分の説明かを明確にする
        6. 文字起こしの際、言い間違いやフィラー（例：「えー」「あのー」「そのー」「えっと」など）、説明の本質に関係ないワードは除去し、分かりやすく簡潔に整形する

        出力形式：
        [スライド1 - 00:00:00]
        スライドの内容：
        [スライドの内容を箇条書きで]
        
        説明：
        [フィラーや言い間違いを除去し、分かりやすく整形した音声の文字起こし]
        """

        # 動画データをBlobとして作成
        video_blob = {
            "mime_type": mime_type,
            "data": video_data
        }

        # 動画の解析
        response = model.generate_content([prompt, video_blob])
        
        # 結果の後処理
        result = response.text
        slides = result.split('---')
        processed_slides = []
        
        for slide in slides:
            if not slide.strip():
                continue
            
            # スライドの説明部分を抽出
            explanation_match = re.search(r'説明：(.*?)(?=\n\n|$)', slide, re.DOTALL)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                # フィラー除去を適用
                cleaned_explanation = remove_fillers(explanation)
                # 元の説明を置き換え
                slide = slide.replace(explanation_match.group(0), f'説明：{cleaned_explanation}')
            
            processed_slides.append(slide)
        
        # 処理済みのスライドを結合
        final_result = '\n---\n'.join(processed_slides)
        
        # 結果を保存
        output_path = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'data', 'output', 'video_analysis.txt')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_result)
        
        print(f'動画の解析が完了しました: {output_path}')
        
        return final_result

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return None

def main():
    # 入力動画ファイルのパス
    video_path = "../data/video/python_basic_guidance.mp4"
    
    # 出力テキストファイルのパス
    output_path = "../data/output/video_analysis.txt"
    
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