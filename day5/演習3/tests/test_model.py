import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
import json
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成（ハイパーパラメータを最適化）
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),

        ]
    )

    # 交差検証による性能評価
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"交差検証スコア: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, X_test, y_test


def test_model_exists():
    """モデルファイルが存在するか確認"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")
    assert os.path.exists(MODEL_PATH), "モデルファイルが存在しません"


def save_metrics(metrics):
    """モデルの性能指標を保存"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 既存のメトリクスを読み込む
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            history = json.load(f)
    else:
        history = []

    # 新しいメトリクスを追加
    metrics["timestamp"] = datetime.now().isoformat()
    history.append(metrics)

    # メトリクスを保存
    with open(METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)


def load_previous_metrics():
    """過去のメトリクスを読み込む"""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            history = json.load(f)
        return history[-1] if history else None
    return None


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 推論時間の計測
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    # 精度計算
    accuracy = accuracy_score(y_test, y_pred)

    # メトリクスの保存
    metrics = {
        "accuracy": accuracy,
        "inference_time": inference_time,
    }
    save_metrics(metrics)

    # 過去のメトリクスと比較
    previous_metrics = load_previous_metrics()
    if previous_metrics:
        logger.info("\n=== 過去のモデルとの比較 ===")
        logger.info(
            f"accuracy: {accuracy:.4f} (前回: {previous_metrics['accuracy']:.4f}, "
            f"差分: {accuracy - previous_metrics['accuracy']:+.4f})"
        )
        logger.info(
            f"inference_time: {inference_time:.4f}秒 (前回: {previous_metrics['inference_time']:.4f}秒, "
            f"差分: {inference_time - previous_metrics['inference_time']:+.4f}秒)"
        )
    else:
        logger.info("\n=== 現在のモデル性能 ===")
        logger.info(f"accuracy: {accuracy:.4f}")
        logger.info(f"inference_time: {inference_time:.4f}秒")

    # 性能基準の検証
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy}"
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3711
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(
        predictions1, predictions2
    ), "モデルの予測結果に再現性がありません"
