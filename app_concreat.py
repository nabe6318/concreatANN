# app.py
import io
import os
import time
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

st.set_page_config(page_title="コンクリート強度 予測ANN", layout="wide")

# -----------------------
# ユーティリティ
# -----------------------
@st.cache_data
def load_csv(file, encoding, sep):
    return pd.read_csv(file, encoding=encoding, sep=sep)

def available_numeric_columns(df: pd.DataFrame):
    # 数値に強制変換してもNaN少なめの列を優先
    num_cols = []
    for c in df.columns:
        try:
            pd.to_numeric(df[c], errors="raise")
            num_cols.append(c)
        except Exception:
            pass
    return num_cols

def parse_hidden_layers(text: str):
    # "128,128,64" -> [128,128,64]
    try:
        units = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
        return [u for u in units if u > 0]
    except Exception:
        return [128, 128, 64]

def build_model(input_dim: int, hidden_units, dropout, l2reg, lr):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for u in hidden_units:
        x = layers.Dense(
            u, activation="relu",
            kernel_regularizer=regularizers.l2(l2reg) if l2reg > 0 else None
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse"), "mae"]
    )
    return model

def metrics_table(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return pd.DataFrame(
        {"RMSE":[rmse], "MAE":[mae], "R^2":[r2]}
    )

def plot_true_vs_pred(y_true, y_pred, title="True vs Pred"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    st.pyplot(fig)

# -----------------------
# サイドバー：データ入力
# -----------------------
st.sidebar.header("1) データの読み込み")
uploaded = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])
encoding = st.sidebar.selectbox("文字コード", ["utf-8", "cp932", "shift_jis"], index=0)
sep = st.sidebar.selectbox("区切り", [",", ";", "\t", "|"], index=0)

df = None
if uploaded is not None:
    try:
        df = load_csv(uploaded, encoding=encoding, sep=sep)
    except Exception as e:
        st.sidebar.error(f"読み込みエラー: {e}")

if df is None:
    st.info("左のサイドバーからCSVをアップロードしてください。")
    st.stop()

st.write("### データプレビュー")
st.dataframe(df.head(20), use_container_width=True)

# 列選択（数値列を候補に）
num_cols = available_numeric_columns(df)
if len(num_cols) == 0:
    st.error("数値列が見つかりません。数値列を含むCSVをアップロードしてください。")
    st.stop()

st.sidebar.header("2) 列の指定")
default_target = "strength" if "strength" in df.columns else num_cols[-1]
target_col = st.sidebar.selectbox("目的変数（予測したい列）", options=num_cols, index=num_cols.index(default_target) if default_target in num_cols else 0)

default_features = [c for c in ["cement","slag","flyash","water","superplasticizer","coarseagg","fineagg","age"] if c in num_cols and c != target_col]
if not default_features:
    # 目的変数以外の上位8列を初期値に
    default_features = [c for c in num_cols if c != target_col][:8]

feature_cols = st.sidebar.multiselect(
    "説明変数（複数選択）", options=[c for c in num_cols if c != target_col],
    default=default_features
)

if len(feature_cols) == 0:
    st.warning("少なくとも1つの説明変数を選んでください。")
    st.stop()

# -----------------------
# サイドバー：前処理・分割
# -----------------------
st.sidebar.header("3) 前処理・分割")
test_size = st.sidebar.slider("テストデータ割合", 0.05, 0.5, 0.2, 0.05)
do_log_age = False
if "age" in feature_cols:
    do_log_age = st.sidebar.checkbox("ageをlog1p変換（任意）", value=False)

# -----------------------
# サイドバー：モデル設定
# -----------------------
st.sidebar.header("4) モデル設定（Keras）")
hidden_text = st.sidebar.text_input("隠れ層（カンマ区切り）", value="128,128,64", help="例: 256,128,64")
dropout = st.sidebar.slider("Dropout", 0.0, 0.8, 0.1, 0.05)
l2reg = st.sidebar.number_input("L2正則化（λ）", min_value=0.0, value=0.0, step=0.0001, format="%.4f")
lr = st.sidebar.number_input("学習率（Adam）", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")
epochs = st.sidebar.number_input("エポック数", min_value=10, max_value=5000, value=1000, step=10)
batch_size = st.sidebar.selectbox("バッチサイズ", [16, 32, 64, 128], index=1)
early_patience = st.sidebar.slider("EarlyStopping patience", 10, 200, 50, 5)

train_button = st.sidebar.button("🔁 学習を実行")

# -----------------------
# データ前処理
# -----------------------
data = df.copy()

# 数値化を保証（文字列の数値も取り込む）
for c in feature_cols + [target_col]:
    data[c] = pd.to_numeric(data[c], errors="coerce")

data = data.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

# 任意: ageのlog1p変換
effective_features = feature_cols.copy()
if do_log_age and "age" in effective_features:
    data["age_log"] = np.log1p(data["age"].astype(float))
    effective_features = [("age_log" if c == "age" else c) for c in effective_features]

X = data[effective_features].values.astype("float32")
y = data[target_col].values.astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

st.write("**説明変数**:", effective_features)
st.write("**目的変数**:", target_col)
st.write(f"学習用: {X_train_sc.shape}, テスト用: {X_test_sc.shape}")

# -----------------------
# 学習
# -----------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "history" not in st.session_state:
    st.session_state.history = None
if "features" not in st.session_state:
    st.session_state.features = effective_features
if "target" not in st.session_state:
    st.session_state.target = target_col

if train_button:
    hidden_units = parse_hidden_layers(hidden_text)
    model = build_model(
        input_dim=X_train_sc.shape[1],
        hidden_units=hidden_units,
        dropout=dropout,
        l2reg=l2reg,
        lr=lr
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_rmse", patience=int(early_patience), restore_best_weights=True
        )
    ]

    with st.spinner("学習中..."):
        history = model.fit(
            X_train_sc, y_train,
            validation_split=0.2,
            epochs=int(epochs),
            batch_size=int(batch_size),
            callbacks=callbacks,
            verbose=0
        )

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.history = history.history
    st.session_state.features = effective_features
    st.session_state.target = target_col

    st.success("学習が完了しました。下に評価結果を表示します。")

# -----------------------
# 評価と可視化
# -----------------------
if st.session_state.model is not None:
    model = st.session_state.model
    scaler = st.session_state.scaler

    y_pred_test = model.predict(X_test_sc, verbose=0).reshape(-1)
    y_pred_train = model.predict(X_train_sc, verbose=0).reshape(-1)

    st.subheader("評価指標")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**テストデータ**")
        st.dataframe(metrics_table(y_test, y_pred_test), use_container_width=True)
    with col2:
        st.write("**学習データ**")
        st.dataframe(metrics_table(y_train, y_pred_train), use_container_width=True)

    st.subheader("予測の可視化")
    col3, col4 = st.columns(2)
    with col3:
        plot_true_vs_pred(y_test, y_pred_test, title="Test: True vs Pred")
    with col4:
        plot_true_vs_pred(y_train, y_pred_train, title="Train: True vs Pred")

    if st.session_state.history:
        st.subheader("学習曲線")
        hist = st.session_state.history
        fig, ax = plt.subplots()
        ax.plot(hist["rmse"], label="rmse")
        ax.plot(hist["val_rmse"], label="val_rmse")
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE")
        ax.legend()
        st.pyplot(fig)

    # -------------------
    # モデルの保存とダウンロード
    # -------------------
    st.subheader("モデルの保存")
    c1, c2 = st.columns(2)
    with c1:
        with tempfile.TemporaryDirectory() as tmpd:
            model_path = os.path.join(tmpd, "best_concrete_ann.keras")
            st.session_state.model.save(model_path, include_optimizer=True)
            with open(model_path, "rb") as f:
                st.download_button(
                    label="学習済みモデル（.keras）をダウンロード",
                    data=f.read(),
                    file_name="best_concrete_ann.keras",
                    mime="application/octet-stream"
                )
    with c2:
        # scaler をnpzで保存（平均・分散・scaleなど）
        buf = io.BytesIO()
        np.savez(
            buf,
            mean_=scaler.mean_,
            scale_=scaler.scale_,
            var_=scaler.var_,
            n_features_in_=np.array([scaler.n_features_in_]),
        )
        st.download_button(
            label="標準化スケーラ（.npz）をダウンロード",
            data=buf.getvalue(),
            file_name="scaler_concrete.npz",
            mime="application/octet-stream"
        )

# -----------------------
# 単発推論（フォーム）
# -----------------------
st.subheader("単発推論（説明変数を入力）")
if st.session_state.model is None:
    st.info("先に学習を実行してください。")
else:
    model = st.session_state.model
    scaler = st.session_state.scaler
    feat_names = st.session_state.features

    cols = st.columns(min(4, len(feat_names)))
    values = []
    for i, name in enumerate(feat_names):
        col = cols[i % len(cols)]
        # 初期値は学習データの中央値
        default_val = float(np.median(data[name]))
        val = col.number_input(name, value=default_val)
        values.append(val)

    if st.button("この入力で予測する"):
        x = np.array(values, dtype="float32").reshape(1, -1)
        x_sc = scaler.transform(x)
        yhat = model.predict(x_sc, verbose=0).reshape(-1)[0]
        st.success(f"予測強度: **{yhat:.3f}**")

st.caption("© Streamlit + TensorFlow/Keras | CSVの列名はUIから自由に指定できます。")
