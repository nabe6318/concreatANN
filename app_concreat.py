# app_mlp.py
import io
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

st.set_page_config(page_title="コンクリート強度 予測（MLPRegressor）", layout="wide")

# -----------------------
# ユーティリティ
# -----------------------
@st.cache_data
def load_csv(file, encoding, sep):
    return pd.read_csv(file, encoding=encoding, sep=sep)

def available_numeric_columns(df: pd.DataFrame):
    # 数値変換できる列のみ抽出
    cols = []
    for c in df.columns:
        try:
            pd.to_numeric(df[c], errors="raise")
            cols.append(c)
        except Exception:
            pass
    return cols

def metrics_table(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return pd.DataFrame({"RMSE":[rmse], "MAE":[mae], "R^2":[r2]})

def plot_true_vs_pred(y_true, y_pred, title):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
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

if uploaded is None:
    st.info("左サイドバーからCSVをアップロードしてください。")
    st.stop()

try:
    df = load_csv(uploaded, encoding=encoding, sep=sep)
except Exception as e:
    st.error(f"読み込みエラー: {e}")
    st.stop()

st.write("### データプレビュー")
st.dataframe(df.head(20), use_container_width=True)

# 列選択
num_cols = available_numeric_columns(df)
if not num_cols:
    st.error("数値列が見つかりません。数値列を含むCSVをアップロードしてください。")
    st.stop()

st.sidebar.header("2) 列の指定")
default_target = "strength" if "strength" in df.columns else num_cols[-1]
target_col = st.sidebar.selectbox(
    "目的変数（予測したい列）",
    options=num_cols,
    index=(num_cols.index(default_target) if default_target in num_cols else 0)
)

default_features = [c for c in ["cement","slag","flyash","water","superplasticizer","coarseagg","fineagg","age"]
                    if c in num_cols and c != target_col]
if not default_features:
    default_features = [c for c in num_cols if c != target_col][:8]

feature_cols = st.sidebar.multiselect(
    "説明変数（複数選択）",
    options=[c for c in num_cols if c != target_col],
    default=default_features
)

if len(feature_cols) == 0:
    st.warning("少なくとも1つの説明変数を選んでください。")
    st.stop()

# -----------------------
# 前処理・分割・モデル設定
# -----------------------
st.sidebar.header("3) 前処理・分割")
test_size = st.sidebar.slider("テストデータ割合", 0.05, 0.5, 0.2, 0.05)
do_log_age = "age" in feature_cols and st.sidebar.checkbox("ageをlog1p変換", value=False)

st.sidebar.header("4) MLP ハイパーパラメータ")
hidden_text = st.sidebar.text_input("隠れ層（カンマ区切り）", value="128,128,64")  # 例: "256,128,64"
def parse_layers(s):
    try:
        return tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
    except:
        return (128,128,64)

learning_rate_init = st.sidebar.number_input("学習率", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")
max_iter = st.sidebar.number_input("max_iter（学習上限反復）", min_value=50, max_value=20000, value=5000, step=50)
early_stopping = st.sidebar.checkbox("EarlyStopping を使う", value=True)
n_iter_no_change = st.sidebar.number_input("n_iter_no_change", min_value=5, max_value=500, value=50, step=5)
validation_fraction = st.sidebar.slider("validation_fraction", 0.05, 0.4, 0.2, 0.05)
alpha = st.sidebar.number_input("L2正則化 (alpha)", min_value=0.0, value=0.0001, step=0.0001, format="%.4f")
batch_size = st.sidebar.selectbox("batch_size", [16, 32, 64, 128, "auto"], index=1)

train_button = st.sidebar.button("🔁 学習を実行")

# -----------------------
# データ整形
# -----------------------
data = df.copy()
for c in feature_cols + [target_col]:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data = data.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

effective_features = feature_cols.copy()
if do_log_age and "age" in effective_features:
    data["age_log"] = np.log1p(data["age"].astype(float))
    effective_features = [("age_log" if c == "age" else c) for c in effective_features]

X = data[effective_features].values.astype("float32")
y = data[target_col].values.astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

st.write("**説明変数**:", effective_features)
st.write("**目的変数**:", target_col)
st.write(f"学習用: {X_train.shape}, テスト用: {X_test.shape}")

# -----------------------
# 学習
# -----------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "fit_info" not in st.session_state:
    st.session_state.fit_info = {}

if train_button:
    hidden_layer_sizes = parse_layers(hidden_text)

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=float(learning_rate_init),
            max_iter=int(max_iter),
            random_state=42,
            early_stopping=bool(early_stopping),
            n_iter_no_change=int(n_iter_no_change),
            validation_fraction=float(validation_fraction),
            alpha=float(alpha),
            batch_size=(None if batch_size == "auto" else int(batch_size)),
            verbose=False
        ))
    ])

    with st.spinner("学習中..."):
        pipe.fit(X_train, y_train)

    st.session_state.pipeline = pipe
    # loss_curve_ などを保存（可視化用）
    mlp = pipe.named_steps["mlp"]
    st.session_state.fit_info = {
        "loss_curve": getattr(mlp, "loss_curve_", None),
        "validation_scores": getattr(mlp, "validation_scores_", None)
    }

    st.success("学習が完了しました。下に評価結果を表示します。")

# -----------------------
# 評価
# -----------------------
if st.session_state.pipeline is not None:
    pipe = st.session_state.pipeline

    y_pred_test = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    st.subheader("評価指標")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**テストデータ**")
        st.dataframe(metrics_table(y_test, y_pred_test), use_container_width=True)
    with c2:
        st.write("**学習データ**")
        st.dataframe(metrics_table(y_train, y_pred_train), use_container_width=True)

    st.subheader("予測の可視化")
    c3, c4 = st.columns(2)
    with c3:
        plot_true_vs_pred(y_test, y_pred_test, title="Test: True vs Pred")
    with c4:
        plot_true_vs_pred(y_train, y_pred_train, title="Train: True vs Pred")

    # 学習曲線
    info = st.session_state.fit_info
    if info.get("loss_curve") is not None:
        st.subheader("学習曲線（loss）")
        fig, ax = plt.subplots()
        ax.plot(info["loss_curve"])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        st.pyplot(fig)

    if info.get("validation_scores") is not None:
        st.subheader("検証スコア（EarlyStopping有効時）")
        fig, ax = plt.subplots()
        ax.plot(info["validation_scores"])
        ax.set_xlabel("Checkpoints")
        ax.set_ylabel("Score (R^2)")
        st.pyplot(fig)

    # モデル保存（ダウンロード）
    st.subheader("モデルの保存")
    with tempfile.TemporaryDirectory() as tmpd:
        model_path = os.path.join(tmpd, "concrete_mlp_pipeline.joblib")
        joblib.dump(pipe, model_path)
        with open(model_path, "rb") as f:
            st.download_button(
                label="学習済みモデル（.joblib）をダウンロード",
                data=f.read(),
                file_name="concrete_mlp_pipeline.joblib",
                mime="application/octet-stream"
            )

# -----------------------
# 単発推論
# -----------------------
st.subheader("単発推論（説明変数を入力）")
if st.session_state.pipeline is None:
    st.info("先に学習を実行してください。")
else:
    pipe = st.session_state.pipeline
    feat_names = effective_features
    cols = st.columns(min(4, len(feat_names)))
    values = []
    for i, name in enumerate(feat_names):
        col = cols[i % len(cols)]
        default_val = float(np.median(data[name])) if len(data) > 0 else 0.0
        values.append(col.number_input(name, value=default_val))
    if st.button("この入力で予測する"):
        x = np.array(values, dtype="float32").reshape(1, -1)
        yhat = pipe.predict(x)[0]
        st.success(f"予測強度: **{yhat:.3f}**")

st.caption("© Streamlit + scikit-learn | CSVの列名はUIから自由に指定できます。")
