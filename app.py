"""Deep Learning Laboratory - public CPU inference app (Streamlit).

This app never trains models and never downloads datasets. It only serves
checkpoints listed as 'available' in config/checkpoints.yaml after SHA-256 verification.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.checkpoint_manager import CheckpointError, get_entry, load_registry  # noqa: E402

st.set_page_config(page_title="Deep Learning Laboratory", page_icon="🧠", layout="wide")

DISCLAIMER = ("Educational and research prototype. Not a medical device. Must not be used for "
              "diagnosis, treatment decisions, or clinical decision-making.")

MODEL_NAMES = {
    "01_mlp": "01 · MLP (tabular classification)",
    "02_cnn": "02 · CNN", "03_rnn": "03 · RNN", "04_lstm": "04 · LSTM", "05_gru": "05 · GRU",
    "06_autoencoder": "06 · Autoencoder", "07_vae": "07 · Variational Autoencoder",
    "08_gan": "08 · GAN", "09_transformer": "09 · Transformer", "10_bert": "10 · BERT-style",
    "11_gpt": "11 · GPT-style", "12_resnet": "12 · ResNet", "13_densenet": "13 · DenseNet",
    "14_vit": "14 · Vision Transformer", "15_unet": "15 · U-Net", "16_yolo": "16 · YOLO-style",
    "17_siamese": "17 · Siamese", "18_seq2seq": "18 · Seq2Seq", "19_diffusion": "19 · Diffusion",
    "20_gnn": "20 · Graph Neural Network",
}

STATUS_LABEL = {
    "available": "✅ Available (verified checkpoint)",
    "not_trained": "⬜ Not trained",
    "training_ready": "🟡 Training-ready, not trained",
    "trained_not_published": "🟠 Trained, not published",
}

st.sidebar.title("🧠 Deep Learning Lab")
page = st.sidebar.radio("Navigate", [
    "Home Dashboard", "Deep Learning Laboratory", "Medical AI", "Smart Agriculture",
    "Model Information", "Documentation", "About the Project",
])
st.sidebar.caption(DISCLAIMER)


def home() -> None:
    st.title("Deep Learning Laboratory")
    st.info(DISCLAIMER)
    entries = load_registry()
    available = [k for k, v in entries.items() if v.status == "available"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Architectures registered", len([k for k in entries if k[:2].isdigit()]))
    c2.metric("Available for inference", len(available))
    c3.metric("Not yet trained", len([v for v in entries.values() if v.status != "available"]))
    st.markdown("Only models whose trained checkpoints pass SHA-256 verification are served. "
                "Other entries are listed honestly as not yet trained.")
    st.dataframe(
        [{"Model": MODEL_NAMES.get(k, k), "Status": STATUS_LABEL.get(v.status, v.status), "Notes": v.notes}
         for k, v in entries.items()],
        width="stretch", hide_index=True,
    )


def mlp_page() -> None:
    from core.inference_engine import get_model

    st.header("01 · MLP — UCI Adult income prediction")
    st.write("Predicts whether a person's annual income is **>50K** from census-style features. "
             "Educational demonstration on 1994 US census data; not for real-world decisions about individuals.")
    try:
        model = get_model("01_mlp")
    except CheckpointError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:  # unexpected load failure: show it, do not hide it
        st.error(f"Model could not be loaded: {exc}")
        return

    vocab = model.vocab
    with st.form("adult_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", 17, 90, 39)
            fnlwgt = st.number_input("fnlwgt (sampling weight)", 10000, 1500000, 77516)
            education_num = st.number_input("Education (years)", 1, 16, 13)
            capital_gain = st.number_input("Capital gain", 0, 100000, 2174)
            capital_loss = st.number_input("Capital loss", 0, 5000, 0)
            hours = st.number_input("Hours per week", 1, 99, 40)
        with right:
            workclass = st.selectbox("Workclass", vocab["workclass"], index=vocab["workclass"].index("State-gov") if "State-gov" in vocab["workclass"] else 0)
            education = st.selectbox("Education level", vocab["education"], index=vocab["education"].index("Bachelors") if "Bachelors" in vocab["education"] else 0)
            marital = st.selectbox("Marital status", vocab["marital_status"])
            occupation = st.selectbox("Occupation", vocab["occupation"])
            relationship = st.selectbox("Relationship", vocab["relationship"])
            race = st.selectbox("Race", vocab["race"])
            sex = st.selectbox("Sex", vocab["sex"])
            country = st.selectbox("Native country", vocab["native_country"])
        submitted = st.form_submit_button("Predict income", type="primary")

    if submitted:
        record = {
            "age": age, "fnlwgt": fnlwgt, "education_num": education_num,
            "capital_gain": capital_gain, "capital_loss": capital_loss, "hours_per_week": hours,
            "workclass": workclass, "education": education, "marital_status": marital,
            "occupation": occupation, "relationship": relationship, "race": race,
            "sex": sex, "native_country": country,
        }
        result = model.predict(record)
        p = result["probability_gt_50k"]
        st.subheader(f"Predicted: {result['label']}")
        st.metric("Probability of income >50K", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        st.caption(f"Decision threshold {result['threshold']:.2f} (chosen on validation data).")


def model_info() -> None:
    st.title("Model Information")
    for k, v in load_registry().items():
        with st.expander(MODEL_NAMES.get(k, k)):
            st.write(f"**Status:** {STATUS_LABEL.get(v.status, v.status)}")
            st.write(v.notes)
            if v.sha256:
                st.code(v.sha256, language="text")


def placeholder(title: str, body: str) -> None:
    st.title(title)
    st.info(body)


if page == "Home Dashboard":
    home()
elif page == "Deep Learning Laboratory":
    st.title("Deep Learning Laboratory")
    mlp_page()
elif page == "Medical AI":
    placeholder("Medical AI", "Not yet implemented. No medical model is trained or served. " + DISCLAIMER)
elif page == "Smart Agriculture":
    placeholder("Smart Agriculture", "Not yet implemented. No agriculture model is trained or served.")
elif page == "Model Information":
    model_info()
elif page == "Documentation":
    st.title("Documentation")
    st.markdown((ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else "README pending.")
else:
    placeholder("About the Project", "An educational deep-learning laboratory built with PyTorch, scikit-learn, and Streamlit.")
