import pathlib

import PIL
import streamlit as st
import torch

from src import DATASET_DIR, RAW_TEST_IMAGES_FOLDER
from src.bayesian_awb.bayesian_awb_algorithm import perform_bayes_AWB
from src.deep_awb.data_loaders import SimpleCubePPDatasetInfo, get_test_dataset
from src.deep_awb.model_inference import load_model, optimize_model
from src.visualization.image_visualization import apply_white_balance, compute_unbalanced_image, perform_color_correction_and_gamma


@st.cache_resource
def load_and_optimize_model(model_path=pathlib.Path(".") / "src" / "deep_awb" / "models" / "trained_traced_model.pt"):
    model = load_model(model_path)
    model = optimize_model(model, torch.randn(1, 3, 288, 432))  # Example input to optimize the model
    return model


@st.cache_data
def get_dataset():
    SimpleCubePPDatasetInfo.setup(1.5)
    return get_test_dataset()


model = load_and_optimize_model()
dataset = get_dataset()

image_paths = list(RAW_TEST_IMAGES_FOLDER.iterdir())
image_names = [path.name for path in image_paths]

st.sidebar.title("Auto White Balance Comparison")
selected_image = st.sidebar.selectbox("Choose an image", image_names)

raw_image_path = RAW_TEST_IMAGES_FOLDER / selected_image
WB_non_applied_image = compute_unbalanced_image(raw_image_path)

preprocessed_path = DATASET_DIR / "test" / "PROCESSED_UNIFORM" / selected_image
preprocessed_img = PIL.Image.open(preprocessed_path).convert("RGB")

img_index = image_names.index(selected_image)

ground_truth_gains = dataset.annotations.iloc[img_index].values[1:3]
ground_truth_img = perform_color_correction_and_gamma(apply_white_balance(WB_non_applied_image, ground_truth_gains))

model_input_image = dataset[img_index][0]
NN_predicted_gains = model(model_input_image.unsqueeze(0)).squeeze(0).detach().numpy()

predicted_nn_img = perform_color_correction_and_gamma(apply_white_balance(WB_non_applied_image, NN_predicted_gains))

classical_gains = perform_bayes_AWB(raw_image_path)
predicted_classical_img = perform_color_correction_and_gamma(apply_white_balance(WB_non_applied_image, classical_gains))

st.header(f"Comparison of White Balance Methods for `{selected_image}`")

col1, col2 = st.columns(2)


with col1:
    st.subheader("Uniform gains")
    st.image(preprocessed_img, caption="Is an input to the classical Bayesian WB.")

    st.subheader("Neural Network Prediction")
    st.image(predicted_nn_img, caption="Prediction by CNN trained on MLE processed images.")


with col2:
    st.subheader("Ground Truth")
    st.image(ground_truth_img, caption="Annotated ground truth applied.")

    st.subheader("Classical Bayesian Method")
    st.image(predicted_classical_img, caption="Classical Method Prediction. The one used in `libcamera`'s pipeline.")
