import json
from pathlib import Path
import threading

import numpy as np
import pyaudio

import dearpygui.dearpygui as dpg

from models import (
    train_model,
    load_model,
    save_model,
    get_ear,
    get_audio_features,
    preprocess_data,
    predict,
)

# Ensure metadata directory exists
Path("metadata/").mkdir(exist_ok=True)


# ——— Device Persistence Helpers ———


def save_metadata():
    with open("metadata/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def load_metadata():
    try:
        with open("metadata/metadata.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"default_device": "", "data_classes": []}


metadata = load_metadata()

# ——— Global State ———

ear = None  # StreamAnalyzer instance or None

device_list = []  # Available input devices

collected_data = {}  # { class_name: [data]}
record_val = None  # Currently selected class

model = None  # Loaded/trained model
infer_classes = []

# ——— Initialization ———
# Populate device_list with input-capable devices
audio = pyaudio.PyAudio()
for idx in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(idx)
    if info.get("maxInputChannels", 0) > 0:
        device_list.append(info["name"])

# Determine default device
default_device = device_list[0] if device_list else None
saved = metadata["default_device"]
if saved in device_list:
    default_device = saved

# Instantiate audio reader ('ear') if possible
if default_device:
    device_idx = device_list.index(default_device)
    ear = get_ear(device=device_idx)
else:
    ear = None

# data management


def set_infer_classes(new_classes):
    global infer_classes
    infer_classes = new_classes
    dpg.configure_item("infer_classes_list", items=infer_classes)


def format_collected_data(collected_data):
    # turns collected data into arrays for training
    x = []
    y = []
    class_names = []

    for i, (key, val) in enumerate(collected_data.items()):
        x.extend(val)
        y.extend([i for _ in range(len(val))])
        class_names.append(key)

    # maybe need to shuffle later

    x = np.array(x)
    y = np.array(y)

    return x, y, class_names


# ——— Callback Definitions ———
def device_select_callback(s, a, u):
    """Handle user selecting a new audio device."""
    device_name = a
    metadata["default_device"] = device_name
    save_metadata()
    global ear
    ear = get_ear(device=device_list.index(device_name))


# Class management callbacks


def add_class_callback():
    cls = dpg.get_value("class_input").strip()
    add_class(cls)


def add_class(cls):
    if not cls or cls in collected_data:
        dpg.set_value("class_input", "")
        return
    collected_data[cls] = []
    # Create UI group for the new class
    with dpg.group(parent="class_list", tag=f"grp_{cls}"):
        dpg.add_text(f"{cls}: 0", bullet=True, tag=f"cls_{cls}")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Select", callback=lambda: select_class_callback(cls))
            dpg.add_button(label="Clear", callback=lambda: clear_class(cls))
            dpg.add_button(label="Delete", callback=lambda: delete_class(cls))
        dpg.add_separator()
    dpg.set_value("class_input", "")

    # updates the metadata
    metadata["data_classes"] = list(collected_data.keys())
    save_metadata()


def select_class_callback(cls):
    """Switch recording target to the chosen class."""
    global record_val
    record_val = cls
    dpg.set_value("selected_class", f"Selected: {cls}")
    dpg.set_value("record_toggle", False)
    dpg.configure_item("record_toggle", enabled=True)


def clear_class(cls):
    collected_data[cls] = []
    dpg.set_value(f"cls_{cls}", f"{cls}: 0")


def clear_all_classes():
    for k in collected_data.keys():
        clear_class(k)


def delete_class(cls):
    collected_data.pop(cls, None)
    dpg.delete_item(f"grp_{cls}")

    # updates the metadata
    metadata["data_classes"] = list(collected_data.keys())
    save_metadata()


def delete_all_classes():
    for k in collected_data.keys():
        delete_class(k)


# Mode switching
def switch_mode(selected):
    """Toggle between Train and Infer UI groups."""
    train_visible = selected == "Train"
    dpg.configure_item("train_group", show=train_visible)
    dpg.configure_item("infer_group", show=not train_visible)
    dpg.set_value("mode", selected)


# Training/inference buttons
def train_model_callback():
    # makes sure theres some data to even train with (2 class min)
    data_collected = False
    n_valid = 0
    for key, val in collected_data.items():
        if len(val) > 0:
            n_valid += 1
        if n_valid >= 2:
            data_collected = True
            break

    dpg.configure_item("train_button", enabled=False)
    dpg.configure_item("training_status_text", show=True)

    def _background_training():
        X, y, class_names = format_collected_data(collected_data)
        X = preprocess_data(X)

        # train and set the global model
        global model
        model, infer_classes = train_model(X, y, class_names)
        set_infer_classes(infer_classes)

        dpg.configure_item("train_button", enabled=True)
        dpg.configure_item("training_status_text", show=False)
        switch_mode("Infer")

    if data_collected:
        threading.Thread(target=_background_training, daemon=True).start()


def load_model_callback():
    dpg.configure_item("load_model_dialog", show=True)


def save_model_callback():
    dpg.configure_item("save_model_dialog", show=True)


# File dialog callbacks
def load_model_dialog_callback(s, a, u):
    path = a.get("file_path_name", None)
    print(path)

    def _background_loading():
        try:
            p = Path(path)
            global model
            model, infer_classes = load_model(p)
            set_infer_classes(infer_classes)
        except:
            print("Loading Failed. Continuing without loading any model.")

    if path:
        threading.Thread(target=_background_loading, daemon=True).start()


def save_model_dialog_callback(s, a, u):
    path = a.get("file_path_name", None)

    def _background_saving():
        p = Path(path)
        global model
        global infer_classes
        save_model(model, infer_classes, p)

    if path:
        threading.Thread(target=_background_saving, daemon=True).start()


def frame_callback():
    # get the live data
    freqs, amps = get_audio_features(ear)

    # update graph
    dpg.set_value("fft_series", [list(freqs), list(amps)])

    # recording data to collected data
    if dpg.get_value("record_toggle") and record_val:
        # TODO: Maybe save the frequency X values for preprocessing?
        collected_data[record_val].append(amps)
        dpg.set_value(
            f"cls_{record_val}", f"{record_val}: {len(collected_data[record_val])}"
        )

    # live prediction
    if (dpg.get_value("mode") == "Infer") and model is not None:
        amps = np.array([amps])
        X = preprocess_data(amps)
        y_pred = predict(model, X)[0]
        if y_pred < len(infer_classes):
            y_label = infer_classes[y_pred]  # get the name
            dpg.set_value("pred_text", y_label)
            dpg.configure_item("infer_classes_list", default_value=y_label)
        else:
            pass
            raise ValueError("Y label predicted is not in the infer classes")

    dpg.set_frame_callback(dpg.get_frame_count() + 1, frame_callback)


# Resize handler
def on_vp_resize(s, a):
    W, H = dpg.get_viewport_width(), dpg.get_viewport_height()
    w2 = int(W * .95 * 0.7)
    h2 = .95 * H
    # Position and size windows dynamically
    dpg.configure_item("fft_vis", pos=(0, 0), width=w2, height=h2)
    dpg.configure_item("ctrl_win", pos=(w2, 0), width=W - w2, height=h2)


# ——— UI Construction ———

dpg.create_context()
dpg.create_viewport(title="FFT Trainer", width=800, height=600)

# Menu bar for device selection
with dpg.viewport_menu_bar():
    with dpg.menu(label="Options"):
        dpg.add_combo(
            items=device_list,
            label="Device",
            tag="device_combo",
            default_value=default_device,
            callback=device_select_callback,
            width=300,
        )  # TODO: ADD A CALLBACK

# FFT Visualization window
with dpg.window(
    tag="fft_vis", label="FFT Spectrum", no_move=True, no_resize=True, no_collapse=True
):
    dpg.add_text("Real-time FFT Signal", bullet=True)
    with dpg.plot(label="FFT Plot", height=-1, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis", label="Frequency (Hz)")
        dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis", label="Amplitude")
        dpg.set_axis_limits_auto("x_axis")
        dpg.set_axis_limits_auto("y_axis")
        dpg.add_line_series([], [], label="FFT", parent="y_axis", tag="fft_series")

# Control window with Train/Infer groups
with dpg.window(
    tag="ctrl_win", label="Controls", no_move=True, no_resize=True, no_collapse=True
):
    # Mode switch combo
    dpg.add_text("Mode:")
    dpg.add_combo(
        items=["Train", "Infer"],
        default_value="Train",
        tag="mode",
        callback=lambda s, a, u: switch_mode(a),
    )
    dpg.add_separator()

    # Training UI group
    with dpg.group(tag="train_group"):
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="class_input", width=-1)
            dpg.add_button(label="Add Class", callback=add_class_callback)
        dpg.add_separator()
        dpg.add_text("Classes & Data:", bullet=True)
        dpg.add_child_window(tag="class_list", width=-1, height=200)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Clear All", callback=delete_all_classes)
            dpg.add_button(label="Delete All", callback=clear_all_classes)
        dpg.add_separator()
        dpg.add_text(f"Selected: {record_val}", tag="selected_class")
        dpg.add_checkbox(label="Record Data", tag="record_toggle", enabled=False)
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Train Model", tag="train_button", callback=train_model_callback
            )
            dpg.add_text("Training...", tag="training_status_text", show=False)

    # Inference UI group
    with dpg.group(tag="infer_group", show=False):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Save Model", callback=save_model_callback, tag="btn_save"
            )
            dpg.add_button(
                label="Load Model", callback=load_model_callback, tag="btn_load"
            )
        dpg.add_text("Classes:", bullet=True)
        dpg.add_listbox(
            items=infer_classes,
            tag="infer_classes_list",
            num_items=5,
            default_value="sss",
        )
        with dpg.group(horizontal=True):
            dpg.add_text("Prediction: ")
            dpg.add_text("---", tag="pred_text")

# File dialogs
with dpg.file_dialog(
    tag="load_model_dialog",
    show=False,
    callback=load_model_dialog_callback,
    width=700,
    height=400,
    modal=True,
    default_path=".",
    label="Choose Model to Load",
):
    dpg.add_file_extension(".model")

with dpg.file_dialog(
    tag="save_model_dialog",
    show=False,
    callback=save_model_dialog_callback,
    label="Choose Save Location",
    width=700,
    height=400,
    modal=True,
    default_filename="model",
    file_count=1,
    default_path=".",
):
    dpg.add_file_extension(".model")

# Setup callbacks and start
dpg.set_viewport_resize_callback(on_vp_resize)
dpg.set_frame_callback(1, frame_callback)
on_vp_resize(None, None)  # initial layout

# loading initial dataclasses
for c in metadata["data_classes"]:
    add_class(c)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
