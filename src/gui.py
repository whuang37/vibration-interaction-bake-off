import numpy as np
import dearpygui.dearpygui as dpg
import pyaudio 
from pathlib import Path
from stream_analyzer import StreamAnalyzer

Path("metadata/").mkdir(exist_ok=True)

def training(data):
    pass

def load_model(file_path):
    pass

def save_model(file_path):
    pass

def get_ear(device):
    return StreamAnalyzer(device=device, smoothing_length_ms=100)

def get_audio_features(ear):
    freqs, amps, _, _ = ear.get_audio_features()
    
    return freqs, amps

def save_default_device(device_name):
    with open('metadata/default_device.txt', "w") as f:
        f.write(device_name)

def load_default_device():
    try:
        with open('metadata/default_device.txt', "r") as f:
            return f.read().strip()
    except:
        return None

pa = pyaudio.PyAudio()
device_list = []
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    # Only include devices with at least one input channel
    if info.get("maxInputChannels", 0) > 0:
        device_list.append(info["name"]
        )
print(device_list)

# load some default device saved
default_device = ""
default_device = device_list[0] if device_list else ""
loaded_device = load_default_device()
if loaded_device is not None:
    if loaded_device in device_list:
        default_device = loaded_device

ear = get_ear(device=device_list.index(default_device))

# Data storage for training
collected_data = {}  # { class_name: [fft_windows, ...] }
record_val = None
recording_on = False

model = None


def device_select_callback(s, a, u):
    device_idx = device_list.index(a)
    save_default_device(a)
    global ear
    ear = get_ear(device=device_idx)

def add_class():
    cls = dpg.get_value("class_input").strip()
    if not cls:
        return
    if cls not in collected_data:
        collected_data[cls] = []
        # Add a child item for this class
        with dpg.group(parent="class_list", tag=f"grp_{cls}"):
            dpg.add_text(f"{cls}: 0", bullet=True, tag=f"cls_{cls}")
            dpg.add_button(label="Select", callback=lambda s,a,u: select_class_callback(cls))
            dpg.add_button(label="Clear Data", callback=lambda s,a,u: clear_class(cls))
            dpg.add_button(label="Delete Class", callback=lambda s,a,u: delete_class(cls))
            dpg.add_separator()
    dpg.set_value("class_input", "")

def switch_mode(selected):
    if selected == "Train":
        dpg.configure_item("train_group", show=True)
        dpg.configure_item("infer_group", show=False)
        dpg.configure_item("mode", default_value="Train")
    else:
        dpg.configure_item("train_group", show=False)
        dpg.configure_item("infer_group", show=True)
        dpg.configure_item("mode", default_value="Infer")

def record_data_callback(s, a, u):
    if record_val is None:
        dpg.configure_item("record_toggle", default_value=False)
        return
    global recording_on
    recording_on = True if a else False
    
def select_class_callback(cls):
    global record_val
    record_val = cls
    
    dpg.configure_item("record_toggle", default_value=False)
    dpg.configure_item("selected_class", default_value=f"Selected: {record_val}")

def clear_class(cls):
    if cls in collected_data:
        collected_data[cls] = []
        dpg.configure_item(f"cls_{record_val}", default_value=f"{record_val}: {len(collected_data[record_val])}")

def delete_class(cls):
    if cls in collected_data:
        collected_data.pop(cls, None)
        dpg.delete_item(f"grp_{cls}")
        
def train_model_callback():
    training(collected_data)
    switch_mode("Infer")

def load_model_callback():
    dpg.configure_item("load_model_dialog", show=True)

def save_model_callback():
    dpg.configure_item("save_model_dialog", show=True)

dpg.create_context()
dpg.create_viewport(title="FFT Trainer", width=800, height=600)

with dpg.viewport_menu_bar():
    with dpg.menu(label="Options"):
        dpg.add_combo(items=device_list,
                        label="Device",
                        tag="device_combo",
                        default_value=default_device,
                        callback=device_select_callback,
                        width=300) # TODO: ADD A CALLBACK

with dpg.window(label="FFT Spectrum", tag="fft_vis", no_move=True, no_resize=True, no_collapse=True):
    dpg.add_text("Real-time FFT Signal", bullet=True)
    plot = dpg.add_plot(label="FFT Plot", height=-1, width=-1)
    x_axis = dpg.add_plot_axis(dpg.mvXAxis, parent=plot, label="Frequency (Hz)")
    y_axis = dpg.add_plot_axis(dpg.mvYAxis, parent=plot, label="Amplitude")
    x_series = dpg.add_line_series([], [], label="FFT", parent=y_axis, tag="fft_series")

with dpg.window(label="Controls", tag="ctrl_win", no_move=True, no_resize=True, no_collapse=True):
    # Mode switch
    with dpg.group(horizontal=True):
        dpg.add_text("Mode:")
        mode_combo = dpg.add_combo(items=["Train", "Infer"], default_value="Train", callback=lambda s,a,u: switch_mode(a), tag="mode")
    dpg.add_separator()
    
    # Train UI container
    with dpg.group(tag="train_group"):
        with dpg.group(horizontal=True):
            dpg.add_text("Class Name:")
            dpg.add_input_text(label="Class name", tag="class_input")
        dpg.add_button(label="Add Class", callback=lambda: add_class(), tag="btn_add_class")
        dpg.add_separator()
        dpg.add_text("Classes & Data:", bullet=True)
        dpg.add_child_window(width=-1, height=200, tag="class_list")
        dpg.add_separator()
        # dpg.add_button(label="Train Model", callback=lambda: on_train(), tag="btn_train")
        dpg.add_text(f"Selected: {record_val}", tag="selected_class")
        dpg.add_checkbox(label="Record data", tag="record_toggle", default_value=False,
                         callback=record_data_callback)
        dpg.add_button(label="Train Model", tag="train_button", callback=train_model_callback)

    # Infer UI container
    with dpg.group(tag="infer_group", show=False):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Load Model", callback=lambda: load_model_callback(), tag="btn_load")
            dpg.add_button(label="Save Model", callback=lambda: save_model_callback(), tag="btn_save")
        dpg.add_text("Prediction:", bullet=True)
        dpg.add_text("---", tag="pred_text")
    
dpg.add_file_dialog(show=False, label="Load Model", tag="load_model_dialog", width=700 ,height=400)
dpg.add_file_extension(".*", label="Save Model", parent="load_model_dialog")
    
dpg.add_file_dialog(show=False, tag="save_model_dialog", width=700 ,height=400)
dpg.add_file_extension(".*", parent="save_model_dialog")

# Every time the viewport changes, recompute each window’s pos & size
def on_vp_resize(_, __):
    W = dpg.get_viewport_width()
    H = dpg.get_viewport_height()
    w2 = int(W * 0.7)
    h_top = int(H)
    h_bot = H - h_top

    # Top row
    dpg.configure_item("fft_vis", pos=(0,      0),       width=w2, height=h_top)
    dpg.configure_item("ctrl_win", pos=(w2,     0),       width=w2, height=h_top)
    # Bottom row
    # dpg.configure_item("win_bl", pos=(0,      h_top),   width=w2, height=h_bot)
    # dpg.configure_item("win_br", pos=(w2,     h_top),   width=w2, height=h_bot)
    
def update_spectrum():
    # get FFT
    freqs, amps = get_audio_features(ear)
    # update plot
    # y = amps / (amps.max() or 1)
    dpg.set_value(x_series, [list(freqs), list(amps)])

def record_data():
    if not recording_on or record_val is None:
        return
    
    else:
        if record_val not in collected_data:
            collected_data[record_val] = []
        
        global ear
        freqs, amps = get_audio_features(ear)
        collected_data[record_val].append((freqs, amps))
        dpg.configure_item(f"cls_{record_val}", default_value=f"{record_val}: {len(collected_data[record_val])}")
    
def frame_callback():
    record_data()
    update_spectrum()
    
    dpg.set_frame_callback(dpg.get_frame_count()+1, frame_callback)
    
    
dpg.set_viewport_resize_callback(on_vp_resize)
dpg.set_frame_callback(1, frame_callback)
on_vp_resize(None, None)  # initialize

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()