import numpy as np
import dearpygui.dearpygui as dpg
# from stream_analyzer import StreamAnalyzer

# Placeholder ML functions (students should implement)
def train_model(data_windows, labels, save_path):
    # data_windows: list of `np.array` FFT vectors
    # labels: list of strings
    # save model to save_path
    pass

def load_model(file_path):
    # load and return a trained model
    return None


def predict_model(model, fft_window):
    # return predicted label for a single FFT window
    return ""

# -----------------------------------------------------------------------------
# Initialize FFT stream
# ear = StreamAnalyzer(verbose=False)

def cleanup():
    # ear.stream_reader.stream_stop()
    dpg.stop_dearpygui()

# Data storage for training
collected_data = {}  # { class_name: [fft_windows, ...] }
model = None

def make_dummy_spectrum(n_bins=512, peaks=[50, 150, 300], rate=1000):
    """
    Returns (freqs, amps) arrays.
     - freqs: 0…rate/2
     - amps: sum of a few Gaussian peaks + small noise
    """
    freqs = np.linspace(0, rate/2, n_bins)
    amps = np.zeros_like(freqs)
    for p in peaks:
        amps += np.exp(-0.5 * ((freqs - p) / 5)**2) * 5
    amps += 0.2 * np.random.rand(n_bins)
    return freqs, amps

def register_button(
    label: str,
    callback,
    toggle: bool = False,
    parent: str = "extension_panel",
    tag: str = None
):
    """
    Adds a button (or toggle button) to the given parent container.

    • label    – text shown on the button
    • callback – function to call on click; for toggles you’ll get an extra bool param
    • toggle   – if True, creates a persistent toggle button
    • parent   – DPG item tag under which the button is inserted
    • tag      – optional unique tag for the new button

    Example usage:
      register_button("Say Hi", lambda: console_print("Hello!"))
      register_button("Dark Mode", toggle=True,
                      callback=lambda s,a,u: set_dark_mode(a))
    """
    # Ensure we have a unique tag
    tag = tag or f"btn_{label.replace(' ', '_')}_{np.random.randint(1e6)}"

    if toggle:
        # create a boolean state and a toggle button
        dpg.add_checkbox(label=label,
                         tag=tag,
                         parent=parent,
                         callback=lambda s,a,u: callback(s, a, u))
    else:
        # normal momentary push button
        dpg.add_button(label=label,
                       tag=tag,
                       parent=parent,
                       callback=lambda s,a,u: callback())

# -----------------------------------------------------------------------------
# GUI setup

# Start DearPyGui context
dpg.create_context()
dpg.create_viewport(title="FFT Trainer", width=900, height=600, resizable=True)

with dpg.window(label="FFT Spectrum", tag="MainWin", width=600, height=400):
    dpg.add_text("Real-time FFT Signal", bullet=True)
    plot = dpg.add_plot(label="FFT Plot", height=-1, width=-1)
    x_axis = dpg.add_plot_axis(dpg.mvXAxis, parent=plot, label="Frequency (Hz)")
    y_axis = dpg.add_plot_axis(dpg.mvYAxis, parent=plot, label="Amplitude")
    x_series = dpg.add_line_series([], [], label="FFT", parent=y_axis, tag="fft_series")

with dpg.window(label="Controls", tag="CtrlWin", width=300, height=400, pos=(610, 0)):
    # Mode switch
    dpg.add_text("Mode:")
    mode_combo = dpg.add_combo(items=["Train", "Infer"], default_value="Train", callback=lambda s,a,u: switch_mode(a), tag="mode")
    dpg.add_separator()

    # Train UI container
    with dpg.group(tag="train_group"):
        dpg.add_input_text(label="Class name", tag="class_input")
        dpg.add_button(label="Add Class", callback=lambda: add_class(), tag="btn_add_class")
        dpg.add_separator()
        dpg.add_text("Classes & Data:", bullet=True)
        dpg.add_child_window(width=280, height=200, tag="class_list")
        dpg.add_separator()
        dpg.add_button(label="Train Model", callback=lambda: on_train(), tag="btn_train")

    # Infer UI container
    with dpg.group(tag="infer_group", show=False):
        dpg.add_button(label="Load Model", callback=lambda: on_load_model(), tag="btn_load")
        dpg.add_text("Prediction:", bullet=True)
        dpg.add_text("---", tag="pred_text")

    dpg.add_button(label="Quit", callback=cleanup)

with dpg.window(label="Console", width= 600, height=150, pos=(0,450)):
    dpg.add_input_text(tag="console", multiline=True, height=-1, readonly=True, default_value="")

with dpg.window(tag="extension_panel",
                      label="Extensions",
                      width=300, height=-1,
                      no_title_bar=True):
    dpg.add_text("Add-on Buttons:", bullet=True)


# helper to append:
def console_print(msg):
    prev = dpg.get_value("console")
    dpg.set_value("console", prev + msg + "\n")


# File dialogs
with dpg.file_dialog(label="Save Model", show=False, callback=lambda s,a,u: do_save(a['file_path_name']), tag="save_dialog", default_path="."):
    dpg.add_button(label="Save", callback=lambda: None)

with dpg.file_dialog(label="Load Model", show=False, callback=lambda s,a,u: do_load(a['file_path_name']), tag="load_dialog", default_path="."):
    dpg.add_button(label="OK", callback=lambda: None)

# -----------------------------------------------------------------------------
# Callbacks and Helpers

def switch_mode(selected):
    if selected == "Train":
        dpg.configure_item("train_group", show=True)
        dpg.configure_item("infer_group", show=False)
    else:
        dpg.configure_item("train_group", show=False)
        dpg.configure_item("infer_group", show=True)


def add_class():
    cls = dpg.get_value("class_input").strip()
    if not cls:
        return
    if cls not in collected_data:
        collected_data[cls] = []
        # Add a child item for this class
        with dpg.group(parent="class_list", tag=f"grp_{cls}"):
            dpg.add_text(f"{cls}:", bullet=True)
            dpg.add_button(label="Collect FFT Window", callback=lambda s,a,u,cls=cls: collect_fft(cls))
            dpg.add_button(label="Clear Data", callback=lambda s,a,u,cls=cls: clear_class(cls))
            dpg.add_separator()
    dpg.set_value("class_input", "")


def collect_fft(cls):
    # _, fft_vals, _, _ = ear.get_audio_features()
    _, fft_vals = make_dummy_spectrum()
    collected_data[cls].append(fft_vals.copy())
    print(f"Collected window for class '{cls}', total={len(collected_data[cls])}")


def clear_class(cls):
    collected_data[cls].clear()
    print(f"Cleared data for class '{cls}'")


def on_train():
    # Build flat lists
    data = []
    labels = []
    for cls, windows in collected_data.items():
        data.extend(windows)
        labels.extend([cls]*len(windows))
    # Open save dialog
    dpg.show_item("save_dialog")
    # Store pending train args
    dpg.set_item_user_data("save_dialog", (data, labels))


def do_save(path):
    data, labels = dpg.get_item_user_data("save_dialog")
    train_model(data, labels, path)
    print(f"Model trained and saved to {path}")
    dpg.hide_item("save_dialog")
    # switch to infer mode
    dpg.set_value(mode_combo, "Infer")
    switch_mode("Infer")


def on_load_model():
    dpg.show_item("load_dialog")


def do_load(path):
    global model
    model = load_model(path)
    print(f"Model loaded from {path}")
    dpg.hide_item("load_dialog")

# -----------------------------------------------------------------------------
# Render/update loop

def update_spectrum():
    # get FFT
    # freqs, amps, _, _ = ear.get_audio_features()
    freqs, amps = make_dummy_spectrum()
    # update plot
    y = amps / (amps.max() or 1)
    dpg.set_value(x_series, [list(freqs), list(y)])

    if dpg.get_value("mode") == "Infer" and model is not None:
        pred = predict_model(model, amps)
        dpg.set_value("pred_text", pred)

# Start GUI
dpg.setup_dearpygui()
dpg.show_viewport()


try:
    while dpg.is_dearpygui_running():
        update_spectrum()            # push your FFT→plot updates
        dpg.render_dearpygui_frame() # draw one frame
        console_print("test")
finally:
    cleanup()      # stops audio, calls dpg.stop_dearpygui()
    dpg.destroy_context()
