"""One long-lived Tk interpreter for CarDpy's interactive tools.

macOS application events retain interpreter pointers. Destroy child windows,
not the interpreter, between tools or notebook runs.
"""
import threading
import tkinter as tk

_root = None


def get_root():
    global _root
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('CarDpy GUI tools must run on the main thread.')
    if _root is None:
        _root = tk._default_root
        if _root is None:
            _root = tk.Tk()
            _root.withdraw()
    if not _root.winfo_exists():
        raise RuntimeError('The Tk application root was destroyed. Restart the kernel before opening a GUI.')
    return _root


def create_window(title):
    window = tk.Toplevel(master=get_root())
    window.title(title)
    return window


def wait_window(window):
    root = get_root()
    root.wait_window(window)
    # Flush native close events before the pipeline resumes CPU-intensive work.
    root.update()
