"""One long-lived Tk interpreter for CarDpy's interactive tools.

macOS application events retain interpreter pointers. Destroy child windows,
not the interpreter, between tools or notebook runs.
"""
import threading
import tkinter as tk
from tkinter import ttk

_root = None
_scan_name = None


def set_scan_name(scan_name):
    """Set the scan label applied to subsequent CarDpy GUI windows."""
    global _scan_name
    _scan_name = str(scan_name).strip() if scan_name else None


def with_scan_name(title):
    title = str(title)
    if _scan_name and _scan_name not in title:
        return f'{title} — {_scan_name}'
    return title


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
    window.title(with_scan_name(title))
    # Give controls a slightly larger hit target across all CardPy windows.
    ttk.Style(window).configure("TButton", padding=(14, 10))
    return window


def wait_window(window):
    root = get_root()
    root.wait_window(window)
    # Flush native close events before the pipeline resumes CPU-intensive work.
    root.update()
