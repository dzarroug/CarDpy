"""Optional review of automatic shot rejection, before any images are replaced."""
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from cardpy.GUI_Tools._runtime import create_window, wait_window


class RejectionReviewWindow:
    def __init__(self, matrix, bvals, bvecs, keep):
        self.matrix = np.asarray(matrix)
        self.bvals = np.asarray(bvals)
        self.bvecs = np.asarray(bvecs)
        self.automatic = np.asarray(keep).copy()
        if self.matrix.ndim != 4 or any(n == 0 for n in self.matrix.shape):
            raise ValueError('Review expects [rows, columns, slices, acquisitions].')
        acquisitions = self.matrix.shape[3]
        if (self.automatic.shape != self.matrix.shape[2:]
                or self.bvals.shape != (acquisitions,)
                or self.bvecs.shape != (acquisitions, 3)
                or not np.isin(self.automatic, [0, 1]).all()):
            raise ValueError('Review metadata and binary keep decisions must match the images.')
        self.keep = self.automatic.copy()
        self.slc = 0
        self.result = None
        self.photos = []
        self.tiles = []
        self.labels = self.acquisition_labels()
        self.window = create_window('CarDpy: Review Shot Rejection')
        width = min(1100, int(self.window.winfo_screenwidth() * 0.85))
        height = min(850, int(self.window.winfo_screenheight() * 0.85))
        self.window.geometry(f'{width}x{height}')
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)
        self.window.protocol('WM_DELETE_WINDOW', self.cancel)
        ttk.Label(self.window, text='Click an image to reject or restore it. Red = reject; green = keep.\n'
                  '“Auto: reject” marks the algorithm’s original selection, even after you restore it.',
                  anchor='center', justify='center').grid(row=0, column=0, sticky='ew', padx=8, pady=8)
        self.status = ttk.Label(self.window, anchor='center')
        self.status.grid(row=1, column=0, sticky='ew', pady=4)
        gallery = ttk.Frame(self.window)
        gallery.grid(row=2, column=0, sticky='nsew')
        gallery.columnconfigure(0, weight=1)
        gallery.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(gallery, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(gallery, orient='vertical', command=self.canvas.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.grid = ttk.Frame(self.canvas)
        self.grid_id = self.canvas.create_window((0, 0), window=self.grid, anchor='nw')
        self.grid.bind('<Configure>', lambda event: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', self.resize)
        self.canvas.bind('<MouseWheel>', self.scroll)
        self.canvas.bind('<Button-4>', lambda event: self.canvas.yview_scroll(-1, 'units'))
        self.canvas.bind('<Button-5>', lambda event: self.canvas.yview_scroll(1, 'units'))
        footer = ttk.Frame(self.window, padding=8)
        footer.grid(row=3, column=0, sticky='ew')
        ttk.Button(footer, text='Cancel', command=self.cancel).pack(side='left')
        ttk.Button(footer, text='Reset slice to automatic', command=self.reset_slice).pack(side='left', padx=6)
        self.back = ttk.Button(footer, text='Previous slice', command=lambda: self.change_slice(-1))
        self.back.pack(side='left', padx=6)
        self.next = ttk.Button(footer, text='Next slice', command=lambda: self.change_slice(1))
        self.next.pack(side='left', padx=6)
        ttk.Button(footer, text='Apply all slices', command=self.apply).pack(side='right')
        ttk.Label(self.window, text='Rejected repetitions use the existing replacement rules. '
                  'If all repetitions of an encoding are rejected, the first is retained.',
                  wraplength=width - 30, anchor='center').grid(row=4, column=0, sticky='ew', padx=8, pady=6)
        self.load_slice()

    def acquisition_labels(self):
        encodings = {}
        labels = []
        for bval, bvec in zip(self.bvals, self.bvecs):
            key = (float(bval), tuple(bvec))
            if key not in encodings:
                encodings[key] = [len(encodings) + 1, 0]
            encodings[key][1] += 1
            direction, repetition = encodings[key]
            labels.append(f'Encoding {direction} · Rep {repetition}\nb = {bval:g}')
        return labels

    def load_slice(self):
        for widget in self.grid.winfo_children():
            widget.destroy()
        self.photos = []
        self.tiles = []
        for index in range(self.matrix.shape[3]):
            image = np.abs(self.matrix[:, :, self.slc, index])
            image = np.where(np.isfinite(image), image, 0)
            high = float(np.percentile(image[image > 0], 99)) if np.any(image > 0) else 1.0
            pixels = (np.clip(image / high, 0, 1) * 255).astype(np.uint8)
            thumbnail = Image.fromarray(pixels)
            thumbnail.thumbnail((175, 175), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(thumbnail, master=self.window)
            self.photos.append(photo)
            tile = tk.Label(self.grid, image=photo, compound='top', width=190,
                            padx=4, pady=6, relief='solid', borderwidth=3,
                            cursor='hand2', takefocus=True)
            tile.bind('<Button-1>', lambda event, idx=index: self.toggle(idx))
            tile.bind('<space>', lambda event, idx=index: self.toggle(idx))
            tile.bind('<Return>', lambda event, idx=index: self.toggle(idx))
            tile.bind('<MouseWheel>', self.scroll)
            tile.bind('<Button-4>', lambda event: self.canvas.yview_scroll(-1, 'units'))
            tile.bind('<Button-5>', lambda event: self.canvas.yview_scroll(1, 'units'))
            self.tiles.append(tile)
            self.update_tile(index)
        self.layout()
        self.canvas.yview_moveto(0)
        self.update_status()

    def layout(self):
        columns = max(1, self.canvas.winfo_width() // 210)
        for index, tile in enumerate(self.tiles):
            tile.grid(row=index // columns, column=index % columns, padx=5, pady=5, sticky='n')

    def resize(self, event):
        self.canvas.itemconfigure(self.grid_id, width=event.width)
        self.layout()

    def scroll(self, event):
        if event.delta:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')
        return 'break'

    def update_tile(self, index):
        kept = bool(self.keep[self.slc, index])
        automatic = 'reject' if self.automatic[self.slc, index] == 0 else 'keep'
        self.tiles[index].configure(
            text=f'{self.labels[index]}\n{"KEEP" if kept else "REJECT"} · Auto: {automatic}',
            background='#d9f0df' if kept else '#ffd7d7', foreground='#174d2a' if kept else '#8b1010')

    def update_status(self):
        count = np.count_nonzero(self.keep[self.slc] == 0)
        total = np.count_nonzero(self.keep == 0)
        self.status.configure(text=f'Slice {self.slc + 1} of {self.matrix.shape[2]} · '
                              f'{count} / {self.matrix.shape[3]} rejected here · {total} rejected across all slices')
        self.back.configure(state='normal' if self.slc else 'disabled')
        self.next.configure(state='normal' if self.slc < self.matrix.shape[2] - 1 else 'disabled')

    def toggle(self, index):
        self.keep[self.slc, index] = 1 - self.keep[self.slc, index]
        self.update_tile(index)
        self.update_status()
        return 'break'

    def reset_slice(self):
        self.keep[self.slc] = self.automatic[self.slc]
        for index in range(len(self.tiles)):
            self.update_tile(index)
        self.update_status()

    def change_slice(self, delta):
        target = self.slc + delta
        if 0 <= target < self.matrix.shape[2]:
            self.slc = target
            self.load_slice()

    def apply(self):
        self.result = self.keep.copy()
        self.window.destroy()

    def cancel(self):
        self.result = None
        self.window.destroy()


def review_rejection(matrix, bvals, bvecs, keep):
    """Return edited stacked decisions; cancellation aborts the calling pipeline."""
    app = RejectionReviewWindow(matrix, bvals, bvecs, keep)
    wait_window(app.window)
    if app.result is None:
        raise RuntimeError('Shot rejection review cancelled.')
    return app.result
