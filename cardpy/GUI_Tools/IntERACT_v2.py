"""Box-based crop selection. Inputs and crop-coordinate outputs match IntERACT."""
import numpy as np
from tkinter import ttk
from matplotlib.widgets import RectangleSelector
from cardpy.GUI_Tools._runtime import create_window, wait_window
from cardpy.GUI_Tools._image_controls import ImagePanel, crop_bounds, size_window


class CropWindow:
    def __init__(self, matrix, organ):
        self.matrix = np.asarray(matrix)
        if self.matrix.ndim != 4 or any(size == 0 for size in self.matrix.shape):
            raise ValueError('Cropping expects [rows, columns, slices, acquisitions].')
        self.slc = 0
        self.saved = [None] * self.matrix.shape[2]
        self.bounds = crop_bounds(None, self.matrix.shape)
        self.result = None
        self.window = create_window('CarDpy: Box Crop — ' + str(organ))
        size_window(self.window)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        self.window.protocol('WM_DELETE_WINDOW', self.cancel)
        self.title = ttk.Label(self.window, anchor='center')
        self.title.grid(row=0, column=0, sticky='ew', pady=8)
        self.panel = ImagePanel(self.window, 'Drag a box; drag its handles to adjust.', self.projection())
        self.panel.grid(row=1, column=0, sticky='nsew')
        self.selector = RectangleSelector(
            self.panel.axes, self.select, button=[1], interactive=True,
            minspanx=1, minspany=1, spancoords='data', useblit=False,
            props={'edgecolor': '#61ba86', 'facecolor': '#61ba86', 'alpha': 0.3, 'linewidth': 2})
        footer = ttk.Frame(self.window, padding=8)
        footer.grid(row=2, column=0, sticky='ew')
        ttk.Button(footer, text='Cancel', command=self.cancel).pack(side='left')
        ttk.Button(footer, text='Draw new box', command=self.selector.clear).pack(side='left', padx=6)
        ttk.Button(footer, text='Full image', command=self.reset).pack(side='left', padx=6)
        self.back_button = ttk.Button(footer, text='Back', command=self.back)
        self.back_button.pack(side='left', padx=6)
        self.next_button = ttk.Button(footer, command=self.next_slice)
        self.next_button.pack(side='right')
        self.update_box()

    def projection(self):
        images = np.abs(self.matrix[:, :, self.slc, :])
        return np.max(np.where(np.isfinite(images), images, 0), axis=2)

    def update_box(self):
        x0, x1, y0, y1 = self.bounds
        self.selector.extents = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)
        self.title.configure(text=f'Slice {self.slc + 1} of {len(self.saved)} | '
                                  f'Box: x [{x0}, {x1}), y [{y0}, {y1}) | '
                                  f'{x1 - x0} × {y1 - y0} pixels')
        self.next_button.configure(text='Save & Finish' if self.slc == len(self.saved) - 1 else 'Save & Next')
        self.back_button.configure(state='normal' if self.slc else 'disabled')
        self.panel.canvas.draw_idle()

    def select(self, press, release):
        x0, x1, y0, y1 = self.selector.extents
        self.bounds = crop_bounds([x0 + 0.5, x1 + 0.5, y0 + 0.5, y1 + 0.5], self.matrix.shape)
        self.update_box()

    def reset(self):
        self.bounds = crop_bounds(None, self.matrix.shape)
        self.update_box()

    def load_slice(self):
        if self.saved[self.slc] is not None:
            self.bounds = self.saved[self.slc].copy()
        self.panel.data = self.projection()
        self.panel.artist.set_data(self.panel.data)
        self.update_box()

    def back(self):
        if self.slc:
            self.saved[self.slc] = self.bounds.copy()
            self.slc -= 1
            self.load_slice()

    def next_slice(self):
        self.saved[self.slc] = self.bounds.copy()
        if self.slc == len(self.saved) - 1:
            self.result = [list(values) for values in zip(*self.saved)]
            self.close()
        else:
            self.slc += 1
            self.load_slice()

    def close(self):
        self.selector.disconnect_events()
        self.panel.close()
        self.window.destroy()

    def cancel(self):
        self.result = None
        self.close()


def INTERACT_GUI(original_matrix, organ_of_intrest):
    app = CropWindow(original_matrix, organ_of_intrest)
    wait_window(app.window)
    if app.result is None:
        raise RuntimeError('Crop selection cancelled.')
    return app.result
