"""Image panels and display-only range controls for the optional v2 tools."""
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def display_limits(data, upper=None):
    values = np.asarray(data)
    values = values[np.isfinite(values)]
    positive = values[values > 0]
    high = float(np.percentile(positive, 99)) if positive.size else 1.0
    high = float(upper) if upper is not None else high
    if not np.isfinite(high) or high <= 0:
        raise ValueError('The display maximum must be positive and finite.')
    return 0.0, high


def crop_bounds(bounds, shape):
    rows, columns = shape[:2]
    if bounds is None:
        return [0, columns, 0, rows]
    values = np.asarray(bounds, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Crop bounds must be [x_start, x_end, y_start, y_end].')
    x0, x1 = sorted(values[:2])
    y0, y1 = sorted(values[2:])
    x0 = min(columns - 1, max(0, int(np.floor(x0))))
    y0 = min(rows - 1, max(0, int(np.floor(y0))))
    x1 = min(columns, max(x0 + 1, int(np.ceil(x1))))
    y1 = min(rows, max(y0 + 1, int(np.ceil(y1))))
    return [x0, x1, y0, y1]


def scale_crop(bounds, source_shape, target_shape):
    x_scale = target_shape[1] / source_shape[1]
    y_scale = target_shape[0] / source_shape[0]
    return crop_bounds(np.asarray(bounds) * [x_scale, x_scale, y_scale, y_scale], target_shape)


class RangeBar(tk.Canvas):
    """Two handles on one display-range bar, with keyboard adjustment."""
    def __init__(self, master, command):
        background = ttk.Style(master).lookup('TFrame', 'background') or '#ECECEC'
        super().__init__(master, height=30, highlightthickness=0, background=background, takefocus=True)
        self.command = command
        self.values = (0, 1)
        self.ceiling = 2
        self.active = 0
        self.bind('<Configure>', lambda event: self.draw())
        self.bind('<Button-1>', self.press)
        self.bind('<B1-Motion>', self.drag)
        self.bind('<Left>', lambda event: self.step(-1))
        self.bind('<Right>', lambda event: self.step(1))
        self.bind('<Home>', lambda event: self.command(self.active, 0))
        self.bind('<End>', lambda event: self.command(self.active, self.ceiling))

    def positions(self):
        width = max(1, self.winfo_width() - 24)
        return [12 + value / self.ceiling * width for value in self.values]

    def set_values(self, low, high, ceiling):
        self.values = (low, high)
        self.ceiling = ceiling
        self.draw()

    def draw(self):
        self.delete('all')
        left, right = self.positions()
        self.create_line(12, 15, max(12, self.winfo_width() - 12), 15, fill='#888888', width=4)
        self.create_line(left, 15, right, 15, fill='#61ba86', width=5)
        for index, position in enumerate((left, right)):
            self.create_oval(position - 7, 8, position + 7, 22,
                             fill='#FFFFFF', outline='#404040', width=2)

    def press(self, event):
        self.focus_set()
        positions = self.positions()
        self.active = int(abs(event.x - positions[1]) < abs(event.x - positions[0]))
        self.drag(event)

    def drag(self, event):
        fraction = min(1, max(0, (event.x - 12) / max(1, self.winfo_width() - 24)))
        self.command(self.active, fraction * self.ceiling)

    def step(self, direction):
        self.command(self.active, self.values[self.active] + direction * self.ceiling / 200)
        return 'break'


class ImagePanel(ttk.Frame):
    def __init__(self, master, title, data, cmap='gray', upper=None):
        super().__init__(master, padding=6)
        self.data = np.asarray(data)
        self.rgb = self.data.ndim == 3
        self.initial = display_limits(self.data, upper)
        self.limits = self.initial
        self.ceiling = self.initial[1] * 2
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, text=title, anchor='center').grid(row=0, column=0, sticky='ew', pady=4)
        figure = Figure(figsize=(3, 3), layout='constrained')
        self.axes = figure.add_subplot(111)
        self.axes.set_axis_off()
        shown = self.render_rgb() if self.rgb else np.ma.masked_invalid(self.data)
        self.artist = self.axes.imshow(shown, cmap=cmap, vmin=self.limits[0], vmax=self.limits[1], interpolation='nearest')
        self.canvas = FigureCanvasTkAgg(figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        self.controls = ttk.Frame(self)
        self.controls.grid(row=2, column=0, sticky='ew')
        self.controls.columnconfigure(1, weight=1)
        self.controls.columnconfigure(3, weight=1)
        self.range_bar = RangeBar(self.controls, self.change_range)
        self.range_bar.grid(row=0, column=0, columnspan=4, sticky='ew', pady=3)
        self.range_bar.set_values(*self.limits, self.ceiling)
        self.entries = []
        self.syncing = False
        for index, label in enumerate(('Min', 'Max')):
            ttk.Label(self.controls, text=label).grid(row=1, column=index * 2, padx=(0, 5))
            entry = ttk.Entry(self.controls, width=8)
            entry.grid(row=1, column=index * 2 + 1, padx=5, sticky='ew')
            entry.insert(0, f'{self.limits[index]:.4g}')
            entry.bind('<Return>', lambda event, i=index: self.commit_range(i))
            entry.bind('<FocusOut>', lambda event, i=index: self.commit_range(i))
            self.entries.append(entry)
        buttons = ttk.Frame(self.controls)
        buttons.grid(row=2, column=0, columnspan=4, sticky='ew', pady=3)
        ttk.Button(buttons, text='Auto range', command=self.auto_range).pack(side='left')
        ttk.Button(buttons, text='Reset', command=self.reset_range).pack(side='right')

    def render_rgb(self):
        low, high = self.limits
        values = np.nan_to_num(self.data, nan=0, posinf=high, neginf=0)
        return np.clip((values - low) / (high - low), 0, 1)

    def change_range(self, index, value):
        if self.syncing:
            return
        value = float(value)
        low, high = self.limits
        epsilon = max(high * 1e-6, 1e-9)
        if index == 0:
            low = min(high - epsilon, max(0, value))
        else:
            high = max(low + epsilon, value)
        self.set_range(low, high)

    def commit_range(self, index):
        try:
            value = float(self.entries[index].get())
            if not np.isfinite(value):
                raise ValueError
            self.change_range(index, value)
        except ValueError:
            self.bell()
            self.set_range(*self.limits)

    def set_range(self, low, high):
        self.limits = (low, high)
        self.ceiling = max(self.ceiling, high)
        self.syncing = True
        for entry, value in zip(self.entries, self.limits):
            entry.delete(0, tk.END)
            entry.insert(0, f'{value:.4g}')
        self.range_bar.set_values(*self.limits, self.ceiling)
        self.syncing = False
        if self.rgb:
            self.artist.set_data(self.render_rgb())
        else:
            self.artist.set_clim(low, high)
        self.canvas.draw_idle()

    def auto_range(self):
        x0, x1 = sorted(self.axes.get_xlim())
        y0, y1 = sorted(self.axes.get_ylim())
        bounds = crop_bounds([x0 + 0.5, x1 + 0.5, y0 + 0.5, y1 + 0.5], self.data.shape)
        x0, x1, y0, y1 = bounds
        self.set_range(*display_limits(self.data[y0:y1, x0:x1]))

    def reset_range(self):
        self.set_range(*self.initial)

    def set_zoom(self, bounds):
        x0, x1, y0, y1 = crop_bounds(bounds, self.data.shape)
        self.axes.set_xlim(x0 - 0.5, x1 - 0.5)
        self.axes.set_ylim(y1 - 0.5, y0 - 0.5)
        self.canvas.draw_idle()

    def close(self):
        idle = getattr(self.canvas, '_idle_draw_id', None)
        if idle is not None:
            self.canvas.get_tk_widget().after_cancel(idle)
            self.canvas._idle_draw_id = None


def size_window(window):
    width = min(1400, int(window.winfo_screenwidth() * 0.9))
    height = min(900, int(window.winfo_screenheight() * 0.85))
    window.geometry(f'{width}x{height}')
    window.minsize(min(650, width), min(460, height))
