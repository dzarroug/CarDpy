"""Responsive contouring with synchronized zoom and per-image display controls."""
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.interpolate import splprep, splev
from matplotlib.widgets import RectangleSelector
from cardpy.Colormaps import cDTI_Colormaps_Generator
from cardpy.GUI_Tools._runtime import create_window, wait_window
from cardpy.GUI_Tools._image_controls import ImagePanel, crop_bounds, size_window


def insertion_index(points, point):
    """Insert beside the nearest edge without changing contour point order."""
    if len(points) < 3:
        return len(points)
    start = np.asarray(points, dtype=float)
    edge = np.roll(start, -1, axis=0) - start
    length = np.sum(edge * edge, axis=1)
    fraction = np.sum((np.asarray(point) - start) * edge, axis=1) / np.maximum(length, 1e-12)
    projected = start + np.clip(fraction, 0, 1)[:, None] * edge
    return int(np.argmin(np.sum((projected - point) ** 2, axis=1))) + 1


class ContourWindow:
    stages = ('epi', 'endo', 'antRVIP', 'infRVIP')
    labels = ('Epicardium', 'Endocardium', 'Anterior RV insertion', 'Inferior RV insertion')
    colors = ('#61ba86', '#ef7070', '#54d5ef', '#d17ef0')
    active_button_colors = ('#26734d', '#b63842', '#08788a', '#70469a')

    def __init__(self, magnitude, MD, E1, bounds=None, md_max=3.0, point_size=8.0,
                 scan_name=None, initial_contours=None):
        magnitude, MD, E1 = np.asarray(magnitude), np.asarray(MD), np.asarray(E1)
        if magnitude.ndim != 2 or MD.shape != magnitude.shape or E1.shape != magnitude.shape + (3,):
            raise ValueError('Contouring requires matching magnitude / MD images and an E1 image with three components.')
        if not np.isfinite(point_size) or not 3 <= point_size <= 20:
            raise ValueError('Point size must be between 3 and 20.')
        self.point_size = float(point_size)
        self.shape = magnitude.shape
        self.initial_crop = crop_bounds(bounds, self.shape)
        self.current_crop = self.initial_crop.copy()
        self.points = {stage: [] for stage in self.stages}
        self.confirmed = {stage: False for stage in self.stages}
        self.contours = {}
        if initial_contours is not None:
            self.load_contours(initial_contours)
        self.stage = 'epi'
        self.result = None
        self.zoom_mode = False
        self.dragging = None
        self.history = []
        self.pending_layout = None
        self.wide = None
        title = 'CarDpy: Contouring v2'
        if scan_name:
            title += f' — {scan_name}'
        self.window = create_window(title)
        size_window(self.window)
        self.window.protocol('WM_DELETE_WINDOW', self.cancel)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)
        toolbar = ttk.Frame(self.window, padding=8)
        toolbar.grid(row=0, column=0, sticky='ew')
        self.stage_buttons = []
        for column, (stage, label) in enumerate(zip(self.stages, self.labels)):
            button_frame = tk.Frame(toolbar, background='#514c59', bd=2,
                                    relief='raised', cursor='hand2')
            button = tk.Label(button_frame, text=label, background='#514c59',
                              foreground='white', padx=10, pady=9,
                              font=('Verdana', 10, 'bold'), cursor='hand2')
            button.pack(fill='both', expand=True)
            for widget in (button_frame, button):
                widget.bind('<Button-1>', lambda event, s=stage: self.edit(s))
            button_frame.grid(row=0, column=column, padx=3, pady=3, sticky='ew')
            self.stage_buttons.append((button_frame, button))
            toolbar.columnconfigure(column, weight=1)
        ttk.Style(self.window).configure('CardpyConfirm.TButton', font=('Verdana', 12, 'bold'), padding=(12, 10))
        self.confirm_button = ttk.Button(toolbar, command=self.confirm, style='CardpyConfirm.TButton')
        self.confirm_button.grid(row=1, column=0, columnspan=2, padx=3, pady=6, sticky='ew')
        ttk.Button(toolbar, text='Undo edit (d)', command=self.undo).grid(row=1, column=2, padx=3, pady=3, sticky='ew')
        point_controls = ttk.Frame(toolbar)
        point_controls.grid(row=2, column=0, columnspan=4, pady=3)
        ttk.Label(point_controls, text='Point size').pack(side='left', padx=6)
        self.point_size_control = ttk.Spinbox(point_controls, from_=3, to=20, increment=1,
                                              width=5, command=self.change_point_size)
        self.point_size_control.set(f'{self.point_size:g}')
        self.point_size_control.pack(side='left')
        self.point_size_control.bind('<Return>', self.change_point_size)
        self.point_size_control.bind('<FocusOut>', self.change_point_size)
        self.status = ttk.Label(self.window, anchor='center', wraplength=850, font=('Verdana', 11, 'bold'))
        self.status.grid(row=1, column=0, sticky='ew', padx=8, pady=4)
        self.body = ttk.Frame(self.window)
        self.body.grid(row=2, column=0, sticky='nsew')
        self.body.rowconfigure(0, weight=1)
        cmaps = cDTI_Colormaps_Generator()
        self.panels = [
            ImagePanel(self.body, 'Magnitude', np.abs(magnitude)),
            ImagePanel(self.body, 'MD (µm²/ms)', MD, cmap=cmaps['MD'], upper=md_max),
            ImagePanel(self.body, 'Primary eigenvector |E1|', np.abs(E1), upper=1.0),
        ]
        self.artists = [[] for _ in self.panels]
        for panel in self.panels:
            panel.canvas.mpl_connect('button_press_event', self.click)
            panel.canvas.mpl_connect('key_press_event', self.key)
            panel.canvas.mpl_connect('motion_notify_event', self.move_point)
            panel.canvas.mpl_connect('button_release_event', self.release_point)
        self.zoom_selector = RectangleSelector(
            self.panels[0].axes, self.select_zoom, button=[1], interactive=False,
            minspanx=1, minspany=1, spancoords='data', useblit=False,
            props={'edgecolor': '#61ba86', 'facecolor': '#61ba86', 'alpha': 0.25})
        self.zoom_selector.set_active(False)
        footer = ttk.Frame(self.window, padding=8)
        footer.grid(row=3, column=0, sticky='ew')
        ttk.Button(footer, text='Cancel', command=self.cancel).pack(side='left')
        ttk.Button(footer, text='Full image', command=lambda: self.set_zoom(None)).pack(side='left', padx=6)
        ttk.Button(footer, text='Reset to crop', command=lambda: self.set_zoom(self.initial_crop)).pack(side='left', padx=6)
        ttk.Button(footer, text='Draw zoom box', command=self.start_zoom).pack(side='left', padx=6)
        self.panel_choice = ttk.Combobox(footer, state='readonly', width=23,
                                       values=('Magnitude', 'MD', 'Primary eigenvector'))
        self.panel_choice.current(0)
        self.panel_choice.bind('<<ComboboxSelected>>', lambda event: self.layout())
        self.finish_button = ttk.Button(footer, text='Confirm segmentation & Finish' if initial_contours is not None else 'Save contours & Finish', command=self.finish, state='disabled')
        self.finish_button.pack(side='right')
        self.window.bind('<Configure>', self.resize)
        self.window.bind('<ButtonRelease-1>', self.release_point, add='+')
        self.set_zoom(self.initial_crop)
        self.layout()
        self.draw_points()
        self.update_status('Saved segmentation loaded. Check alignment, drag points to adjust, or select Confirm segmentation & Finish.' if initial_contours is not None else None)

    def load_contours(self, initial_contours):
        """Keep exact saved curves; show a manageable number of edit handles."""
        for stage in self.stages:
            points = np.asarray(initial_contours[stage], dtype=float)
            minimum = 4 if stage in ('epi', 'endo') else 1
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum or not np.isfinite(points).all():
                raise ValueError(f'Invalid saved {stage} coordinates.')
            if stage in ('epi', 'endo'):
                self.contours[stage] = points.copy()
                handles = points[:-1] if np.allclose(points[0], points[-1]) else points
                if len(handles) < 4:
                    raise ValueError(f'Saved {stage} must contain four distinct contour points.')
                indices = np.linspace(0, len(handles), min(20, len(handles)), endpoint=False, dtype=int)
                self.points[stage] = [tuple(point) for point in handles[indices]]
            else:
                if len(points) != 1:
                    raise ValueError(f'Saved {stage} must contain exactly one insertion point.')
                self.points[stage] = [tuple(points[0])]

    def change_point_size(self, event=None):
        try:
            size = float(self.point_size_control.get())
            if not np.isfinite(size):
                raise ValueError
            self.point_size = min(20, max(3, size))
        except ValueError:
            pass
        self.point_size_control.set(f'{self.point_size:g}')
        self.draw_points()

    def resize(self, event):
        if event.widget is self.window and self.pending_layout is None:
            self.pending_layout = self.window.after(60, self.layout)

    def layout(self):
        if self.pending_layout is not None:
            self.window.after_cancel(self.pending_layout)
        self.pending_layout = None
        wide = self.window.winfo_width() >= 1050
        if wide:
            self.panel_choice.pack_forget()
        else:
            self.panel_choice.pack(side='left', padx=8)
        selected = self.panel_choice.current()
        for index, panel in enumerate(self.panels):
            self.body.columnconfigure(index, weight=1 if wide else (1 if index == 0 else 0))
            if wide or selected == index:
                panel.grid(row=0, column=index if wide else 0, sticky='nsew')
            else:
                panel.grid_remove()
        self.wide = wide

    def update_status(self, message=None):
        label = self.labels[self.stages.index(self.stage)]
        count = len(self.points[self.stage])
        for (button_frame, button), stage, name in zip(self.stage_buttons, self.stages, self.labels):
            marker = '▶ ' if stage == self.stage else ('✓ ' if self.confirmed[stage] else '')
            active = stage == self.stage
            color = self.active_button_colors[self.stages.index(stage)] if active else '#514c59'
            button_frame.configure(background=color, relief='sunken' if active else 'raised',
                                   bd=3 if active else 2)
            button.configure(text=marker + name, background=color,
                             foreground='white' if active else '#f4f1f7')
        self.confirm_button.configure(text='Confirm ' + label)
        text = message or f'{label}: {count} point(s). Drag points to move; click to add. Confirm when ready.'
        self.status.configure(text=text)
        complete = self.has_complete_contours()
        self.finish_button.configure(state='normal' if complete else 'disabled')

    def has_complete_contours(self):
        return (len(self.points['epi']) >= 4 and len(self.points['endo']) >= 4
                and bool(self.points['antRVIP']) and bool(self.points['infRVIP']))

    def edit(self, stage):
        self.dragging = None
        self.zoom_mode = False
        self.zoom_selector.set_active(False)
        self.stage = stage
        self.draw_points()
        self.update_status()

    def remember(self):
        self.history.append((self.stage, self.points[self.stage].copy(),
                             self.contours.get(self.stage), self.confirmed[self.stage]))
        self.confirmed[self.stage] = False
        self.contours.pop(self.stage, None)

    def nearest_point(self, event):
        if not hasattr(event, 'x') or not hasattr(event, 'y'):
            return None
        nearest, distance = None, max(10.0, self.point_size * event.inaxes.figure.dpi / 144 + 3)
        for stage in self.stages:
            for index, point in enumerate(self.points[stage]):
                pixel = event.inaxes.transData.transform(point)
                delta = np.linalg.norm(pixel - [event.x, event.y])
                if delta < distance:
                    nearest, distance = (stage, index), delta
        return nearest

    def click(self, event):
        if self.zoom_mode or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        if not any(event.inaxes is panel.axes for panel in self.panels):
            return
        x, y = float(event.xdata), float(event.ydata)
        if not (0 <= x < self.shape[1] and 0 <= y < self.shape[0]):
            return
        nearest = self.nearest_point(event)
        if nearest is not None:
            self.edit(nearest[0])
            self.remember()
            self.dragging = nearest
            return
        if self.confirmed[self.stage]:
            self.edit(self.stage)
        self.remember()
        if self.stage in ('antRVIP', 'infRVIP'):
            self.points[self.stage] = [(x, y)]
        else:
            index = insertion_index(self.points[self.stage], (x, y))
            self.points[self.stage].insert(index, (x, y))
        self.draw_points()
        self.update_status()

    def move_point(self, event):
        if self.dragging is None or event.xdata is None or event.ydata is None:
            return
        if not any(event.inaxes is panel.axes for panel in self.panels):
            return
        x = min(self.shape[1] - 1, max(0, float(event.xdata)))
        y = min(self.shape[0] - 1, max(0, float(event.ydata)))
        stage, index = self.dragging
        self.points[stage][index] = (x, y)
        self.draw_points()

    def release_point(self, event):
        self.dragging = None
        self.update_status()

    def key(self, event):
        if event.key in ('d', 'ctrl+z', 'cmd+z'):
            self.undo()

    def undo(self):
        if self.history:
            stage, points, contour, confirmed = self.history.pop()
            self.edit(stage)
            self.confirmed[stage] = confirmed
            self.contours.pop(stage, None)
            if contour is not None:
                self.contours[stage] = contour
            self.points[stage] = points
            self.dragging = None
            self.draw_points()
            self.update_status()

    def fit_contour(self, stage):
        points = self.points[stage]
        if len(points) < 4:
            raise ValueError('Pick at least four spread-out contour points.')
        closed = np.asarray(points + [points[0]], dtype=float)
        tck, _ = splprep(closed.T, s=0)
        return np.column_stack(splev(np.linspace(0, 1, 200), tck))

    def confirm(self):
        if self.zoom_mode:
            self.update_status('Finish drawing the zoom box before confirming points.')
            return
        try:
            if self.stage in ('epi', 'endo'):
                if self.stage not in self.contours:
                    self.contours[self.stage] = self.fit_contour(self.stage)
            elif not self.points[self.stage]:
                raise ValueError('Pick the RV insertion point before confirming.')
        except (ValueError, TypeError):
            self.update_status('Pick at least four distinct contour points, or one RV insertion point.')
            return
        self.confirmed[self.stage] = True
        self.draw_points()
        self.update_status('Stage saved. Select any contour or insertion point to continue editing.')

    def draw_points(self):
        for index, panel in enumerate(self.panels):
            for artist in self.artists[index]:
                artist.remove()
            self.artists[index] = []
            for stage, color in zip(self.stages, self.colors):
                points = self.points[stage]
                if points:
                    points = np.asarray(points)
                    self.artists[index].extend(panel.axes.plot(points[:, 0], points[:, 1], 'o', color=color, markersize=self.point_size))
                contour = self.contours.get(stage)
                if contour is None and stage in ('epi', 'endo') and len(self.points[stage]) >= 4:
                    try:
                        contour = self.fit_contour(stage)
                    except (ValueError, TypeError):
                        pass
                if contour is not None:
                    self.artists[index].extend(panel.axes.plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5))
            # Keep fixed full-image coordinates even while the view is zoomed.
            panel.set_zoom(self.current_crop)

    def start_zoom(self):
        self.dragging = None
        self.zoom_mode = True
        self.zoom_selector.set_active(True)
        self.panel_choice.current(0)
        self.layout()
        self.update_status('Drag a zoom box on the magnitude image. Point selection resumes afterward.')

    def select_zoom(self, press, release):
        x0, x1, y0, y1 = self.zoom_selector.extents
        self.set_zoom([x0 + 0.5, x1 + 0.5, y0 + 0.5, y1 + 0.5])
        self.zoom_selector.set_visible(False)
        self.update_status()

    def set_zoom(self, bounds):
        self.current_crop = crop_bounds(bounds, self.shape)
        self.zoom_mode = False
        self.zoom_selector.set_active(False)
        for panel in self.panels:
            panel.set_zoom(self.current_crop)

    def finish(self):
        if not self.has_complete_contours():
            self.update_status('Draw both contours and place both RV insertion points before finishing.')
            return
        try:
            endo = self.contours.get('endo')
            epi = self.contours.get('epi')
            if endo is None:
                endo = self.fit_contour('endo')
            if epi is None:
                epi = self.fit_contour('epi')
        except (ValueError, TypeError):
            self.update_status('Pick at least four distinct points on each contour before finishing.')
            return
        self.result = [endo[:, 0].tolist(), endo[:, 1].tolist(),
                       epi[:, 0].tolist(), epi[:, 1].tolist(),
                       self.points['antRVIP'][-1], self.points['infRVIP'][-1]]
        self.close()

    def close(self):
        if self.pending_layout is not None:
            self.window.after_cancel(self.pending_layout)
        self.zoom_selector.disconnect_events()
        for panel in self.panels:
            panel.close()
        self.window.destroy()

    def cancel(self):
        self.result = None
        self.close()


def New_GUI(avg_diff_image, ADC_image, E1_image, crop_bounds=None, md_max=3.0,
            point_size=8.0, scan_name=None, initial_contours=None):
    app = ContourWindow(avg_diff_image, ADC_image, E1_image, bounds=crop_bounds,
                        md_max=md_max, point_size=point_size, scan_name=scan_name,
                        initial_contours=initial_contours)
    wait_window(app.window)
    if app.result is None:
        raise RuntimeError('Contour selection cancelled.')
    return app.result
