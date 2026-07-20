import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from   sys                               import platform
import tkinter                           as tk
import matplotlib.pyplot                 as plt
import numpy                             as np
from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from   screeninfo                        import get_monitors

def INTERACT_GUI(original_matrix, organ_of_intrest):
    from   sys                               import platform
    import tkinter                           as tk
    import matplotlib.pyplot                 as plt
    import numpy                             as np
    from   matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
    global hVar1, hVar2, hVar3, hVar4, hVar5, hVar6
    global slc, image_normalization, Next_Button, Finish_Button, Image_Label
    global dummy_scale1_idx, dummy_scale2_idx, dummy_scale3_idx, dummy_scale4_idx
    global min_clim, max_clim, matrix, organ
    global dummy1, dummy_canvas1
    global x_start, x_end, y_start, y_end
    global root, MEDIUMFONT, app_txt_col, app_bkg_col
    global SMALLFONT
    
    matrix = original_matrix
    organ  = organ_of_intrest
    
    if matrix.shape[0] > matrix.shape[1]:
        matrix_ratio = matrix.shape[1] / matrix.shape[0]
        matrix_type = 'SQUARE'
        matrix_type = 'TALL'
    if matrix.shape[1] > matrix.shape[0]:
        matrix_ratio = matrix.shape[0] / matrix.shape[1]
        matrix_type = 'LONG'
    if matrix.shape[0] == matrix.shape[1]:
        matrix_ratio = 1
        matrix_type = 'SQUARE'


    #matrix_ratio = 2 
    slc = 0
    image               = np.max(matrix[:, :, slc, :], axis = 2)
    image_normalization = image / image.max()
    x_start = []
    x_end   = []
    y_start = []
    y_end   = []

    for ii in range(matrix.shape[2]):
        x_start.append([])
        x_end.append([])
        y_start.append([])
        y_end.append([])
    # root window
    root = tk.Tk()

    if platform == 'darwin':
        from tkmacosx import Button
    else:
        from tkinter import Button
    LARGEFONT  = ("Verdana", 35)
    MEDIUMFONT = ("Verdana", 25)
    SMALLFONT  = ("Verdana", 15)
    app_bkg_col    = '#1A2028'    # Dark Blue 1  (Notebook Background)
    app_txt_col    = '#FFEC8E'    # Yellow
    frame_bkg_col1 = '#30394A'    # Dark Blue 2  (Notebook Base 1)
    frame_bkg_col2 = '#363F4E'    # Dark Blue 3  (Notebook Base 2)
    frame_txt_col1 = '#fea47f'    # Orange
    frame_txt_col2 = '#e17e85'    # Light Red

    button_bkg_col  = app_bkg_col
    button_txt_col1 = '#B5C2D9'
    button_txt_col2 = '#B5C2D9'
    button_txt_col3 = '#B5C2D9'




    root.configure(bg = app_bkg_col)

    # Force a 1:1 Tk scaling factor. conda-forge's Tk build has a bug on macOS Retina displays 
    try:
        root.tk.call('tk', 'scaling', 1.0)
    except Exception:
        pass

    # Query the OS directly for monitor size via screeninfo instead of Tk's
    try:
        primary_monitor = next((m for m in get_monitors() if getattr(m, 'is_primary', False)), get_monitors()[0])
        screen_width  = primary_monitor.width
        screen_height = primary_monitor.height
    except Exception:
        root.update_idletasks()
        screen_width  = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
    scale_factor = 0.75

    screen_min = min(screen_width, screen_height)
    if matrix_type == 'SQUARE':
        window_width  = int(screen_min * scale_factor)
        window_height = int(screen_min * scale_factor * matrix_ratio)
    if matrix_type == 'TALL':
        window_width  = int(screen_min * scale_factor)
        window_height = int(screen_min * scale_factor)
    if matrix_type == 'LONG':
        window_width  = int(screen_min * scale_factor)
        window_height = int(screen_min * scale_factor * matrix_ratio)

    geometry_string = str(window_width) + 'x' + str(window_height) + '+0+0'
    root.geometry(geometry_string)
    root.minsize(int(window_width * 0.5), int(window_height * 0.5))
    root.resizable(1, 1)  # allow manual resize as a safety net if auto-sizing is still off
    root.title('INTeractive Enhanced Rectangular Area Cropping Tool')
    root.update_idletasks()
    
    #new
    window_width  = root.winfo_width()
    window_height = root.winfo_height()


    Image_Label = tk.Label(root, font = MEDIUMFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Image_Label.config(text = 'Normalized Maximum Intensity Projection (MIP) for Slice ' + str(slc + 1) +
                              '\nOrgan to Crop: '+ str(organ))
    Image_Label.place(relheight = 0.1,
                      relwidth  = 1.0,
                      relx      = 0.0,
                      rely      = 0.0)

    dummy_figure1 = plt.Figure(facecolor = '#ECECEC', tight_layout = True)
    dummy1        = dummy_figure1.add_subplot(111)
    dummy2        = dummy1.imshow(image_normalization, cmap = 'gray')
    min_clim      = 0
    max_clim      = 1
    dummy2.set_clim([min_clim, max_clim])
    dummy1.set_aspect('equal')
    
    dummy_scale1_idx = 0
    dummy_scale2_idx = matrix.shape[1] - 1
    dummy_scale3_idx = 0
    dummy_scale4_idx = matrix.shape[0] - 1

    dummy1.axvline(x = dummy_scale1_idx, color = '#61ba86', linewidth = 10)
    dummy1.axvline(x = dummy_scale2_idx, color = '#61ba86', linewidth = 10)
    dummy1.axhline(y = dummy_scale3_idx, color = '#be86e3', linewidth = 10)
    dummy1.axhline(y = dummy_scale4_idx, color = '#be86e3', linewidth = 10)

    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)
    dummy_canvas1 = FigureCanvasTkAgg(dummy_figure1, root)
    dummy_canvas1.get_tk_widget().place(relheight = 0.8,
                                        relwidth  = 0.8,
                                        relx      = 0.1,
                                        rely      = 0.1)

    hVar1 = tk.DoubleVar()  # left handle variable
    hVar2 = tk.DoubleVar()  # right handle variable
    hVar3 = tk.DoubleVar()  # left handle variable
    hVar4 = tk.DoubleVar()  # right handle variable
    hVar5 = tk.DoubleVar()  # left handle variable
    hVar6 = tk.DoubleVar()  # right handle variable
    hVar5.set(0)   # ADD THIS — matches initial min_clim
    hVar6.set(1)   # ADD THIS — matches initial max_clim
    # ------------------------------------------------------------------
    # Range controls.
    # ------------------------------------------------------------------
    x_max = matrix.shape[1] - 1
    y_max = matrix.shape[0] - 1

    hVar1.set(0)          # x start (left)
    hVar2.set(x_max)      # x end   (right)
    hVar3.set(0)          # y start (in slider space; inverted by callers)
    hVar4.set(y_max)      # y end   (in slider space; inverted by callers)

    scale_bg = '#ECECEC'
    scale_fg = '#000000'

    # ----- Horizontal pair: x bounds -----
    X_Frame = tk.Frame(root, bg = scale_bg)
    X_Frame.place(relheight = 0.10, relwidth = 0.80, relx = 0.10, rely = 0.90)

    x_lo_scale = tk.Scale(X_Frame, from_ = 0, to = x_max, orient = tk.HORIZONTAL,
                          variable = hVar1, font = SMALLFONT, resolution = 1,
                          bg = scale_bg, fg = scale_fg, highlightthickness = 0,
                          troughcolor = '#61ba86', label = 'X1')
    x_lo_scale.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

    x_hi_scale = tk.Scale(X_Frame, from_ = 0, to = x_max, orient = tk.HORIZONTAL,
                          variable = hVar2, font = SMALLFONT, resolution = 1,
                          bg = scale_bg, fg = scale_fg, highlightthickness = 0,
                          troughcolor = '#61ba86', label = 'X2')
    x_hi_scale.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

    # ----- Vertical pair: y bounds -----
    Y_Frame = tk.Frame(root, bg = scale_bg)
    Y_Frame.place(relheight = 0.80, relwidth = 0.14, relx = 0.00, rely = 0.10)

    y_lo_scale = tk.Scale(Y_Frame, from_ = y_max, to = 0, orient = tk.VERTICAL,
                          variable = hVar3, font = SMALLFONT, resolution = 1,
                          bg = scale_bg, fg = scale_fg, highlightthickness = 0,
                          troughcolor = '#be86e3', label = 'Y1')
    y_lo_scale.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

    y_hi_scale = tk.Scale(Y_Frame, from_ = y_max, to = 0, orient = tk.VERTICAL,
                          variable = hVar4, font = SMALLFONT, resolution = 1,
                          bg = scale_bg, fg = scale_fg, highlightthickness = 0,
                          troughcolor = '#be86e3', label = 'Y2')
    y_hi_scale.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

    # ----- Window/level pair -----
    Limit_Label = tk.Label(root, font = SMALLFONT, fg = scale_fg, bg = scale_bg)
    Limit_Label.config(text = 'Window \nLimit')
    Limit_Label.place(relheight = 0.10, relwidth = 0.10, relx = 0.90, rely = 0.10)

    W_Frame = tk.Frame(root, bg = scale_bg)
    W_Frame.place(relheight = 0.70, relwidth = 0.10, relx = 0.90, rely = 0.20)

    w_lo_scale = tk.Scale(W_Frame, from_ = 1.0, to = 0.0, orient = tk.VERTICAL,
                          variable = hVar5, font = SMALLFONT, resolution = 0.01,
                          bg = scale_bg, fg = scale_fg, highlightthickness = 0,
                          troughcolor = '#808080', label = 'Min')
    w_lo_scale.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

    w_hi_scale = tk.Scale(W_Frame, from_ = 1.0, to = 0.0, orient = tk.VERTICAL,
                          variable = hVar6, font = SMALLFONT, resolution = 0.01,
                          bg = scale_bg, fg = scale_fg, highlightthickness = 0,
                          troughcolor = '#808080', label = 'Max')
    w_hi_scale.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

    hVar1.trace_add('write', update_plots)
    hVar2.trace_add('write', update_plots)
    hVar3.trace_add('write', update_plots)
    hVar4.trace_add('write', update_plots)
    hVar5.trace_add('write', update_plots)
    hVar6.trace_add('write', update_plots)

    Crop_Button = Button(root, text = "Crop", font = SMALLFONT,
                            fg = app_txt_col, bg = app_bkg_col, command = execute_crop)
    Crop_Button.place(relheight = 0.1,
                      relwidth  = 0.1,
                      relx      = 0.0,
                      rely      = 0.9)
    if slc == matrix.shape[2] - 1:
        Finish_Button = Button(root, text = "Exit", font = SMALLFONT,
                               fg = app_txt_col, bg = app_bkg_col, command =lambda: quit_program())
        Finish_Button.place(relheight = 0.1,
                            relwidth  = 0.1,
                            relx      = 0.9,
                            rely      = 0.9)
        Finish_Button["state"] = "disabled"
    else:
        Next_Button = Button(root, text = "Next", font = SMALLFONT,
                             fg = app_txt_col, bg = app_bkg_col, command = next_slice)
        Next_Button.place(relheight = 0.1,
                          relwidth  = 0.1,
                          relx      = 0.9,
                          rely      = 0.9)
        Next_Button["state"] = "disabled"
    root.mainloop()
    return [x_start, x_end, y_start, y_end]

def execute_crop():
    import numpy as np
    global x_start, x_end, y_start, y_end, Next_Button, Finish_Button
    y_start[slc] = image_normalization.shape[0] - 1 - int(np.round(hVar4.get()))
#    print(hVar4.get())
#    print(x_start)
    y_end[slc]   = image_normalization.shape[0] - 1 - int(np.round(hVar3.get()))
#    print(hVar3.get())
#    print(x_end)
    x_start[slc] = int(np.round(hVar1.get()))
#    print(hVar1.get())
#    print(y_start)
    x_end[slc]   = int(np.round(hVar2.get()))
#    print(hVar2.get())
#    print(y_end)
    #Slice_Crop_Coordinates = [x_start, x_end, y_start, y_end]
    if slc == matrix.shape[2] - 1:
        Finish_Button["state"] = "normal"
    else:
        Next_Button["state"] = "normal"
    return x_start, x_end, y_start, y_end

def next_slice():
    import tkinter as tk
    from sys import platform
    import numpy as np
    if platform == 'darwin':
        from tkmacosx import Button
    else:
        from tkinter import Button
    global slc, organ, image_normalization, Next_Button, Finish_Button, Image_Label
    global dummy_scale1_idx, dummy_scale2_idx, dummy_scale3_idx, dummy_scale4_idx
    global dummy1

    slc = slc + 1
    image               = np.max(matrix[:, :, slc, :], axis = 2)
    image_normalization = image / image.max()
    Image_Label.destroy()
    Image_Label = tk.Label(root, font = MEDIUMFONT,
                           fg = app_txt_col, bg = app_bkg_col)
    Image_Label.config(text = 'Normalized Maximum Intensity Projection (MIP) for Slice ' + str(slc + 1) +
                              '\nOrgan to Crop: '+ str(organ))
    Image_Label.place(relheight = 0.1,
                      relwidth  = 1.0,
                      relx      = 0.0,
                      rely      = 0.0)
    Next_Button["state"] = "disabled"
    if slc == matrix.shape[2] - 1:
        Next_Button.destroy()
        Finish_Button = Button(root, text = "Exit", font = SMALLFONT,
                       fg = app_txt_col, bg = app_bkg_col, command =lambda: quit_program())
        Finish_Button.place(relheight = 0.1,
                            relwidth  = 0.1,
                            relx      = 0.9,
                            rely      = 0.9)
        Finish_Button["state"] = "disabled"
    dummy1.cla()
    tmp1 = dummy1.imshow(image_normalization, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])

    dummy1.axvline(x = dummy_scale1_idx, color = '#61ba86', linewidth = 10)
    dummy1.axvline(x = dummy_scale2_idx, color = '#61ba86', linewidth = 10)
    dummy1.axhline(y = dummy_scale3_idx, color = '#be86e3', linewidth = 10)
    dummy1.axhline(y = dummy_scale4_idx, color = '#be86e3', linewidth = 10)

    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)

    dummy_canvas1.draw()

def finish_program():
    root.destroy()

def update_plots(var, index, mode):
    ### Set Figure Plots
    import numpy as np
    global slc, dummy1, dummy_canvas1
    global dummy_scale1_idx, dummy_scale2_idx, dummy_scale3_idx, dummy_scale4_idx
    global min_clim, max_clim
    dummy_scale1_idx = int(np.round(hVar1.get()))
    dummy_scale2_idx = int(np.round(hVar2.get()))
    dummy_scale3_idx = matrix.shape[0] - 1 - int(np.round(hVar3.get()))
    dummy_scale4_idx = matrix.shape[0] - 1 - int(np.round(hVar4.get()))
    min_clim         = hVar5.get()
    max_clim         = hVar6.get()
    dummy1.cla()
    tmp1 = dummy1.imshow(image_normalization, cmap = 'gray')
    tmp1.set_clim([min_clim, max_clim])

    dummy1.axvline(x = dummy_scale1_idx, color = '#61ba86', linewidth = 10)
    dummy1.axvline(x = dummy_scale2_idx, color = '#61ba86', linewidth = 10)
    dummy1.axhline(y = dummy_scale3_idx, color = '#be86e3', linewidth = 10)
    dummy1.axhline(y = dummy_scale4_idx, color = '#be86e3', linewidth = 10)

    dummy1.set_aspect('equal')
    dummy1.axes.xaxis.set_visible(False)
    dummy1.axes.yaxis.set_visible(False)
    dummy1.spines.top.set_visible(False)
    dummy1.spines.left.set_visible(False)
    dummy1.spines.bottom.set_visible(False)
    dummy1.spines.right.set_visible(False)

    dummy_canvas1.draw()

def quit_program():
    root.destroy()