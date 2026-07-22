import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cardpy.Colormaps import cDTI_Colormaps_Generator

cDTI_cmaps = cDTI_Colormaps_Generator()

def New_GUI(avg_diff_image, ADC_image, E1_image):

    # import numpy as np
    # import tkinter as tk
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import Normalize
    from scipy.interpolate import splprep, splev

    from   sys               import platform
    import tkinter as tk
    from matplotlib.colors import Normalize
    from PIL import Image, ImageTk
    if platform == 'darwin':
        from tkmacosx import Button
    else:
        from tkinter import Button

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

    class ImagePlotter(tk.Tk):
        from   sys               import platform
        import numpy as np
        from PIL import Image, ImageTk
        import matplotlib.pyplot as plt
        from scipy.interpolate import splprep, splev

    #     var = tk.IntVar()


    #     app_bkg_col    = '#1A2028'    # Dark Blue 1  (Notebook Background)
    #     app_txt_col    = '#FFEC8E'    # Yellow
    #     frame_bkg_col1 = '#30394A'    # Dark Blue 2  (Notebook Base 1)
    #     frame_bkg_col2 = '#363F4E'    # Dark Blue 3  (Notebook Base 2)
    #     frame_txt_col1 = '#fea47f'    # Orange
    #     frame_txt_col2 = '#e17e85'    # Light Red

    #     button_bkg_col  = app_bkg_col
    #     button_txt_col1 = '#B5C2D9'
    #     button_txt_col2 = '#B5C2D9'
    #     button_txt_col3 = '#B5C2D9'

        def __init__(self):
            super().__init__()
            self.title("CarDpy: Beta GUI")
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
            self.config(bg = app_bkg_col)
            # Create a frame for buttons and instructions
            self.left_frame = tk.Frame(self)
            self.left_frame.grid(row = 0, column = 0, sticky="nsew")
            self.left_frame.config(bg = app_bkg_col)

            self.create_canvases()  # Now called before adding buttons and instructions
            self.add_buttons_and_instructions()  # Now added after creating canvases

            self.initialize_images()
            self.plot_images()
            self.epicardium_mouse_clicks  = []  # List to store epicardium mouse click data
            self.is_epicardium_active     = True
            self.epicardium_spline_id     = None  # ID for the epicardium spline line on the canvas
            self.epicardium_spline_data   = []

            self.endocardium_mouse_clicks = []  # List to store endocardium mouse click data
            self.is_endocardium_active    = False
            self.endocardium_spline_id    = None  # ID for the endocardium spline line on the canvas
            self.endocardium_spline_data  = []

            self.is_antRVIP_active        = False    
            self.antRVIP_mouse_clicks     = []
            self.Anterior_RVIP            = []

            self.is_infRVIP_active        = False  
            self.infRVIP_mouse_clicks     = []
            self.Inferior_RVIP            = []


            # Bind the "d" key to the delete_previous_point method
            self.bind("<KeyPress-d>", self.delete_previous_point)

        def add_buttons_and_instructions(self):
            # Add buttons for functionality
            # Create the Confirm Epicardium Points button
            confirm_epicardium_button = Button(self.left_frame, 
                                               text = "Confirm Epicardium Points", 
                                               font = ("Verdana", 20),
                                               fg = 'light green',
                                               bg = app_bkg_col,
                                               command = self.confirm_epicardium)
            confirm_epicardium_button.pack(pady = 10)
            # Create the Confirm Endocardium Points button
            confirm_endocardium_button = Button(self.left_frame, 
                                                text    = "Confirm Endocardium Points", 
                                                font    = ("Verdana", 20),
                                                fg      = 'red',
                                                bg      = app_bkg_col,
                                                command = self.confirm_endocardium)
            confirm_endocardium_button.pack(pady = 10)
            # Create the Confirm Anterior RV Insertion Point button
            confirm_AntRVIP_button = Button(self.left_frame, 
                                                text    = "Confirm Anterior Insertion Point", 
                                                font    = ("Verdana", 20),
                                                fg      = 'cyan',
                                                bg      = app_bkg_col,
                                                command = self.confirm_antRVIP)
            confirm_AntRVIP_button.pack(pady = 10)
            # Create the Confirm Inferior RV Insertion Point button
            confirm_InfRVIP_button = Button(self.left_frame, 
                                                text    = "Confirm Inferior Insertion Point", 
                                                font    = ("Verdana", 20),
                                                fg      = 'magenta',
                                                bg      = app_bkg_col,
                                                command = self.confirm_infRVIP)
            confirm_InfRVIP_button.pack(pady = 10)        
            # Create the Delte Previous Point button
            delete_previous_point_button = Button(self.left_frame, 
                                                  text = "Delete Last Point (d)", 
                                                  font    = ("Verdana", 20),
                                                  fg      = button_txt_col3,
                                                  bg      = app_bkg_col,
                                                  command = self.delete_previous_point)
            delete_previous_point_button.pack(pady = 10)
            # Add instructions
            instructions_label = tk.Label(self.left_frame, 
                                          text = "Instructions:\nStep 1) Use the cDTI images to pick at least 4 epicardial points.\nStep 2) Select Confirm Epicardial Points button when completed.\nStep 3) Use the cDTI images to pick at least 4 endocardial points.\nStep 4) Select Confirm Endocardial Points button when completed.\n\nNotes:\n "                                     )
            instructions_label.pack(pady = 10)

        def delete_previous_point(self, event=None):
            ### Epi
            if self.is_epicardium_active == True:
                if self.epicardium_mouse_clicks:
                    # Remove the last point from the stored clicks
                    deleted_epicardium_point = self.epicardium_mouse_clicks.pop()

                    # Remove the markers from all canvases
                    for canvas in self.canvases:
                        canvas.delete("epi_point_{}".format(len(self.epicardium_mouse_clicks) + 1))  # Unique tag for each point

                    # Clear existing epicardium spline
                    self.clear_epicardium_spline()

                    # Redraw epicardium spline if there are at least four points
                    if len(self.epicardium_mouse_clicks) >= 4:
                        self.draw_epicardium_spline()
            ### Endo
            if self.is_endocardium_active == True:
                if self.endocardium_mouse_clicks:
                    # Remove the last point from the stored clicks
                    deleted_endocardium_point = self.endocardium_mouse_clicks.pop()

                    # Remove the markers from all canvases
                    for canvas in self.canvases:
                        canvas.delete("endo_point_{}".format(len(self.endocardium_mouse_clicks) + 1))  # Unique tag for each point

                    # Clear existing epicardium spline
                    self.clear_endocardium_spline()

                    # Redraw epicardium spline if there are at least four points
                    if len(self.endocardium_mouse_clicks) >= 4:
                        self.draw_endocardium_spline()

        def confirm_epicardium(self):
            if len(self.epicardium_mouse_clicks) >= 4:
                first_point = self.epicardium_mouse_clicks[0]
                self.epicardium_mouse_clicks.append(first_point)

                # Plot a marker at the first point on all canvases
                for canvas, data in zip(self.canvases, self.data):
                    scaled_x = (first_point[0] * (canvas.winfo_width() / data.shape[1]))
                    scaled_y = (first_point[1] * (canvas.winfo_height() / data.shape[0]))
                    canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5, fill = "light green", tags = "epi_point_{}".format(len(self.epicardium_mouse_clicks)))

                # Clear existing spline and draw new spline
                self.clear_epicardium_spline()
                self.draw_epicardium_spline()
                self.is_epicardium_active  = False
                self.is_endocardium_active = True
                self.is_antRVIP_active     = False
                self.is_infRVIP_active     = False
                # Call get_spline_data to retrieve the spline data
                spline_data0 = self.get_epicardium_spline_data()
    #             print("Endocardium Spline Data:", spline_data0)
    #             app.get_epicardium_spline_data()

        def confirm_endocardium(self):
            if len(self.endocardium_mouse_clicks) >= 4:
                first_point = self.endocardium_mouse_clicks[0]
                self.endocardium_mouse_clicks.append(first_point)

                # Plot a marker at the first point on all canvases
                for canvas, data in zip(self.canvases, self.data):
                    scaled_x = (first_point[0] * (canvas.winfo_width() / data.shape[1]))
                    scaled_y = (first_point[1] * (canvas.winfo_height() / data.shape[0]))
                    canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5, fill = "red", tags = "endo_point_{}".format(len(self.endocardium_mouse_clicks)))

                # Clear existing spline and draw new spline
                self.clear_endocardium_spline()
                self.draw_endocardium_spline()
                self.is_epicardium_active  = False
                self.is_endocardium_active = False
                self.is_antRVIP_active     = True
                self.is_infRVIP_active     = False
                # Call get_spline_data to retrieve the spline data
                spline_data1 = self.get_endocardium_spline_data()
    #             print("Epicardium Spline Data:", spline_data1)
    #             endocardium_spline = self.get_endocardium_spline_data()

                # Optionally, you can do something with the spline_data, such as updating a display or saving it.
        def confirm_antRVIP(self):
            # Plot a marker at the first point on all canvases
    #         for canvas, data in zip(self.canvases, self.data):
    #             scaled_x = int(first_point[0] * (canvas.winfo_width() / data.shape[1]))
    #             scaled_y = int(first_point[1] * (canvas.winfo_height() / data.shape[0]))
    #             canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5, fill = "cyan", tags = "antRVIP_point_{}".format(len(self.antRVIP_mouse_clicks)))

            # Clear existing spline and draw new spline
            self.is_epicardium_active  = False
            self.is_endocardium_active = False
            self.is_antRVIP_active     = False
            self.is_infRVIP_active     = True
            self.Anterior_RVIP         = self.antRVIP_mouse_clicks[-1]

        def confirm_infRVIP(self):
            # Plot a marker at the first point on all canvases
    #         for canvas, data in zip(self.canvases, self.data):
    #             scaled_x = int(first_point[0] * (canvas.winfo_width() / data.shape[1]))
    #             scaled_y = int(first_point[1] * (canvas.winfo_height() / data.shape[0]))
    #             canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5, fill = "cyan", tags = "infRVIP_point_{}".format(len(self.infRVIP_mouse_clicks)))

            # Clear existing spline and draw new spline
            self.is_epicardium_active  = False
            self.is_endocardium_active = False
            self.is_antRVIP_active     = False
            self.is_infRVIP_active     = False
            self.Inferior_RVIP         = self.infRVIP_mouse_clicks[-1]



        def clear_epicardium_spline(self):
            if self.epicardium_spline_id is not None:
                for canvas in self.canvases:
                    canvas.delete(self.epicardium_spline_id)
                self.epicardium_spline_id = None
        def clear_endocardium_spline(self):
            if self.endocardium_spline_id is not None:
                for canvas in self.canvases:
                    canvas.delete(self.endocardium_spline_id)
                self.endocardium_spline_id  = None

        def draw_epicardium_spline(self):
            if len(self.epicardium_mouse_clicks) >= 4:
                x_points, y_points = zip(*self.epicardium_mouse_clicks)
                tck, _             = splprep([x_points, y_points], s = 0)  # Spline parameters, s=0 for interpolation
                u                  = np.linspace(0, 1, num = 1000)
                x_spline, y_spline = splev(u, tck)

                # Convert spline coordinates to image matrix coordinates
                canvas_width              = self.canvases[0].winfo_width()
                canvas_height             = self.canvases[0].winfo_height()
                image_width, image_height = self.data[0].shape[::-1]
                x_ratio                   = canvas_width / image_width
                y_ratio                   = canvas_height / image_height
                scaled_x_spline           = [(x * x_ratio) for x in x_spline]
                scaled_y_spline           = [(y * y_ratio) for y in y_spline]

                # Zip scaled_x_spline and scaled_y_spline to provide pairs of coordinates for create_line
                spline_epicardium_coords = list(zip(scaled_x_spline, scaled_y_spline))
                for canvas in self.canvases:
                    canvas.create_line(spline_epicardium_coords, fill = "light green", width = 2, tags = "epi_spline")
                self.epicardium_spline_id = "epi_spline"

        def draw_endocardium_spline(self):
            if len(self.endocardium_mouse_clicks) >= 4:
                x_points, y_points = zip(*self.endocardium_mouse_clicks)
                tck, _             = splprep([x_points, y_points], s = 0)  # Spline parameters, s=0 for interpolation
                u                  = np.linspace(0, 1, num = 1000)
                x_spline, y_spline = splev(u, tck)

                # Convert spline coordinates to image matrix coordinates
                canvas_width              = self.canvases[0].winfo_width()
                canvas_height             = self.canvases[0].winfo_height()
                image_width, image_height = self.data[0].shape[::-1]
                x_ratio                   = canvas_width / image_width
                y_ratio                   = canvas_height / image_height
                scaled_x_spline           = [(x * x_ratio) for x in x_spline]
                scaled_y_spline           = [(y * y_ratio) for y in y_spline]

                # Zip scaled_x_spline and scaled_y_spline to provide pairs of coordinates for create_line
                spline_endocardium_coords = list(zip(scaled_x_spline, scaled_y_spline))

                for canvas in self.canvases:
                    canvas.create_line(spline_endocardium_coords, fill = "red", width = 2, tags = "endo_spline")
                self.endocardium_spline_id = "endo_spline"
        def get_epicardium_spline_data(self):
            if len(self.epicardium_mouse_clicks) >= 4:
                x_points, y_points = zip(*self.epicardium_mouse_clicks)
                tck, _             = splprep([x_points, y_points], s = 0)  # Spline parameters, s=0 for interpolation
                u                  = np.linspace(0, 1, num = 200)
                x_spline, y_spline = splev(u, tck)

                # Convert spline coordinates to image matrix coordinates
                canvas_width              = self.canvases[0].winfo_width()
                canvas_height             = self.canvases[0].winfo_height()
                image_width, image_height = self.data[0].shape[::-1]
                x_ratio                   = image_width / canvas_width 
                y_ratio                   = image_height / canvas_height 
                scaled_x_spline           = [(x * x_ratio) for x in x_spline]
                scaled_y_spline           = [(y * y_ratio) for y in y_spline]

                # Zip scaled_x_spline and scaled_y_spline to provide pairs of coordinates for create_line
                spline_epicardium_coords = list(zip(x_spline, y_spline))
                self.epicardium_spline_data.append(spline_epicardium_coords)
            else:
                self.epicardium_spline_data.append(None)
            return self.epicardium_spline_data
        def get_endocardium_spline_data(self):            
            if len(self.endocardium_mouse_clicks) >= 4:
                x_points, y_points = zip(*self.endocardium_mouse_clicks)
                tck, _             = splprep([x_points, y_points], s = 0)  # Spline parameters, s=0 for interpolation
                u                  = np.linspace(0, 1, num = 200)
                x_spline, y_spline = splev(u, tck)
    #             print(x_spline, y_spline)

                # Convert spline coordinates to image matrix coordinates
                canvas_width              = self.canvases[0].winfo_width()
                canvas_height             = self.canvases[0].winfo_height()
                image_width, image_height = self.data[0].shape[::-1]
                x_ratio                   = image_width / canvas_width
                y_ratio                   = image_height / canvas_height
                scaled_x_spline           = [(x * x_ratio) for x in x_spline]
                scaled_y_spline           = [(y * y_ratio) for y in y_spline]

                # Zip scaled_x_spline and scaled_y_spline to provide pairs of coordinates for create_line
                spline_endocardium_coords = list(zip(x_spline, y_spline))
                self.endocardium_spline_data.append(spline_endocardium_coords)
            else:
                self.endocardium_spline_data.append(None)
            return self.endocardium_spline_data

        def on_canvas_click(self, event):
            canvas = event.widget  # Get the canvas where the click occurred
            idx = self.canvases.index(canvas)  # Get the index of the canvas in the list

            # Convert monitor coordinates to image matrix coordinates
            canvas_width  = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            image_width, image_height = self.data[idx].shape[:2]  # Get the shape of the data

            x_ratio = image_width / canvas_width
            y_ratio = image_height / canvas_height

            image_x = (event.x * x_ratio)
            image_y = (event.y * y_ratio)

            ### Epi
            if self.is_epicardium_active == True:
                # Append the click coordinates and timestamp to the list
                self.epicardium_mouse_clicks.append((image_x, image_y))

                # Plot a marker at the clicked coordinates on all canvases
                for canvas, data in zip(self.canvases, self.data):
                    scaled_x = (image_x * (canvas.winfo_width() / image_width))
                    scaled_y = (image_y * (canvas.winfo_height() / image_height))
                    canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5,
                                       fill = "light green", 
                                       tags = "epi_point_{}".format(len(self.epicardium_mouse_clicks)))

                # Clear existing spline and draw new spline if there are at least four points
                self.clear_epicardium_spline()
                if len(self.epicardium_mouse_clicks) >= 4:
                    self.draw_epicardium_spline()
            ### Endo
            if self.is_endocardium_active == True:
                # Append the click coordinates and timestamp to the list
                self.endocardium_mouse_clicks.append((image_x, image_y))

                # Plot a marker at the clicked coordinates on all canvases
                for canvas, data in zip(self.canvases, self.data):
                    scaled_x = (image_x * (canvas.winfo_width() / image_width))
                    scaled_y = (image_y * (canvas.winfo_height() / image_height))
                    canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5,
                                       fill = "red", 
                                       tags = "endo_point_{}".format(len(self.endocardium_mouse_clicks)))

                # Clear existing spline and draw new spline if there are at least four points
                self.clear_endocardium_spline()
                if len(self.endocardium_mouse_clicks) >= 4:
                    self.draw_endocardium_spline()
            ### Anterior RVIP
            if self.is_antRVIP_active == True:
                # Append the click coordinates and timestamp to the list
                self.antRVIP_mouse_clicks.append((image_x, image_y))

                # Plot a marker at the clicked coordinates on all canvases
                for canvas, data in zip(self.canvases, self.data):
                    scaled_x = (image_x * (canvas.winfo_width() / image_width))
                    scaled_y = (image_y * (canvas.winfo_height() / image_height))
                    canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5,
                                       fill = "cyan", 
                                       tags = "antRVIP_point_{}".format(len(self.antRVIP_mouse_clicks)))  
            ### Inferior RVIP
            if self.is_infRVIP_active == True:
                # Append the click coordinates and timestamp to the list
                self.infRVIP_mouse_clicks.append((image_x, image_y))

                # Plot a marker at the clicked coordinates on all canvases
                for canvas, data in zip(self.canvases, self.data):
                    scaled_x = (image_x * (canvas.winfo_width() / image_width))
                    scaled_y = (image_y * (canvas.winfo_height() / image_height))
                    canvas.create_oval(scaled_x - 5, scaled_y - 5, scaled_x + 5, scaled_y + 5,
                                       fill = "magenta", 
                                       tags = "infRVIP_point_{}".format(len(self.infRVIP_mouse_clicks)))   

        def initialize_images(self):
            self.data      = [avg_diff_image, ADC_image, np.abs(E1_image)]
            self.titles    = ["Mean Diffusion Image", "Mean Diffusivity Map", "Primary Eigenvector Map"]
            self.colormaps = [
                              ("gray", Normalize(vmin=self.data[0].min(), vmax=self.data[0].max() * 0.75)),
                              (cDTI_cmaps['MD'], Normalize(vmin=self.data[1].min(), vmax=self.data[1].max())),
                              (None, None)
                             ]

        def create_canvases(self):
            screen_width  = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()

            shorter_dimension = min(screen_width, screen_height)
            canvas_size       = int(shorter_dimension // 2.75)  # Adjust canvas size based on the shorter dimension

            self.canvases = []
            for idx, (row, col) in enumerate([(0, 1), (1, 0), (1, 1)]):
                canvas = tk.Canvas(self, width=canvas_size, height=canvas_size, bg="white")
                canvas.grid(row=row, column=col, padx=10, pady=10)
                self.canvases.append(canvas)

                # Create a label for the title and place it above the canvas by spanning multiple columns
                self.titles = ["Average Diffusion Image", "Mean Diffusivity Map", "Primary Eigenvalue Map"]
                title_label = tk.Label(self, 
                                       font = ("Verdana", 22, "bold"),
                                       text = self.titles[idx],
                                       fg = app_txt_col,
                                       bg = app_bkg_col)
                title_label.grid(row = row, column = col, columnspan = 1, pady = (0, canvas_size + 40))

                # Bind mouse click event to all canvases
                canvas.bind("<Button-1>", self.on_canvas_click)

        def plot_images(self):
            for idx, (canvas, data, (cmap_name, norm)) in enumerate(zip(self.canvases, self.data, self.colormaps)):
                if cmap_name is not None:
                    cmap = plt.get_cmap(cmap_name)
                    colored_data = cmap(norm(data))
                else:
                    colored_data = data  # No colormap needed for RGB data
                self.plot_image(canvas, colored_data)

        def plot_image(self, canvas, data):
            def plot_delayed():
                img = Image.fromarray((data * 255).astype(np.uint8))
                img = img.resize((canvas.winfo_width(), canvas.winfo_height()), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                canvas.image = photo

            # Schedule the plotting after a short delay to ensure the canvas is fully rendered
            self.after(100, plot_delayed)

    
    app = ImagePlotter()
    app.mainloop()
    endocardium_x  = [x[0] for x in app.endocardium_spline_data[0]]
    endocardium_y  = [x[1] for x in app.endocardium_spline_data[0]]
    epicardium_x   = [x[0] for x in app.epicardium_spline_data[0]]
    epicardium_y   = [x[1] for x in app.epicardium_spline_data[0]]
    Anterior_RVIP  = app.Anterior_RVIP
    Inferior_RVIP  = app.Inferior_RVIP
    return[endocardium_x, endocardium_y, epicardium_x, epicardium_y, Anterior_RVIP, Inferior_RVIP]