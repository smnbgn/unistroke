import tkinter as tk
import math
from tkinter import simpledialog, filedialog, messagebox, PhotoImage
import csv
import symb
import os
import numpy
from network import NeuralNetwork


class NeuralDraw:

    default_width = 800
    default_height = 500
    file_name = None

    points = []
    training_set = []

    def create_icons(self):
        self.new_file_icon = PhotoImage(file='icons/new_file.gif')
        self.open_file_icon = PhotoImage(file='icons/open_file.gif')
        self.save_file_icon = PhotoImage(file='icons/save.gif')
        self.save_file_as_icon = PhotoImage(file='icons/save_as.gif')

    def __init__(self, parent, title):
        self.parent = parent
        self.create_gui()
        self.bind_mouse()
        self.translation = symb.SymbDict()
        self.title = title

    def create_gui(self):
        self.create_icons()
        self.create_menu_bar()
        self.create_top_bar()
        self.create_drawing_canvas()
        self.create_status_bar()

    def create_menu_bar(self):
        self.menu_bar = tk.Menu(self.parent)
        self.create_file_menu()
        self.create_mode_menu()
        self.create_algorithm_menu()

    def create_top_bar(self):
        self.top_bar = tk.Frame(self.parent, height=30, relief='raised', padx=2, pady=2, bg="light gray")
        vec_size_lb = tk.Label(self.top_bar, text="Vector size: ", bg="light gray")
        vec_size_lb.pack(side="left")

        self.vec_size_var = tk.IntVar()
        self.vec_size_var.set(20)
        self.vec_size_en = tk.Entry(self.top_bar, width="3", text=self.vec_size_var, justify="center")
        self.vec_size_en.pack(side="left", padx=2)

        hidden_layer_lb = tk.Label(self.top_bar, text="Hidden layer nodes: ", bg="light gray")
        hidden_layer_lb.pack(side="left")

        self.hidden_layer_var = tk.IntVar()
        self.hidden_layer_var.set(100)
        self.hidden_layer_en = tk.Entry(self.top_bar, width="3", text=self.hidden_layer_var, justify="center")
        self.hidden_layer_en.pack(side="left", padx=2)

        learning_rate_lb = tk.Label(self.top_bar, text="Learning rate: ", bg="light gray")
        learning_rate_lb.pack(side="left")

        self.learning_rate_var = tk.DoubleVar()
        self.learning_rate_var.set(0.3)
        self.learning_rate_en = tk.Entry(self.top_bar, width="3", text=self.learning_rate_var, justify="center")
        self.learning_rate_en.pack(side="left", padx=2)

        max_epochs_lb = tk.Label(self.top_bar, text="Max epochs: ", bg="light gray")
        max_epochs_lb.pack(side="left")

        self.max_epochs_var = tk.IntVar()
        self.max_epochs_var.set(5000)
        self.max_epochs_en = tk.Entry(self.top_bar, width="6", text=self.max_epochs_var, justify="center")
        self.max_epochs_en.pack(side="left", padx=2)

        tolerance_error_lb = tk.Label(self.top_bar, text="Tolerance error: ", bg="light gray")
        tolerance_error_lb.pack(side="left")

        self.error_tolerance_var = tk.DoubleVar()
        self.error_tolerance_var.set(0.001)
        self.error_tolerance_en = tk.Entry(self.top_bar, width="8", text=self.error_tolerance_var, justify="center")
        self.error_tolerance_en.pack(side="left")

        self.train_b = tk.Button(self.top_bar, text="Train", command=self.train_network, state=tk.DISABLED)
        self.train_b.pack(side="right")

        self.top_bar.pack(expand="yes", fill="both")

    def create_status_bar(self):
        self.status_bar = tk.Frame(self.parent, height=30, relief='raised', padx=2, pady=2, bg="light gray")
        samples_lb = tk.Label(self.status_bar, text="Training samples loaded: ", bg="light gray")
        samples_lb.pack(side="left")
        self.samples_number_var = tk.IntVar()
        self.samples_number_var.set(0)
        samples_number_lb = tk.Label(self.status_bar, textvariable=self.samples_number_var, bg="gray")
        samples_number_lb.pack(side="left")

        output_lb = tk.Label(self.status_bar, text="Unique symbols / output layer nodes: ", bg="light gray")
        output_lb.pack(side="left")
        self.output_number_var = tk.IntVar()
        self.output_number_var.set(0)
        output_number_lb = tk.Label(self.status_bar, textvariable=self.output_number_var, bg="gray")
        output_number_lb.pack(side="left")

        clear_b = tk.Button(self.status_bar, text="Erase", command=self.reset_canvas)
        clear_b.pack(side="right")

        self.status_bar.pack(expand="yes", fill="both")

    def reset_canvas(self):
        self.canvas.delete("all")

    def train_network(self):
        # create instance of neural network
        self.n = NeuralNetwork(self.vec_size_var.get() * 2, self.hidden_layer_var.get(), self.output_number_var.get(),
                               self.learning_rate_var.get())
        # train the network
        self.n.online_train(self.training_set, self.max_epochs_var.get(), self.error_tolerance_var.get())
        self.mode_menu.entryconfig("Recognition", state=tk.NORMAL)

    def create_file_menu(self):
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="New training set", command=self.on_new_clicked,
                                   compound='left', image=self.new_file_icon, underline=0)
        self.file_menu.add_command(label="Load training set", command=self.on_load_clicked,
                                   compound='left', image=self.open_file_icon)
        self.file_menu.add_command(label="Save training set", command=self.on_save_clicked,
                                   compound='left', image=self.save_file_icon)
        self.file_menu.add_command(label="Save training set as...", command=self.on_save_as_clicked,
                                   compound='left', image=self.save_file_as_icon)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.entryconfig(0, state=tk.DISABLED)
        self.file_menu.entryconfig(2, state=tk.DISABLED)
        self.file_menu.entryconfig(3, state=tk.DISABLED)
        self.parent.config(menu=self.menu_bar)

    def on_new_clicked(self):
        self.vec_size_en.config(state='normal')
        self.vec_size_var.set(20)
        self.samples_number_var.set(0)
        self.output_number_var.set(0)
        self.training_set = []
        self.translation.clear()
        self.file_name = None
        self.set_title()
        self.file_menu.entryconfig(0, state=tk.DISABLED)
        self.file_menu.entryconfig(2, state=tk.DISABLED)
        self.file_menu.entryconfig(3, state=tk.DISABLED)
        self.train_b.config(state=tk.DISABLED)
        self.mode_menu.entryconfig("Recognition", state=tk.DISABLED)

    def set_title(self):
        if self.file_name:
            self.parent.title(self.title + " [" + os.path.basename(self.file_name) + "]")
        else:
            self.parent.title(self.title)

    def on_load_clicked(self):
        self.file_name = filedialog.askopenfilename(defaultextension=".txt", filetypes=[("All Files", "*.*"),
                                                        ("Text documents", "*.txt")])
        error = False
        starting_set_size = len(self.training_set)
        if self.file_name:
            self.set_title()
            with open(self.file_name, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:
                    symbol = row[0]
                    norm_vector = row[1:]
                    if (self.vec_size_en.cget('state') == 'disabled'
                            and len(norm_vector) == self.vec_size_var.get() * 2)\
                            or self.vec_size_en.cget('state') == 'normal':

                        self.translation.insert_symbol(symbol)
                        self.training_set.append([self.translation.get_index_by_symbol(symbol)] + norm_vector)
                        self.samples_number_var.set(self.samples_number_var.get() + 1)
                    else:
                        error = True
                        messagebox.showerror("Error", message="Incompatible vector size")
                        break
            if not error:
                self.output_number_var.set(self.translation.get_dict_length())
                self.vec_size_var.set(int(len(norm_vector) / 2))
                self.vec_size_en.config(state='disabled')
                self.file_menu.entryconfig(0, state=tk.NORMAL)
                if starting_set_size > 0:
                    self.file_menu.entryconfig(2, state=tk.NORMAL)
                self.file_menu.entryconfig(3, state=tk.NORMAL)
                self.train_b.config(state=tk.NORMAL)

    def on_save_clicked(self):
        self.write_memory_to_file()
        self.file_menu.entryconfig(2, state=tk.DISABLED)

    def write_memory_to_file(self):
        with open(self.file_name, 'w+', newline='') as csv_file:
            wr = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
            for row in self.training_set:
                wr.writerow([self.translation.get_symbol_by_index(row[0])] + row[1:])

    def on_save_as_clicked(self):
        self.file_name = filedialog.asksaveasfilename(title="Select file", filetypes=(("csv files", "*.csv"),
                                                                                      ("all files", "*.*")),
                                                 defaultextension="*.csv")
        if self.file_name:
            self.write_memory_to_file()
            self.set_title()

    def create_mode_menu(self):
        self.mode_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.mode_choice = tk.StringVar()
        self.mode_choice.set("Labeling")
        self.mode_menu.add_radiobutton(label="Labeling", variable=self.mode_choice, command=self.change_mode)
        self.mode_menu.add_radiobutton(label="Recognition", variable=self.mode_choice, command=self.change_mode)
        self.mode_menu.entryconfig("Recognition", state=tk.DISABLED)
        self.menu_bar.add_cascade(label="Mode", menu=self.mode_menu)
        self.parent.config(menu=self.menu_bar)

    def change_mode(self):
        if self.mode_choice.get() == "Recognition":
            self.menu_bar.entryconfig("Algorithm", state=tk.NORMAL)
        elif self.mode_choice.get() == "Labeling":
            self.menu_bar.entryconfig("Algorithm", state=tk.DISABLED)

    def create_algorithm_menu(self):
        self.algorithm_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.algorithm_choice = tk.StringVar()
        self.algorithm_choice.set("Backpropagation")
        self.algorithm_menu.add_radiobutton(label="Backpropagation", variable=self.algorithm_choice)
        self.menu_bar.add_cascade(label="Algorithm", menu=self.algorithm_menu)
        self.menu_bar.entryconfig("Algorithm", state=tk.DISABLED)
        self.parent.config(menu=self.menu_bar)

    def create_drawing_canvas(self):
        self.canvas_frame = tk.Frame(self.parent)
        self.canvas_frame.pack(expand="yes", fill="both")
        self.canvas = tk.Canvas(self.canvas_frame, width=self.default_width, height=self.default_height)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

    def bind_mouse(self):
        self.canvas.bind("<Button-1>", self.on_mouse_button_pressed)
        self.canvas.bind("<Button1-Motion>", self.on_mouse_button_pressed_motion)
        self.canvas.bind("<Button1-ButtonRelease>", self.on_mouse_button_released)

    def on_mouse_button_pressed(self, event):
        self.points.append([event.x, event.y])

    def on_mouse_button_pressed_motion(self, event):
        self.points.append([event.x, event.y])

    def on_mouse_button_released(self, event):
        if len(self.points) > 0:  # prevent canvas frame from grabbing filedialog mouse button events
            self.draw_line_from_array(self.points)
            red_points = self.reduce_points(self.points, self.vec_size_var.get() + 1)
            self.draw_circle_from_array(red_points)
            norm_vector = self.gen_vector(red_points)

            if self.mode_choice.get() == "Labeling":
                self.labeling(norm_vector)
            elif self.mode_choice.get() == "Recognition":
                self.recognition(norm_vector)

    def translate_result(self, result):
        return [[self.translation.get_symbol_by_index(e[0]), float(e[1])] for e in enumerate(result)]

    def get_top_result(self, result):
        return max(result, key=lambda x: x[1])

    def get_sorted_result(self, result):
        return sorted(result, key=lambda x: x[1], reverse=True)

    def bounding_box(self, points):
        bot_left_x = min(point[0] for point in points)
        bot_left_y = min(point[1] for point in points)
        top_right_x = max(point[0] for point in points)
        top_right_y = max(point[1] for point in points)

        return [(bot_left_x, bot_left_y), (top_right_x, top_right_y)]

    def labeling(self, norm_vector):
        self.points.clear()

        answer = simpledialog.askstring("Input", "Which symbol is this?", parent=self.parent)
        if answer and len(norm_vector) == (self.vec_size_var.get() * 2):
            self.translation.insert_symbol(answer)
            self.training_set.append([self.translation.get_index_by_symbol(answer)] + norm_vector)

            self.vec_size_en.config(state='disabled')
            self.samples_number_var.set(self.samples_number_var.get() + 1)
            self.output_number_var.set(self.translation.get_dict_length())

            self.file_menu.entryconfig(0, state=tk.NORMAL)
            if self.file_name:
                self.file_menu.entryconfig(2, state=tk.NORMAL)
            self.file_menu.entryconfig(3, state=tk.NORMAL)
            self.train_b.config(state=tk.NORMAL)
        else:
            print("Drawing of symbol aborted")

    def recognition(self, norm_vector):
        result = self.n.query((numpy.asfarray(norm_vector)))
        tr_result = self.translate_result(result)
        print(self.get_sorted_result(tr_result))
        point1, point2 = self.bounding_box(self.points)
        offset = 8
        self.canvas.create_rectangle(point1[0] - offset, point1[1] - offset,
                                     point2[0] + offset, point2[1] + offset, outline="light green")
        result_symbol, result_confidence = self.get_top_result(tr_result)
        self.canvas.create_text(point2[0] - 15, point1[1] - 15,
                                text=result_symbol + ' [' + '{0:.3g}'.format(result_confidence * 100) + '%]')
        self.points = []

    def calc_vector(self, i, j):
        x = j[0] - i[0]
        y = j[1] - i[1]
        d = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
        if d == 0:
            d = 0.001
        # rescale to the interval [0.01 - 0.99]
        return [(x/d * 0.49) + 0.50, (y/d * 0.49) + 0.50]

    def gen_vector(self, a):
        v = [self.calc_vector(i, j) for i, j in zip(a[: -1], a[1:])]
        return [item for sublist in v for item in sublist]  # flatten array

    def draw_line_from_array(self, a):
        [self.draw_line(i, j) for i, j in zip(a[: -1], a[1:])]

    def draw_line(self, p1, p2):
        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1])

    def draw_circle_from_array(self, a):
        for p in a:
            self.draw_circle(p)

    def draw_circle(self, p):
        self.canvas.create_oval(p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4, fill="red")

    def distances(self, l):
        return [math.dist(i, j) for i, j in zip(l[: -1], l[1:])]

    def midpoint(self, a, b):
        return [int((a[0] + b[0]) / 2.0), int((a[1] + b[1]) / 2.0)]

    def reduce_points(self, li, goal):
        if goal == 1:
            return li[0]
        elif len(li) == goal:
            return li
        else:
            core_li = li[1:-1]

            while len(core_li) > goal - 2:
                d = self.distances(core_li)
                i = d.index(min(d))
                p1 = core_li.pop(i)
                p2 = core_li.pop(i)
                core_li.insert(i, self.midpoint(p1, p2))

            core_li = [li[0]] + core_li
            core_li.append(li[-1])
            return core_li

if __name__ == '__main__':
    root = tk.Tk()
    title = "Single stroke symbol recognition"
    root.title(title)
    app = NeuralDraw(root, title)
    root.mainloop()