from objdet import vid_runner
from tkinter import *
from tkinter.filedialog import askopenfilename

class App(Frame):

    def pass_config(self):
        self.yo_config.delete(0,END)
        self.yo_config.insert(INSERT,askopenfilename())

    def pass_weights(self):
        self.yo_weights.delete(0,END)
        self.yo_weights.insert(INSERT,askopenfilename())

    def pass_classes(self):
        self.yo_classes.delete(0,END)
        self.yo_classes.insert(INSERT,askopenfilename())

    def pass_vid(self):
        self.yo_vid.delete(0,END)
        self.yo_vid.insert(INSERT,askopenfilename())

    def make_ui(self):
        self.grid()
        self.yo_config_lbl = Label(self)
        self.yo_config_lbl["text"] = "Path to the yolo config file:"
        self.yo_config_lbl.grid(column=0, row=0, sticky='nesw', padx=3, pady=3)
        self.yo_config = Entry(self,width=50)
        self.yo_config.insert(INSERT,'YOLO_Tiny/yolov3-tiny-F1-10C.cfg')
        self.yo_config.grid(column=1, row=0, sticky='nesw', padx=3, pady=3)
        self.yo_config_btn = Button(self,text="...",command=self.pass_config)
        self.yo_config_btn.grid(column=2,row=0,sticky='nesw', padx=3, pady=3)
        self.yo_weights_lbl = Label(self)
        self.yo_weights_lbl["text"] = "Path to the yolo weight file:"
        self.yo_weights_lbl.grid(column=0, row=1, sticky='nesw', padx=3, pady=3)
        self.yo_weights = Entry(self,width=50)
        self.yo_weights.insert(INSERT,'YOLO_Tiny/yolov3-tiny-F1-10C_20000.weights')
        self.yo_weights.grid(column=1, row=1, sticky='nesw', padx=3, pady=3)
        self.yo_weights_btn = Button(self,text="...",command=self.pass_weights)
        self.yo_weights_btn.grid(column=2,row=1,sticky='nesw', padx=3, pady=3)
        self.run_vid = Button(self)
        self.yo_classes_lbl = Label(self)
        self.yo_classes_lbl["text"] = "Path to the yolo class file:"
        self.yo_classes_lbl.grid(column=0, row=2, sticky='nesw', padx=3, pady=3)
        self.yo_classes = Entry(self,width=50)
        self.yo_classes.insert(INSERT,'YOLO_Tiny/F1-obj-10C.names')
        self.yo_classes.grid(column=1, row=2, sticky='nesw', padx=3, pady=3)
        self.yo_classes_btn = Button(self,text="...",command=self.pass_classes)
        self.yo_classes_btn.grid(column=2,row=2,sticky='nesw', padx=3, pady=3)
        self.yo_vid_lbl = Label(self)
        self.yo_vid_lbl["text"] = "Path to the video to process:"
        self.yo_vid_lbl.grid(column=0, row=3, sticky='nesw', padx=3, pady=3)
        self.yo_vid = Entry(self,width=50)
        self.yo_vid.insert(INSERT,'F1_Austria_Race_Trimmed30.mkv')
        self.yo_vid.grid(column=1, row=3, sticky='nesw', padx=3, pady=3)
        self.yo_vid_btn = Button(self,text="...",command=self.pass_vid)
        self.yo_vid_btn.grid(column=2,row=3,sticky='nesw', padx=3, pady=3)
        self.run_vid["text"] = "Process the video"
        self.run_vid["command"] = lambda: vid_runner(self.yo_config.get(),self.yo_weights.get(),self.yo_classes.get(),self.yo_vid.get())
        self.run_vid.grid(column=0, row=4, sticky='nesw', padx=3, pady=3, columnspan=2)

    def __init__(self, master=None):
        Frame.__init__(self,master)
        self.pack()
        self.make_ui()

def window_worker():
    root = Tk()
    root.title("Video Processor")
    app = App(master=root)
    #root.geometry("420x210")
    app.mainloop()

if __name__ == "__main__":
    window_worker()