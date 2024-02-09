from tkinter import *
import PIL
from PIL import Image, ImageDraw
import numpy as np


class Drawer:

    def __init__(self, model, model_type='cnn'):
        self.model = model
        self.model_type = model_type

        self.root = Tk()
        self.lastx, self.lasty = None, None

        self.cv = Canvas(self.root, width=300, height=200, bg='black')
        # --- PIL
        self.image1 = PIL.Image.new('RGB', (100, 100), 'black')
        self.draw = ImageDraw.Draw(self.image1)
        # ---
        self.image_number = 0
        self.cv = Canvas(self.root, width=100, height=100, bg='black')
        self.image1 = PIL.Image.new('RGB', (100, 100), 'black')
        self.draw = ImageDraw.Draw(self.image1)
        self.cv.bind('<1>', self.activate_paint)
        self.cv.pack(expand=YES, fill=BOTH)
        self.btn_save = Button(text="predict", command=self.predict)
        self.btn_save.pack()
        self.root.mainloop()

    def clear(self):
        self.cv.delete("all")
        self.image1 = PIL.Image.new('RGB', (100, 100), 'black')
        self.draw = ImageDraw.Draw(self.image1)

    def predict(self):
        gray_image = self.image1.convert('L')
        small_image = gray_image.resize((28, 28))

        # Convert the image to a numpy array
        image_array = np.array(small_image)
        normalized_image_array = image_array / 255.0

        if self.model_type == 'cnn':
            reshaped_image_array = np.expand_dims(normalized_image_array, axis=0)
            reshaped_image_array = np.expand_dims(reshaped_image_array, axis=-1)
            prediction = self.model.predict(reshaped_image_array)

        else:
            prediction = self.model.predict_from_normalized_image_array(normalized_image_array)

        print(f'Prediction: {prediction}')
        self.clear()

    def activate_paint(self, e):
        self.cv.bind('<B1-Motion>', self.paint)
        self.lastx, self.lasty = e.x, e.y

    def paint(self, e):
        x, y = e.x, e.y
        self.cv.create_line((self.lastx, self.lasty, x, y), width=7, fill='white')
        #  --- PIL
        self.draw.line((self.lastx, self.lasty, x, y), fill='white', width=7)
        self.lastx, self.lasty = x, y
