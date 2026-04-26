import tkinter as tk
from PIL import Image, ImageTk
from mpu6050 import mpu6050
import time


class SensorController:
    UPDATE_INTERVAL = 1000  # in milliseconds
    THRESHOLD = 2           # movement threshold

    def __init__(self, master):
        self.master = master
        self.master.title("Control with MPU6050")

        self.active_keys = set()

        # Load base image
        self.base_img = Image.open("controle_base.png")
        self.base_img_tk = ImageTk.PhotoImage(self.base_img)

        # Setup canvas
        self.canvas = tk.Canvas(master, width=self.base_img.width, height=self.base_img.height)
        self.canvas.pack()
        self.bg = self.canvas.create_image(0, 0, anchor="nw", image=self.base_img_tk)

        # Load pressed button images
        self.buttons = {
            'right': ImageTk.PhotoImage(Image.open("direita_pressionado.png")),
            'left': ImageTk.PhotoImage(Image.open("esquerda_pressionado.png")),
            'up': ImageTk.PhotoImage(Image.open("cima_pressionado.png")),
            'down': ImageTk.PhotoImage(Image.open("baixo_pressionado.png"))
        }

        # Initialize sensors
        self.head_sensor = mpu6050(0x68)
        self.chair_sensor = mpu6050(0x69)

        time.sleep(2)  # allow sensor stabilization

        # Start periodic check
        self.check_sensors()

    def check_sensors(self):
        try:
            head = self.head_sensor.get_accel_data()
            chair = self.chair_sensor.get_accel_data()
        except Exception as e:
            print(f"Sensor read error: {e}")
            head, chair = {'x': 0.0, 'y': 0.0}, {'x': 0.0, 'y': 0.0}

        delta_x = head['x'] - chair['x']
        delta_y = head['y'] - chair['y']

        print(f"Head sensor - x: {head['x']}, y: {head['y']}")
        print(f"Chair sensor - x: {chair['x']}, y: {chair['y']}")

        # Movement logic
        if delta_y < -self.THRESHOLD:
            self.press('right')
        elif delta_y > self.THRESHOLD:
            self.press('left')
        elif delta_x > self.THRESHOLD:
            self.press('up')
        elif delta_x < -self.THRESHOLD:
            self.press('down')
        else:
            self.release()

        self.master.after(self.UPDATE_INTERVAL, self.check_sensors)

    def press(self, direction):
        if direction not in self.active_keys:
            self.active_keys = {direction}
            self.canvas.create_image(0, 0, anchor="nw", image=self.buttons[direction])

    def release(self):
        if self.active_keys:
            self.active_keys.clear()
            self.canvas.create_image(0, 0, anchor="nw", image=self.base_img_tk)


if __name__ == "__main__":
    root = tk.Tk()
    app = SensorController(root)
    root.mainloop()
