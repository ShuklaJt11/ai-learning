# Importing general libraries
import matplotlib.pyplot as plt
import numpy as np
import time
from random import random, randint

# Import Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing Dqn from ai file
from ai import Dqn

# Adding click configuration
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing variables to keep track of sand
last_x, last_y, n_points, length = [0] * 4

# Setting up our AI and other variables
brain = Dqn(5, 3, 0.9, 100_000, 0)
action_to_rotation = [0, 20, -20]
last_reward = 0
scores = []
first_update = True

# Initializing the map
def init():
    """Initialize the map for Agent
    """
    global sand, goal_x, goal_y, first_update
    sand = np.zeros((map_width, map_height))
    goal_x = 20
    goal_y = map_height - 20
    first_update = False

# Initializing the last distance
last_distance = 0

# Creating Car class
class Car (Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        """Function for movement of the car

        Args:
            rotation (Float): Rotation of the movement in degrees
        """
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x) - 10 : int(self.sensor1_x) + 10, int(self.sensor1_y) - 10 : int(self.sensor1_y) + 10])) / (20 * 20.)
        self.signal2 = int(np.sum(sand[int(self.sensor2_x) - 10 : int(self.sensor2_x) + 10, int(self.sensor2_y) - 10 : int(self.sensor2_y) + 10])) / (20 * 20.)
        self.signal3 = int(np.sum(sand[int(self.sensor3_x) - 10 : int(self.sensor3_x) + 10, int(self.sensor3_y) - 10 : int(self.sensor3_y) + 10])) / (20 * 20.)
        if self.sensor1_x > (map_width - 10) or self.sensor1_x < 10 or self.sensor1_y > (map_height - 10) or self.sensor1_y < 10:
            self.signal1 = 1
        if self.sensor2_x > (map_width - 10) or self.sensor2_x < 10 or self.sensor2_y > (map_height - 10) or self.sensor2_y < 10:
            self.signal2 = 1
        if self.sensor3_x > (map_width - 10) or self.sensor3_x < 10 or self.sensor3_y > (map_height - 10) or self.sensor3_y < 10:
            self.signal3 = 1

class Ball1 (Widget):
    pass
class Ball2 (Widget):
    pass
class Ball3 (Widget):
    pass

# Creating Game class
class Game (Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        """Funtion to initialize game car properties
        """
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
    
    def update (self, dt):
        global brain, last_reward, scores, last_distance, goal_x, goal_y, map_width, map_height

        map_width = self.width
        map_height = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal)
        scores.append(brain.score())
        rotation = action_to_rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x) ** 2 + (self.car.y - goal_y) ** 2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > (self.width - 10):
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > (self.height - 10):
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            goal_x = self.width - goal_x
            goal_y = self.width - goal_y
        last_distance = distance

# Creating class to add painting tools
class MapPaintWidget (Widget):
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1
    
    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = touch.x
            y = touch.y
            length += np.sqrt(max((x - last_x) ** 2 + (y - last_y) ** 2, 2))
            n_points += 1.
            density = n_points / (length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Creating class to add buttons and paint widget to the game
class CarApp (App):
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MapPaintWidget()
        clearbtn = Button(text='Clear')
        savebtn = Button(text='Save', pos=(parent.width, 0))
        loadbtn = Button(text='Load', pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent
    
    def clear_canvas(self):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((map_width, map_height))

    def save(self):
        print('Saving Brain...')
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self):
        print('Loading last saved Brain...')
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()       