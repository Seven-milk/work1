# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import kivy
from kivy.app import App
from kivy.uix.label import Label

class EpicApp(App):
    def build(self):
        return Label(text='hello world')

if __name__ == "__main__":
    EpicApp().run()