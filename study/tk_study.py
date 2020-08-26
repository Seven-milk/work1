# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import tkinter as tk

# window = tk.Tk()
# window.title('study_tk')
# window.geometry('200x200')
# var = tk.StringVar()
# l = tk.Label(window, textvariable=var, bg='green', font=('Arial', 12), width=15, height=2)
# l.pack()
# on_hit = False
#
#
# def hit_function():
#     global on_hit
#     if on_hit == False:
#         on_hit = True
#         var.set('you hit')
#     else:
#         on_hit = False
#         var.set('')

# window = tk.Tk()
# window.title('study_tk')
# window.geometry('200x200')
# e = tk.Entry(window, show=None)
# e.pack()
#
#
# def insert_point():
#     var = e.get()
#     t.insert('insert', var)
#
#
# def insert_end():
#     var = e.get()
#     t.insert('end', var)
#
#
# b1 = tk.Button(window, text='insert_point', width=10, height=2, command=insert_point)
# b1.pack()
# b2 = tk.Button(window, text='insert_end', width=10, height=2, command=insert_end)
# b2.pack()
# t = tk.Text(window, height=2)
# t.pack()
# window.mainloop()

# window = tk.Tk()
# window.title('study_tk')
# window.geometry('200x200')
# var1 = tk.StringVar()
# l = tk.Label(window, textvariable=var1, bg='yellow', width=4)
# l.pack()
#
#
# def print_selection():
#     global var1
#     value = lb.get(lb.curselection())
#     var1 = var1.set(value)
#
#
# b1 = tk.Button(window, text='print selection', width=10, height=2, command=print_selection)
# b1.pack()
# var2 = tk.StringVar()
# var2.set((11, 22, 33, 44))
# lb = tk.Listbox(window, listvariable=var2)
# lb.insert(1, 'first')
# lb.pack()
# window.mainloop()

# window = tk.Tk()
# window.title('study_tk')
# window.geometry('200x200')
# var = tk.StringVar()
# l = tk.Label(window, text='empty', bg='yellow', width=15)
# l.pack()
#
#
# def print_selection():
#     l.config(text='you have selected'+var.get())
#
#
# r1 = tk.Radiobutton(window, text='Option A', variable=var, value='A', command=print_selection)
# r1.pack()
# r2 = tk.Radiobutton(window, text='Option B', variable=var, value='B', command=print_selection)
# r2.pack()
# r3 = tk.Radiobutton(window, text='Option C', variable=var, value='C', command=print_selection)
# r3.pack()
# window.mainloop()


# window = tk.Tk()
# window.title('study_tk')
# window.geometry('200x200')
# var = tk.StringVar()
# l = tk.Label(window, text='empty', bg='yellow', width=15)
# l.pack()
#
#
# def print_selection(v):
#     l.config(text='you have selected'+v)
#
#
# s = tk.Scale(window, label='try me', from_=5, to=11, orient=tk.HORIZONTAL, length=200, showvalue=0,
#              tickinterval=3, resolution=0.01, command=print_selection)
# s.pack()
# window.mainloop()

window = tk.Tk()
window.title('study_tk')
window.geometry('200x200')
var = tk.StringVar()
l = tk.Label(window, text='empty', bg='yellow', width=15)
l.pack()
var1 = tk.IntVar()
var2 = tk.IntVar()


def print_selection():
    if (var1.get() == 1) & (var2.get() == 0):
        l.config(text='I love only Python')
    elif (var1.get() == 0) & (var2.get() == 1):
        l.config(text='I love only C++')
    elif (var1.get() == 0) & (var2.get() == 0):
        l.config(text='I do not love either')
    else:
        l.config(text='I love both')


c1 = tk.Checkbutton(window, text='Python', variable=var1, onvalue=1, offvalue=0, command=print_selection)
c1.pack()
c2 = tk.Checkbutton(window, text='C++', variable=var2, onvalue=1, offvalue=0, command=print_selection)
c2.pack()
window.mainloop()
