import tkinter as tk

window = tk.Tk()

window.geometry("640x400+100+100")
window.overrideredirect(1)
#window.title('EyeMouse')

def prt():
    print("left")

tk.Button(window, text = 'Left Click',command=prt).pack(side="left")
tk.Button(window, text = 'Right Click',command=prt).pack(side="center")
tk.Button(window, text = 'Cancel',command=prt).pack(side="right")


window.mainloop()