import win32api
import win32con
import pyautogui
class MouseController:
    def __init__(self):
        print('생성자')

    def clickMouseLeft(x,y):
        win32api.SetCursorPos((x,y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)


    def clickMouseRight(x,y):
        win32api.SetCursorPos((x,y))
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)


    def moveMouseXY(x,y):
        win32api.SetCursorPos((x,y))

    # import pyautogui
    # screenWidth, screenHeight = pyautogui.size()
    # currentMouseX, currentMouseY = pyautogui.position()
    # pyautogui.moveTo(100, 150)
    # pyautogui.click()
    # pyautogui.moveRel(None, 10)  # move mouse 10 pixels down
    # pyautogui.doubleClick()
    # pyautogui.moveTo(500, 500, duration=2, tween=pyautogui.tweens.easeInOutQuad)  # use tweening/easing function to move mouse over 2 seconds.
    # pyautogui.typewrite('Hello world!', interval=0.25)  # type with quarter-second pause in between each key
    # pyautogui.press('esc')
    # pyautogui.keyDown('shift')
    # pyautogui.typewrite(['left', 'left', 'left', 'left', 'left', 'left'])
    # pyautogui.keyUp('shift')
    # pyautogui.hotkey('ctrl', 'c')

    # m = MouseController
    # m.moveMouseXY(800,800)

    # pyautogui.moveTo(500, 500, duration=2, tween=pyautogui.easeInOutQuad)

    if pow(4-2,2)+pow(4-2,2)<16 :
        print('a')
    else :
        print('b')