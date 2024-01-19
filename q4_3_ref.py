import tkinter as tk

from PIL import  ImageTk, Image, ImageDraw
import matplotlib.pyplot as plt


def button_43(): # 丟入模型訓練
    out = image1.resize((32, 32)) # out是圖片
    # out.save('55.png')

def button_44(): # 移除
    c4.delete("line") #清除 tags = "line"的图像
    revoke[:] = []
    draw.rectangle([(0, 0), (250, 250)], fill='#000') # 覆蓋掉畫出來的結果 

def _canvas_draw(event):
    if not event: # 清空紀錄點
        draw_point[:] = ['',''] 
        return
    point = [event.x, event.y]   #首次座標
    if draw_point==['','']:      #滑鼠按下去
        draw_point[:] = point    #首次座標
        revoke.append([])        #紀錄
    else:
        revoke[-1].append(c4.create_line(draw_point[0], draw_point[1], event.x, event.y, fill="#FFF", width=5, tags = "line")) #畫線
        draw.line([draw_point[0], draw_point[1], point[0], point[1]], '#FFF', width=9)
        draw_point[:] = point    #二次座標
   
if __name__ == '__main__':
    
    # GUI setting
    window = tk.Tk()
    window.title("Homework2 v1") #title名稱
    window.minsize(width=1200, height=900) #最小視窗大小
    window.resizable(width=False, height=False) #能否更改視窗大小
    
    # ----- 按鈕GUI -----
    btn43 = tk.Button(window, text="3. Predict", command=button_43, height=1, width=20)
    btn43.grid(row=2, column=0, padx=10, pady=10)
    btn44 = tk.Button(window, text="4. Reset", command=button_44, height=1, width=20)
    btn44.grid(row=3, column=0, padx=10, pady=10)
    
    # 畫布GUI
    c4 = tk.Canvas(window, width=250, height=250, bg='#000')
    c4.grid(row=0, column=1, rowspan=4, padx=15, pady=20)
    
    image1 = Image.new("RGB", (250, 250), '#000') #PIL畫圖
    draw = ImageDraw.Draw(image1)
    
    draw_point = ['', '']  #紀錄滑鼠點
    revoke = [] #滑鼠ID
    c4.bind("<B1-Motion>", _canvas_draw) 
    c4.bind("<ButtonRelease-1>", lambda event:_canvas_draw(0)) 

    window.mainloop() #等待處理視窗事件，讓程式可以繼續執行