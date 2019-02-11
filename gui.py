from tkinter import Tk, Label, Button, Text, END
from PIL import Image, ImageTk
import pandas as pd


class MyFirstGUI:
    def __init__(self, master):
        self.catalog = pd.read_csv("catalog_new.csv", delimiter=',', encoding='utf8')
        self.master = master
        master.title("Michael rulz")

        self.label = Label(master, text="Get images from item id!")
        self.label.pack()

        self.txt = Text(height=5, width=10)
        self.txt.insert("1.0", '100224568')
        self.txt.pack()

        self.get_item = Button(master, text="get item", command=self.get_item)
        self.get_item.pack()

        self.del_txt = Button(master, text="clear txt", command=self.deltxt)
        self.del_txt.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def download_img(self, url):
        import urllib2
        digitalpdf = urllib2.urlopen(url)
        output = open('test.jpg', 'wb')
        output.write(digitalpdf.read())
        output.close()

    def deltxt(self):
        self.txt.delete('1.0', END)

    def get_item(self):
        try:
            try:
                self.panel.destroy()
            except:
                pass
            try:
                self.desc.destroy()
            except:
                pass

                item_id = int(self.txt.get("1.0", 'end-1c'))
            row = self.catalog[self.catalog[u'product_id'] == item_id]
            url = row.image_url.values[0]
            description = row.description.values[0]
            self.download_img(url)
            loc = 'test.jpg'
            self.img = ImageTk.PhotoImage(Image.open(loc).resize((250, 250), Image.ANTIALIAS))
            self.panel = Label(self.master, image=self.img)
            self.panel.pack(side="bottom", fill="both", expand="yes")

            self.desc = Text(self.master, height=10, width=20)
            self.desc.insert("1.0", description)
            self.desc.pack()

        except:
            print("Error!")
            return


root = Tk()

my_gui = MyFirstGUI(root)
root.mainloop()
