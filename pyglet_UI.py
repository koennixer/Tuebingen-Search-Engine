#search engine UI created with pyglet
import pyglet
import csv
from pyglet import window

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

win = window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="Cool Search Engine")

with open('test_result.txt', 'r', encoding='utf8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    rows = list(reader)

labels = []

y = win.height - 30


#display column headers
headers = rows[0].keys()
header_text = " | ".join(headers)
labels.append(pyglet.text.Label(
                header_text,
                x=20,
                y=y,
                anchor_x='left',
                anchor_y='top'))

y -= 30

#display each row
for row in rows:
    text = " | ".join(str(row[h]) for h in headers)
    labels.append(pyglet.text.Label(
                    text,
                    x=20,
                    y=y,
                    anchor_x='left',
                    anchor_y='top'))
    y -= 25


@win.event
def on_draw():
    win.clear()
    for label in labels:
        label.draw()

pyglet.app.run()