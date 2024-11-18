class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Rectangle:
    def __init__(self, x=0, y=0, w=0, h=0, x1=0, y1=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x1 = x1
        self.y1 = y1

    @staticmethod
    def from_xywh(x, y, w, h):
        return Rectangle(x, y, w, h, x + w, y + h)

    @staticmethod
    def from_xyxy(x, y, x1, y1):
        return Rectangle(x, y, abs(x1 - x), abs(y1 - y), x1, y1)

def is_point_inside_rectangle(p, r):
    return r.x <= p.x <= r.x1 and r.y <= p.y <= r.y1

def calculate_iou(rect1, rect2):
    xA = max(rect1.x, rect2.x)
    yA = max(rect1.y, rect2.y)
    xB = min(rect1.x1, rect2.x1)
    yB = min(rect1.y1, rect2.y1)

    intersection_area = max(0, xB - xA) * max(0, yB - yA)
    rect1_area = rect1.w * rect1.h
    rect2_area = rect2.w * rect2.h
    union_area = rect1_area + rect2_area - intersection_area

    return intersection_area / union_area
