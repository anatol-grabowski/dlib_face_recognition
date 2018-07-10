import re
import cv2
import numpy as np

def image_from_url(url, convert_to_bgr=False):
    import urllib
    buff = urllib.request.urlopen(url).read()
    arr = np.frombuffer(buff, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if not convert_to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    import PIL
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return PIL.Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def rect2pts(rect):
    x, y, w, h = rect
    return (x, y), (x+w, y+h)

def dlib_rect2cv_rect(rect):
    cv_rect = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
    return cv_rect

def cv_rect2dlib_rect(rect):
    import dlib
    x, y, w, h = rect
    dlib_rect = dlib.rectangle(x, y, x+w, y+h)
    return dlib_rect

def dlib_rect2pts(rect):
    cv_rect = dlib_rect2cv_rect(rect)
    pts = rect2pts(cv_rect)
    return pts

def rect2ellipse(rect):
    # call: cv2.ellipse(frame, *imtools.rect2ellipse(rect), (255,0,0), 2, 1)
    p1 = (int(rect[0]), int(rect[1]))
    p2 = (int(rect[0] + rect[2]), int(rect[1] + rect[3]))
    c = (int(rect[0] + rect[2]/2), int(rect[1] + rect[3]/2))
    axes = (int(rect[2]/2), int(rect[3]/2))
    angle, start_deg, end_deg = 0, 0, 360
    return c, axes, angle, start_deg, end_deg

def video_url_from_youtube(url):
    import pafy
    vPafy = pafy.new(url)
    play = vPafy.getbest(preftype="webm")
    video_url = play.url
    return video_url

def open_video(source):
    re_youtube = '^https?://'
    if type(source) is str and re.match(re_youtube, source) != None:
        source = video_url_from_youtube(source)
    video = cv2.VideoCapture(source)
    if not video.isOpened(): raise Error('Unable to open video from source {}'.format(source))
    return video

def read_frame(video, bgr=False, vflip=False):
    ok, frame = video.read()
    if not ok: return
    if vflip: frame = cv2.flip(frame, 1)
    if not bgr: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame
