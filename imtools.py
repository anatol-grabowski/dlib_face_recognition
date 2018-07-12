import re
import os
import types
import cv2
import numpy as np

def image_from_url(url, bgr=False):
    import urllib
    buff = urllib.request.urlopen(url).read()
    arr = np.frombuffer(buff, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if not bgr: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    # usage: cv2.ellipse(frame, *imtools.rect2ellipse(rect), (255,0,0), 2, 1)
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

def open_sequence(source):
    if os.path.isdir(source):
        return (os.path.join(source, filename) for filename in os.listdir(source))
    re_youtube = '^https?://'
    if type(source) is str and re.match(re_youtube, source) != None:
        source = video_url_from_youtube(source)
    video = cv2.VideoCapture(source)
    if not video.isOpened(): raise Error('Unable to open video from source {}'.format(source))
    return video

def read_frame(source, bgr=False, vflip=False):
    if type(source) == types.GeneratorType:
        path = next(source, None)
        if path == None: return
        frame = open_image(path, bgr)
    else:
        ok, frame = source.read()
        if not ok: return
        if not bgr: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if vflip: frame = cv2.flip(frame, 1)
    return frame

def open_image(source, bgr=False):
    re_http_url = '^https?://'
    if re.match(re_http_url, source) != None:
        return image_from_url(source, bgr)
    img = cv2.imread(source)
    if not bgr: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def dlib_detect(frame, detector):
    import dlib
    dets, scores, idx = detector.run(frame, 0, 0.5)
    dets = sorted(dets, key=lambda det: -det.area())
    dets = dlib.rectangles(dets)
    return dets

def cv_detect(frame, detector):
    gray = frame
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray)
    rects = sorted(rects, key=lambda rect: -rect[2] * rect[3])
    return rects

def detect(frame, detector, do_return_dlib_rects):
    if type(detector) == cv2.CascadeClassifier:
        rects = cv_detect(frame, detector)
        if do_return_dlib_rects:
            rects = dlib.rectangles(list(map(cv_rect2dlib_rect, rects)))
    else:
        rects = dlib_detect(frame, detector)
        if not do_return_dlib_rects:
            rects = list(map(dlib_rect2cv_rect, rects))
    return rects
