import cv2
import dlib
import numpy as np
import sys
import os
import imtools
import recog

video_source = sys.argv[1] if len(sys.argv) >= 2 else 0
video_vflip = type(video_source) is int
window_name = 'recognition'

dlib_models_path = './'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(dlib_models_path, 'shape_predictor_5_face_landmarks.dat'))
describer = dlib.face_recognition_model_v1(os.path.join(dlib_models_path, 'dlib_face_recognition_resnet_model_v1.dat'))

recognizer_file = 'recognizer.pickle'
recognizer = recog.Recognizer(recognizer_file)
images_path = './images/'
dist_treshold = 0.5

cv_cascades_path = '/_tot/projects/opencv/opencv-3.4.1/data/haarcascades/'
face_cascade = cv2.CascadeClassifier(os.path.join(cv_cascades_path, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cv_cascades_path, 'haarcascade_eye.xml'))

# from timefn import timefn



# @timefn
def check_if_face_straight(frame, det, shape):
    points = np.array(list(map(lambda p: (p.x, p.y), shape.parts())))
    p1, p2 = points[2], points[0]
    (x1, y1), (x2, y2) = p1, p2
    theta = np.arctan((x2-x1)/(y2-y1))
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s], [s,  c]])
    p1 = np.dot(rot_mat, p1)
    p2 = np.dot(rot_mat, p2)
    pn = np.dot(rot_mat, points[4])
    mid = p1[1] + (p2[1] - p1[1]) / 2
    tolerance = np.abs(p2[1] - p1[1]) * 0.15
    for p in shape.parts():
        cv2.circle(frame, (p.x, p.y), 10, (0, 255, 0), 1)
    if not (mid - tolerance <= pn[1] <= mid + tolerance):
        return False

    x, y, w, h = imtools.dlib_rect2cv_rect(det)
    face_img = frame[y:y+h, x:x+w]
    eyes = imtools.detect(frame_img, eye_cascade, False)

    # print(eyes)
    return True

def get_descriptor(frame, dlib_rect):
    shape = predictor(frame, dlib_rect)
    # print(check_if_face_straight(frame, dlib_rect, shape))
    descr = describer.compute_face_descriptor(frame, shape)
    descr = np.array(descr)
    return descr

def get_descriptors(frame, dets):
    descrs = []
    for det in dets:
        descr = get_descriptor(frame, det)
        descrs.append(descr)
    return descrs

def format_face_text(face):
    return '[{}] {}'.format(chr(face.id), face.name if face.name != None else '<Enter name in cmd>')

def train_once(frame, det, descr, id):
    face = recognizer.update(id, descr)
    pts = imtools.dlib_rect2pts(det)
    (x1, y1), (x2, y2) = pts
    x, y = x1, y1
    offset = x2 - x1
    x1, x2 = np.clip([x1-offset, x2+offset], 0, frame.shape[1]-1)
    y1, y2 = np.clip([y1-offset, y2+offset], 0, frame.shape[0]-1)
    face_img = frame[y1:y2,x1:x2].copy()
    cv2.putText(frame, "{} / {}".format(format_face_text(face), 'training'), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.rectangle(frame, *pts, (255, 0, 0), 3)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    if face.name == None:
        face.name = input('Enter name: ')
    recognizer.save(recognizer_file)
    filename = os.path.join(images_path, '{}-{}.jpeg'.format(face.name, str(np.random.randint(2**32))))
    cv2.imwrite(filename, face_img)
    print('updated ' + format_face_text(face))

def recognize_faces(frame, dets, descrs):
    for det, descr in zip(dets, descrs):
        face, dist = recognizer.match(descr)
        pts = imtools.dlib_rect2pts(det)
        x, y = pts[0]
        if face == None or dist > dist_treshold:
            cv2.putText(frame, "{} / {:.2f}".format('???', dist), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, format_face_text(face) + ' / {:.2f}'.format(dist), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def main():
    num_descrs = 0
    print('Known faces:')
    for face in recognizer.faces:
        print(format_face_text(face))
        num_descrs += len(face.descriptors)
    print('Total descriptors #:', num_descrs)

    if not os.path.isdir(images_path):
        os.makedirs(images_path)
        print('Created directory for images at', images_path)

    video = imtools.open_sequence(video_source)
    while True:
        frame = imtools.read_frame(video, bgr=True, vflip=video_vflip)
        if frame is None: return

        dets = imtools.detect(frame, detector, True)
        descrs = get_descriptors(frame, dets)

        key = cv2.waitKey(1) & 0xff
        if key == 27: break
        if ord('0') <= key <= ord('9') or ord('a') <= key <= ord('z'):
            if len(descrs) > 0:
                det, descr, id = dets[0], descrs[0], key
                train_once(frame, det, descr, id)
                continue

        for det in dets:
            pts = imtools.dlib_rect2pts(det)
            cv2.rectangle(frame, *pts, (255, 0, 0), 1)
            if det is dets[0]:
                cv2.rectangle(frame, *pts, (255, 0, 0), 2)
        recognize_faces(frame, dets, descrs)
        cv2.imshow(window_name, frame)
    print('Done reading frames')

if __name__ == '__main__':
    main()
