import cv2
import dlib
import numpy as np
import sys
import imtools
import recog

video_source = sys.argv[1] if len(sys.argv) >= 2 else 0
video_vflip = type(video_source) is int
window_name = 'recognition'

dlib_models_path = './'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dlib_models_path + 'shape_predictor_5_face_landmarks.dat')
describer = dlib.face_recognition_model_v1(dlib_models_path + 'dlib_face_recognition_resnet_model_v1.dat')

recognizer_file = './recognizer.pickle'
recognizer = recog.Recognizer(recognizer_file)
images_path = './images/'
dist_treshold = 0.5

print('Known faces')
for face in recognizer.faces:
    print('{} {}'.format(chr(face.id), face.name))

def detect_faces(frame):
    dets = detector(frame, 0)
    dets = sorted(dets, key=lambda det: det.area())
    dets = dlib.rectangles(dets)
    return dets

def get_descriptor(frame, dlib_rect):
    shape = predictor(frame, dlib_rect)
    descr = describer.compute_face_descriptor(frame, shape)
    descr = np.array(descr)
    return descr

def get_descriptors(frame, dets):
    descrs = []
    for det in dets:
        descr = get_descriptor(frame, det)
        descrs.append(descr)
    return descrs

def train_once(frame, det, descr, id):
    face = recognizer.update(id, descr)
    pts = imtools.dlib_rect2pts(det)
    x, y = pts[0]
    x2, y2 = pts[1]
    face_img = frame[y:y2,x:x2]
    filename = images_path + '{}-{}.jpeg'.format(face.name, str(np.random.randint(2**32)))
    cv2.imwrite(filename, face_img)

    cv2.putText(frame, "{} / {}".format(face.name, 'training'), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.rectangle(frame, *pts, (255, 0, 0), 2)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    if face.name == None:
        face.name = input('Enter name: ')
    recognizer.save(recognizer_file)
    print('updated', face.name)

def recognize_faces(frame, dets, descrs):
    for det, descr in zip(dets, descrs):
        face, dist = recognizer.match(descr)
        pts = imtools.dlib_rect2pts(det)
        x, y = pts[0]
        if face == None or dist > dist_treshold:
            cv2.putText(frame, "{} / {:.2f}".format('???', dist), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "{} ({}) / {:.2f}".format(face.name, chr(face.id), dist), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def main():
    video = imtools.open_video(video_source)
    while True:
        frame = imtools.read_frame(video, bgr=True, vflip=video_vflip)
        if frame is None: return

        dets = detect_faces(frame)
        descrs = get_descriptors(frame, dets)

        key = cv2.waitKey(1) & 0xff
        if key == 27: break
        if ord('0') <= key <= ord('9'):
            if len(descrs) > 0:
                det, descr, id = dets[0], descrs[0], key
                train_once(frame, det, descr, id)
                continue

        for det in dets:
            pts = imtools.dlib_rect2pts(det)
            cv2.rectangle(frame, *pts, (255, 0, 0), 1)
        recognize_faces(frame, dets, descrs)
        cv2.imshow(window_name, frame)
    print('Done reading frames')

if __name__ == '__main__':
    main()
