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
ask_for_names = True
auto_learn = False
min_score = 1.4
ok_subdetector_ids = [0, 3, 4]
min_dist = 0.57

def detect_faces(frame):
    dets, scores, idx = detector.run(frame, 1, 0.2)
    return dets, scores, idx

def recognize_faces(frame, dets, dist_treshold):
    faces = []
    descrs = []
    dists = []
    for det in dets:
        shape = predictor(frame, det)
        descr = describer.compute_face_descriptor(frame, shape)
        descr = np.array(descr)
        face, dist = recognizer.match(descr)
        if dist > dist_treshold: face = None
        faces.append(face)
        descrs.append(descr)
        dists.append(dist)
    return faces, descrs, dists

def learn_new_faces(frame, dets, scores, subdetector_ids, faces, descrs, dists):
    new_faces = []
    for det, score, subdetector_id, face, descr, dist in zip(dets, scores, subdetector_ids, faces, descrs, dists):
        if face != None: continue
        if score < min_score or subdetector_id not in ok_subdetector_ids or dist < min_dist: continue
        print('new face? score: {}, subdetector: {}, dist: {}'.format(score, subdetector_id, dist))
        id = np.random.randint(1e3, 1e4)
        face = train_once(frame, det, descr, id, ask_for_names)
        new_faces.append(face)
        return [face] # don't learn more than one new face from one frame for now, because train_once draws on frame
    return new_faces

def format_face_text(face):
    id = "'{}'".format(chr(face.id)) if face.id < 1000 else face.id
    return '[{}] {}'.format(id, face.name if face.name != None else '<Enter name in cmd>')

def train_once(frame, det, descr, id, ask_name=True):
    face = recognizer.update(id, descr)
    pts = imtools.dlib_rect2pts(det)
    (x1, y1), (x2, y2) = pts
    x, y = x1, y1
    offset = x2 - x1
    x1, x2 = np.clip([x1-offset, x2+offset], 0, frame.shape[1]-1)
    y1, y2 = np.clip([y1-offset, y2+offset], 0, frame.shape[0]-1)
    face_img = frame[y1:y2,x1:x2].copy()
    cv2.putText(frame, "{} / {}".format(format_face_text(face), 'training'), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
    cv2.rectangle(frame, *pts, (255, 0, 0), 3)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)
    if face.name == None and ask_name:
        face.name = input('Enter name: ')
    recognizer.save(recognizer_file)
    filename = os.path.join(images_path, '{}-{}.jpeg'.format(face.name, str(np.random.randint(2**32))))
    cv2.imwrite(filename, face_img)
    print('updated ' + format_face_text(face))
    return face

def draw_results(frame, dets, faces, dists):
    for det, face, dist in zip(dets, faces, dists):
        pts = imtools.dlib_rect2pts(det)
        cv2.rectangle(frame, *pts, (255, 0, 0), 1)
        if det is dets[0]:
            cv2.rectangle(frame, *pts, (255, 0, 0), 2)
        x, y = pts[0]
        if face == None:
            cv2.putText(frame, "{} / {:.2f}".format('???', dist), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, format_face_text(face) + ' / {:.2f}'.format(dist), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)

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

        dets, scores, subdetector_ids = detect_faces(frame)
        faces, descrs, dists = recognize_faces(frame, dets, dist_treshold)
        if auto_learn:
            learn_new_faces(frame, dets, scores, subdetector_ids, faces, descrs, dists)

        delay = 1 if type(video) != type((k for k in [])) else 100
        key = cv2.waitKey(delay) & 0xff
        if key == 27: break
        if (ord('0') <= key <= ord('9') or ord('a') <= key <= ord('z')) and len(descrs) > 0:
            det, descr, id = dets[0], descrs[0], key
            train_once(frame, det, descr, id)
            continue

        draw_results(frame, dets, faces, dists)

    print('Done reading frames')

if __name__ == '__main__':
    main()
