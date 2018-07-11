import numpy as np
import pickle

class Face():
    def __init__(self, id):
        self.id = id
        self.name = None
        self.descriptors = []

class Recognizer():
    def __init__(self, filename=None):
        self.faces = []
        if filename == None: return
        try:
            self.load(filename)
            print('Loaded recognizer from file')
        except:
            print('Recognizer file did not exist')

    def update(self, id, descriptor, name=None):
        face = self.find(id)
        if face == None:
            face = Face(id)
            self.faces.append(face)
        if name != None: face.name = name
        face.descriptors.append(descriptor)
        return face

    def match(self, descriptor):
        closest_face = None
        min_dist = np.inf
        for face in self.faces:
            for descr in face.descriptors:
                dist = np.linalg.norm(descr-descriptor)
                if dist < min_dist:
                    min_dist = dist
                    closest_face = face
        return closest_face, min_dist

    def find(self, id):
        face = next((f for f in self.faces if f.id == id), None)
        return face

    def load(self, filename):
        faces = []
        file = open(filename, 'rb')
        faces = pickle.load(file)
        self.faces = faces

    def save(self, filename):
        file = open(filename, '+wb')
        pickle.dump(self.faces, file)