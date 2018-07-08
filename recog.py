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
        if filename != None: self.load(filename)

    def update(self, id, descriptor, name=None):
        face = next((f for f in self.faces if f.id == id), Face(id))
        if name != None: face.name = name
        face.descriptors.append(descriptor)
        self.faces.append(face)
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
        try:
            file = open(filename, 'rb')
            faces = pickle.load(file)
            print('Loaded recognizer from file')
        except:
            print('Could not load recognizer from file, a new one will be created')
        self.faces = faces

    def save(self, filename):
        file = open(filename, '+wb')
        pickle.dump(self.faces, file)