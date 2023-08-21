import face_recognition as  frec
import os, sys
import cv2
import numpy as np
import math


# This function return the value of how much accuracy is in your face to your image
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    faceLocations = []      # faceLocation is for storing the rgb frames for matching
    faceEncodings = []      # faceEncoding is for encode the image for the video
    faceNames = []          # faceNames is store the name and confidence for the video
    knownFaceEncodings = [] # knownFaceEncodings is for the store the encoded of the external image
    knownFaceNames = []     # knownFaceNames is for the the store name in order to added a image in knownFaceEncodings
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
    def encode_faces(self):
        # this method get the images in form of encoded system , image name
        shivam = frec.load_image_file('Shivam.jpg')
        shivam_encoding = frec.face_encodings(shivam)[0]
        keyush = frec.load_image_file('Keyush.jpg')
        keyush_encoding = frec.face_encodings(keyush)[0]  
        kunal = frec.load_image_file('Kunal.jpg')
        kunal_encoding = frec.face_encodings(kunal)[0]
        yash = frec.load_image_file('Yash.jpg')
        yash_encoding = frec.face_encodings(yash)[0]
        manav = frec.load_image_file('Manav.jpg')
        manav_encoding = frec.face_encodings(manav)[0]
        self.knownFaceEncodings=[shivam_encoding,keyush_encoding,kunal_encoding,yash_encoding,manav_encoding]
        self.knownFaceNames=['shivam','keyush','kunal','yash','manav']
        print(self.knownFaceNames)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)     # This will take Web-cam source

        if not video_capture.isOpened():    #This will check your camera found or not
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()   # Take the frame if avialable then ret = True else false
            frame = cv2.flip(frame,1)# flip your web cam video

            cv2.putText(frame , 'L.J. University',(10,30),cv2.FONT_HERSHEY_SIMPLEX , 0.8 , (255, 0, 0),2)   # Put the L.J. university  name in corner in live cam (BGR)


            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)   # set the video frame and resize 1/4
                rgb_small_frame = small_frame[:, :, ::-1]   # return  the bgr small_frame in the form of rgb_small_frame
                self.faceLocations = frec.face_locations(rgb_small_frame)
                self.faceEncodings = frec.face_encodings(rgb_small_frame, self.faceLocations)
                self.faceNames = []
                for face_encoding in self.faceEncodings:        # Check if  the video person is known or unknown
                    matches = frec.compare_faces(self.knownFaceEncodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'
                    face_distances = frec.face_distance(self.knownFaceEncodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:   # if person is known then return the face name and confidence(percentage of accuracy)
                        name = self.knownFaceNames[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        
                    self.faceNames.append(f'{name} ({confidence})')
            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.faceLocations, self.faceNames):        # make a annotation box and name box for live web came
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)   #   return the full functional face recognition window
            v=cv2.waitKey(1)
            if  v== ord('c'):       # Exit key = 'c'
                break
        video_capture.release()     #   Release video after caputer and clear the buffer
        cv2.destroyAllWindows()


try:
    fr = FaceRecognition()
    fr.run_recognition()
except:
    print('You have an error ocured solve this')

