import dlib
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize global variables and models
try:
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    emotion_model_path = "_mini_XCEPTION.102-0.66.hdf5"

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Could not find facial landmark predictor file: {predictor_path}")
    if not os.path.exists(emotion_model_path):
        raise FileNotFoundError(f"Could not find emotion classifier model: {emotion_model_path}")

    logger.info("Loading predictor model...")
    predictor = dlib.shape_predictor(predictor_path)
    logger.info("Loading emotion classifier model...")
    emotion_classifier = load_model(emotion_model_path, compile=False)
    logger.info("Models loaded successfully")

except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    raise

points = []
points_lip = []

class VideoCamera(object):
    def __init__(self):
        logger.info("Initializing camera...")
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DSHOW backend on Windows
        
        if not self.video.isOpened():
            logger.error("Failed to open camera with DSHOW backend")
            # Try again with default backend
            self.video = cv2.VideoCapture(0)
            if not self.video.isOpened():
                logger.error("Failed to open camera with default backend")
                raise RuntimeError("Could not start camera. Please check if camera is connected and available.")
        
        logger.info("Camera initialized successfully")
            
    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()
            
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            logger.error("Failed to capture frame from camera")
            raise RuntimeError("Failed to capture frame from camera")
            
        try:
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=500, height=500)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector(gray, 0)
            
            for detection in detections:
                emotion = emotion_finder(detection, gray)
                cv2.putText(frame, emotion, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                shape = predictor(frame, detection)
                shape = face_utils.shape_to_np(shape)
                
                # Get facial landmarks indices
                (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
                (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
                (l_lower, l_upper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
                
                leyebrow = shape[lBegin:lEnd]
                reyebrow = shape[rBegin:rEnd]
                openmouth = shape[l_lower:l_upper]
                
                # Calculate convex hulls
                reyebrowhull = cv2.convexHull(reyebrow)
                leyebrowhull = cv2.convexHull(leyebrow)
                openmouthhull = cv2.convexHull(openmouth)
                
                # Draw contours
                cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [openmouthhull], -1, (0, 255, 0), 1)
                
                # Calculate distances
                lipdist = lpdist(openmouthhull[-1], openmouthhull[0])
                eyedist = ebdist(leyebrow[-1], reyebrow[0])
                
                # Calculate stress values
                stress_value, stress_label = normalize_values(points, eyedist, points_lip, lipdist)
                
                # Add text to frame
                cv2.putText(frame, emotion, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 52, 52), 2)
                cv2.putText(frame, f"stress value:{int(stress_value*100)}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 66, 232), 2)
                cv2.putText(frame, f"Stress level:{stress_label}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (35, 189, 25), 2)
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                raise RuntimeError("Failed to encode frame as JPEG")
                
            return jpeg.tobytes()
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            raise
            
    def plt_show():
        plot_stress=plt.plot(range(len(points)),points,'ro')
        plt.title("Stress Levels")
        plt.show()
        return plot_stress



#calculating eye distance in terms of the facial landmark
def ebdist(leye,reye):
    eyedist = dist.euclidean(leye,reye)
    points.append(int(eyedist))
    return eyedist

#calculating lip dostance using facial landmark
def lpdist(l_lower, l_upper):
    # Extract x,y coordinates from the hull points
    l_lower = l_lower.flatten()
    l_upper = l_upper.flatten()
    lipdist = dist.euclidean(l_lower, l_upper)
    points_lip.append(int(lipdist))
    return lipdist

#finding stressed or not using the emotions 
def emotion_finder(faces,frame):
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ['scared','sad','angry']:
        label = 'Stressed'
    else:
        label = 'Not Stressed'
    return label

#calculating stress value using the distances
def normalize_values(points,disp,points_lip,dis_lip):
    normalize_value_lip = abs(dis_lip - np.min(points_lip))/abs(np.max(points_lip) - np.min(points_lip))
    normalized_value_eye =abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    normalized_value =( normalized_value_eye + normalize_value_lip)/2
    stress_value = (np.exp(-(normalized_value)))
    if stress_value>=0.65:
        stress_label="High Stress"
    else:
        stress_label="Low Stress"
    return stress_value,stress_label
 
#processing real time video input to display stress 
