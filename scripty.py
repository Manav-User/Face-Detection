import cv2
def generate_dataset(img, id, img_id):
    cv2.imwrite("dataset/user."+str(id)+"."+str(img_id)+".jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color_dict, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        if id == 1:
            name = "Manav"
            color = color_dict['manav']
        elif id == 2:
            name = "Zalak"
            color = color_dict['zalak']
        elif id == 3:
            name = "Tejas"
            color = color_dict['tejas']
        elif id == 4:
            name = "Avani"
            color = color_dict['avani']
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, name, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords.append([x, y, w, h])
    return coords

def recognize(img, clf, faceCascade):
    color_dict = {"manav":(255,255,255), "zalak":(128,0,128), "tejas":(255,105,180), "unknown":(0,0,255), "avani":(225,0,0)}
    coords = draw_boundary(img, faceCascade, 1.1, 15, color_dict, "Face", clf)
    return img

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouthCascade = cv2.CascadeClassifier("Mouth.xml")
noseCascade = cv2.CascadeClassifier("Nariz.xml")

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video_capture = cv2.VideoCapture(0)

img_id = 0

while True:
    _, img = video_capture.read()   
    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face Detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()