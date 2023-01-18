import cvzone
import cv2
import os 
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(1)
detector = PoseDetector()
""" FIND THE ROOT DIRECTORTY TO CONVERT THE IMAGES INTO LIST """
shirtFolderPath = "C:\DEVELOPER\Fitting-Clothes\Resources\Resources\Shirts"
""" GETTING THE LIST INTO THE PATH  """
listShirts = os.listdir(shirtFolderPath)
""" PRINT THE LIST """
#print(listShirts)
fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440 #pixels of our images
imageNumber = 0
imgButtonRight = cv2.imread("Resources\Resources\button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    # img = cv2.flip(img,1)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    if lmList:
        #center = bboxInfo["center"]
        """ READ LANDMARK FROM SHOULDER AND THE POINT (X,Y) IN THE ARRAY """
        landMark11 = lmList[11][1:3]
        landMark12 = lmList[12][1:3]
        """ PUT THE PATH AND THE IMAGES TOGETHER IN imgShirt VARIABLE  """
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

        imgShirt = cv2.resize(imgShirt,(0,0), None,0.5,0.5)
        
        """ CURRENT SCALE IT IS THE WIDTH
        FROM THE 11 AND 12 LANDMARK """
        currentScale = (landMark11[0] - landMark12[0]) / 190

        offset = int(44 * currentScale), int(48 * currentScale)
        try:
            img = cvzone.overlayPNG(img, imgShirt, (landMark12[0] - offset[0], landMark12[1] - offset[1]))
        except:
            pass

        img = cvzone.overlayPNG(img, imgButtonRight, (1074,293))
        img = cvzone.overlayPNG(img, imgButtonLeft, (72,293))

        """ IF LANDMARK IT IS EQUALS TO A WRIST POSITION AND THE X AXIS: """
        if lmList[16][1] < 300:
            counterRight += 1
            cv2.ellipse(img, (139,360),(66,66),0,0,counterRight*selectionSpeed,(0,255,0),20)
            if counterRight*selectionSpeed>360:
                counterRight = 0
                imageNumber +=1
        else:
            counterRight = 0
            counterLeft = 0



    cv2.imshow("Image", img)
    cv2.waitKey(1)
