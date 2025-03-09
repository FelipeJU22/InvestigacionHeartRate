import numpy as np
import cv2
import sys
from cvzone.FaceDetectionModule import FaceDetector
import cvzone
import time

realWidth = 640
realHeight = 480
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15

# Video Parameters
test = "test4"
video_path = f'Pruebas/{test}.mp4'
video = cv2.VideoCapture(video_path)

output_video_path = f'ResultadoHeartRate/{test}.avi'

detector = FaceDetector()

# Obtener el número total de frames
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Imprimir el resultado
print(f"Número total de frames: {total_frames}")

fps_out = int(video.get(cv2.CAP_PROP_FPS))

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps_out, (width, height))

video.set(3, realWidth)
video.set(4, realHeight)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (30, 40)
bpmTextLocation = (videoWidth // 2, 40)
fpsTextLocation = (500, 600)

fontScale = 1
fontColor = (0, 0, 0)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 10
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))
bpm_values = []

i = 0
ptime = 0
ftime = 0
while True:
    ret, frame = video.read()
    if not ret:
        break

    frame, bboxs = detector.findFaces(frame, draw=False)
    frameDraw = frame.copy()
    ftime = time.time()
    fps = 1 / (ftime - ptime)
    ptime = ftime

    cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), font, fontScale, fontColor, thickness=2, lineType=cv2.LINE_AA)
    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
        detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
        detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Grab a Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            i += 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize
        outputFrame_show = cv2.resize(outputFrame, (videoWidth // 2, videoHeight // 2))
        frameDraw[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show

        bpm_value = bpmBuffer.mean()

        if i > bpmBufferSize:
            cvzone.putTextRect(frameDraw, f'BPM: {bpm_value}', bpmTextLocation, font=font, scale=1, colorR=fontColor, colorB=boxColor)
            bpm_values.append(bpm_value)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation, font=font, scale=1, colorR=fontColor, colorB=fontColor)

        cv2.imshow("Heart Rate Monitor", frameDraw)

        # Write the processed frame to the output video
        out.write(frameDraw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imshow("Heart Rate Monitor", frameDraw)

        # Write the processed frame to the output video
        out.write(frameDraw)

# Release resources
video.release()
out.release()
cv2.destroyAllWindows()