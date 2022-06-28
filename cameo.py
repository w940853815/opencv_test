from tkinter import Frame
import cv2
import filters
from managers import WindowManager, CaptureManager
import depth


class Cameo:
    def __init__(self) -> None:
        self._windowManager = WindowManager("Cameo", self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(1), self._windowManager, True
        )
        self._curverFilter = filters.BGRPortraCurveFilter()

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is not None:
                filters.strokeEdges(frame, frame)
                self._curverFilter.apply(frame, frame)
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        # 空格截图
        if keycode == 32:
            self._captureManager.writeImage("screenshot.png")
        # tab录屏
        elif keycode == 9:
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo("2.mp4")
        # esc 退出程序
        elif keycode == 27:
            self._windowManager.destroyWindow()
        else:
            self._captureManager.stopWritingVideo()


class CmameDepth(Cameo):
    def __init__(self) -> None:
        self._windowManager = WindowManager("Cameo", self.onKeypress)
        device = cv2.CAP_OPENNI2  # for kinect
        # device = cv2.CAP_OPENNI2_ASUS  # for Xtion
        print(device)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(device), self._windowManager, True
        )
        self._curverFilter = filters.BGRPortraCurveFilter()

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            self._captureManager.channel = cv2.CAP_OPENNI_DISPARITY_MAP
            disparityMap = self._captureManager.frame
            self._captureManager.channel = cv2.CAP_OPENNI_VALID_DEPTH_MASK
            validDepthMask = self._captureManager.frame
            self._captureManager.channel = cv2.CAP_OPENNI_BGR_IMAGE
            frame = self._captureManager.frame
            if frame is None:
                self._captureManager.channel = cv2.CAP_OPENNI_IR_IMAGE
                frame = self._captureManager.frame

            if frame is not None:
                mask = depth.createMedianMask(disparityMap, validDepthMask)
                frame[mask == 0] = 0

                if self._captureManager.channel == cv2.CAP_OPENNI_BGR_IMAGE:
                    filters.strokeEdges(frame, frame)
                    self._curverFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()


if __name__ == "__main__":
    # Cameo().run()
    CmameDepth().run()
