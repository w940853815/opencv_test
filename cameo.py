import cv2
import filters
from managers import WindowManager, CaptureManager


class Cameo:
    def __init__(self) -> None:
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(1), self._windowManager, True)
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
        
if __name__ == "__main__":
    Cameo().run()