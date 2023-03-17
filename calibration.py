import cv2 as cv
import numpy as np
import glob
import xml.etree.ElementTree as ET

class CameraCalibrator(object):
    def __init__(self, image_size:tuple):
        super(CameraCalibrator, self).__init__()
        self.image_size = image_size
        self.matrix = np.zeros((3, 3), np.float_)
        self.new_camera_matrix = np.zeros((3, 3), np.float_)
        self.dist = np.zeros((1, 5))
        self.roi = np.zeros(4, np.int_)

    def calibration(self, corner_height: int, corner_width: int, square_size: float):
        # 获取图片文件名列表
        file_names = glob.glob('../cal_chess/*.jpg') + glob.glob('../cal_chess/*.png') + glob.glob('../cal_chess/*.jpeg')
        # 储存棋盘格角点的世界坐标和图像坐标对
        objs_corner = []  # 在世界坐标系中的三维点
        imgs_corner = []  # 在图像平面的二维点
        # 设置迭代终止条件
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 计算真实角点坐标
        obj_corner = np.zeros([corner_height * corner_width, 3], np.float32)  # 构造0矩阵，用于存放角点的世界坐标
        obj_corner[:, :2] = np.mgrid[0:corner_height, 0:corner_width].T.reshape(-1, 2)  # 三维网格坐标划分
        # obj_corner = obj_corner * square_size

        for file_name in file_names:
            # 读取图片
            chess_img = cv.imread(file_name)
            assert (chess_img.shape[0] == self.image_size[1] and chess_img.shape[1] == self.image_size[0]), \
                "Image size does not match the given value {}.".format(self.image_size)
            # to gray
            gray = cv.cvtColor(chess_img, cv.COLOR_BGR2GRAY)
            # 粗略找到棋盘格角点 这里找到的是这张图片中角点的亚像素点位置，gray必须是8位灰度或者彩色图，（corner_height, corner_width）为角点规模
            ret, img_corners = cv.findChessboardCorners(gray, (corner_height, corner_width))
            # 如果找到足够点对，将其存储起来
            if ret:
                # 将正确的obj_corner点放入objs_corner中
                objs_corner.append(obj_corner)
                # 精确找到角点坐标（亚像素）
                '''其中winsize和zerozone'''
                img_corners = cv.cornerSubPix(gray, img_corners, winSize=(square_size // 2, square_size // 2),
                                              zeroZone=(-1, -1), criteria=criteria)
                # img_corners = cv.cornerSubPix(gray, img_corners, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
                imgs_corner.append(img_corners)
                # 将角点在图像上显示
                cv.drawChessboardCorners(chess_img, (corner_height, corner_width), img_corners, ret)
                cv.imwrite('sign_image/' + 'sign_' + file_name, chess_img)  # 保存图片
                # 展示角点检测的效果
                # cv.namedWindow('findCorners', 0)
                # cv.resizeWindow('findCorners', 1920, 1080)
                # cv.imshow('findCorners', chess_img)
                # cv.waitKey()
                # cv.destroyAllWindows()
            else:
                print("Fail to find corners in {}.".format(file_name))

        # 相机标定
        '''
        cv.calibrateCamera
         参数：
            objectPoints: 实际三维坐标，类型为 float32 的数组，是每张图片中的角点的三维坐标的列表。
            imagePoints: 每个角点的图像坐标，类型为 float32 的数组，是每张图片中的角点的图像坐标的列表。
            imageSize: 图像的尺寸，元组形式，例如（height, width）。
            cameraMatrix: 初始相机矩阵，可以是 None。
            distCoeffs: 相机的畸变系数，可以是 None。

        返回：
            ret: 标定结果，当为 True 时，说明标定成功。
            cameraMatrix: 相机矩阵，3x3 的矩阵。
            distCoeffs: 相机的畸变系数，1x5 或 1x8 的数组。
            rvecs: 每张图片对应的旋转向量，与旋转矩阵对应。
            tvecs: 每张图片对应的平移向量，与旋转矩阵对应。
        '''
        ret, self.matrix, self.dist, rvecs, tveces = cv.calibrateCamera(objs_corner, imgs_corner, self.image_size, None, None, flags=cv.CALIB_RATIONAL_MODEL)
        # ret, matrix, dist, rvecs, tveces = cv.calibrateCamera(objs_corner, imgs_corner, gray.shape[::-1], None, None, flags=cv.CALIB_RATIONAL_MODEL)
        # ret, self.matrix, self.dist, rvecs, tveces = cv.calibrateCamera(objs_corner, imgs_corner, self.image_size, None, None)
        # CV_CALIB_RATIONAL_MODEL：计算k4，k5，k6三个畸变参数。如果没有设置，则只计算其它5个畸变参数。
        '''
        cv.getOptimalNewCameraMatrix
        参数：
            cameraMatrix：相机矩阵，3x3 的数组。
            distCoeffs：相机的畸变系数，1x5 或 1x8 的数组。
            imageSize：图像的大小，是一个二元组（宽度，高度）。
            alpha：要投影的 ROI 区域的角落处的额外增加的空间的因子，以便不剪切角落。
            如果 alpha=0，表示没有剪切。如果 alpha=1，表示完全额外增加空间。

        返回值：
            newCameraMatrix：计算出的最佳新相机矩阵。
            validPixROI：最终的图像 ROI，它是一个四元组（x，y，width，height），表示一个矩形。
        '''
        self.new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(self.matrix, self.dist, self.image_size, alpha=1)
        self.roi = np.array(roi)

        return ret

    def show_calibration_result(self, test_file):
        img = cv.imread(test_file)
        dst = cv.undistort(img, self.matrix, self.dist, None, self.new_camera_matrix)
        sv_im = 'adjust_image/' + 'adjust.jpg'
        # cv.imwrite(sv_im, dst)  # 保存图片

        # 展示畸变矫正的效果
        cv.namedWindow('undistort_picture', 0)
        cv.resizeWindow('undistort_picture', 1920, 1080)
        cv.imshow('undistort_picture', dst)
        cv.waitKey()
        cv.destroyAllWindows()

    def rectify_image(self, img):
        if not isinstance(img, np.ndarray):
            AssertionError("Image type '{}' is not numpy.ndarray.".format(type(img)))
        dst = cv.undistort(img, self.matrix, self.dist, self.new_camera_matrix)
        x, y, w, h = self.roi
        dst = dst[y:y + h, x:x + w]
        dst = cv.resize(dst, (self.image_size[0], self.image_size[1]))

        print(self.matrix, 'matrix\n', self.dist, 'dist\n', self.new_camera_matrix, 'new_camera_matrix\n')
        '''可以增加 将数组写进xml文件中'''

        return dst

    def rectify_video(self, video_path: str):
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print("Unable to open video.")
            return False
        fourcc = int(cap.get(cv.CAP_PROP_FOURCC))
        out_format = video_path.split('.')[-1]
        fps = int(cap.get(cv.CAP_PROP_FPS))
        out = cv.VideoWriter(filename='out.'+out_format, fourcc=0x00000021, fps=fps, frameSize=self.image_size)
        cv.namedWindow("origin", cv.WINDOW_NORMAL)
        cv.namedWindow("dst", cv.WINDOW_NORMAL)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        for _ in range(frame_count):
            ret, img = cap.read()
            if ret:
                img = cv.resize(img, (self.image_size[0], self.image_size[1]))
                cv.imshow("origin", img)
                dst = self.rectify_image(img)
                cv.imshow("dst", dst)
                out.write(dst)
                cv.waitKey(1)
        cap.release()
        out.release()
        cv.destroyAllWindows()
        return True


if __name__ == '__main__':
    CameraCalibrator = CameraCalibrator(image_size=(3840, 2160))
    ret = CameraCalibrator.calibration(corner_height=12, corner_width=8, square_size=31)
    CameraCalibrator.show_calibration_result(test_file='../test_image/Image.jpg')
    CameraCalibrator.rectify_video('../test.mp4')

    print(ret)