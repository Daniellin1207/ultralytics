# minAreaRect
import random
import numpy as np
import cv2


class OperateImg():
    def __init__(self, imgPath,img = None):
        if imgPath:
            print(imgPath)
            self.image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        elif img is not None:
            self.image = img
            # self.image = cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE)
        else:
            return
        w,h = self.image.shape[1],self.image.shape[0]
        self.image = cv2.copyMakeBorder(self.image, h//2, h//2, w//2, w//2, cv2.BORDER_CONSTANT, value=0)
        self.imageWidth, self.imageHeight = self.image.shape[1],self.image.shape[0]
        self.imageCopy = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGBA)

    def GenerateDrawBackPic(self, width=200, height=500, hgap=50, wgap=50) -> cv2.typing.MatLike:
        # 创建一个 500x500 的透明图像（四个通道，最后一个通道表示透明度）
        w, h = int(width), int(height)
        image = np.zeros((h, w, 4), dtype=np.uint8)
        # 将图像的透明度通道设置为完全透明
        image[:, :, 3] = 0
        # 计算每一份的高度
        hstep = h // hgap
        wstep = w // wgap
        for i in range(wstep + 1):
            # 生成随机颜色（不包括透明度通道）
            color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 200))
            for j in range(1, hstep + 1):
                y = j * hgap
                cv2.line(image, (i * wgap, y), (i * wgap + wgap, y), color, 2)
        return image

    def MixTextureAndBackPic(self,backPic,texture):
        # self.OutputNoImage(backPic,"backPic")
        # self.OutputNoImage(texture,"texture")
        sizeBack = backPic.shape
        sizeText = texture.shape
        bw, bh = sizeBack[1], sizeBack[0]  # 宽度 # 高度
        tw, th = sizeText[1], sizeText[0]
        # 创建一个 500x500 的透明图像（四个通道，最后一个通道表示透明度
        image = np.zeros((th, tw, 4), dtype=np.uint8)

        for i in range(tw):
            for j in range(th):
                try:
                    tmpBack = backPic[j][i]
                    tmpText = texture[j][i]
                    tmpValue = tmpBack * tmpText
                    if tmpBack > 0:
                        image[j][i] = tmpValue if sum(tmpValue) else [255, 255, 255, 255]
                    # image[centery+j][centerx+i] = backPic[centery+j][centerx+i]*texture[half_minAreaWidth+j][half_minAreaHeight+i]
                except Exception as e:
                    print(f"An exception occurred: {e}")
                    import traceback
                    traceback.print_exc()
        # self.OutputImage(image,"Mixture")
        return image

    def TextureContent(self, backPic, minAreaWidth, minAreaHeight):
        # flg = backPic.shape[0]>backPic.shape[1]
        assert minAreaWidth>minAreaHeight
        texture = self.GenerateDrawBackPic( minAreaWidth, minAreaHeight,hgap=5, wgap=50)
        # self.OutputImage(texture,"texture")
        return self.MixTextureAndBackPic(backPic,texture)
    def OperateSingleContour(self, contour):
        # 获取轮廓的最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 获取矩形的中心点、角度和尺寸
        center, (width, height), angle = rect
        # (centerX,centerY), (width, height), angle = rect
        # cv2.drawContours(self.imageCopy, [box], 0, (0, 0, 255), 2)
        # 找到最小外接矩形的外接矩形
        boundingrect = cv2.boundingRect(box)
        blx, bly, bwidth, bheight = boundingrect
        # cv2.rectangle(self.imageCopy,(blx,bly),(blx+bwidth,bly+bheight),(255,0,0),2)
        # self.OutputImage(self.imageCopy)
        # print(center,centerX,centerY,blx,bly)
        if blx<0 or bly<0:
            raise "blx < 0 or bly < 0"
        # 拿到该区域的图片内容
        contour_content = self.image[bly:bly+bheight, blx:blx+bwidth].copy()
        # self.OutputNoImage(contour_content,"get content")
        # 旋转最小外接矩形的角度，长边在下
        transX,transY = bwidth/2,bheight/2
        rotation_matrix = cv2.getRotationMatrix2D((transX,transY), angle, 1)
        rotation_matrix[0, 2] += (width - bwidth) / 2  # 重点在这步，目前不懂为什么加这步
        rotation_matrix[1, 2] += (height - bheight) / 2  # 重点在这步
        rotated_content = cv2.warpAffine(contour_content, rotation_matrix, (int(width), int(height)))
        # self.OutputNoImage(rotated_content,"8.rotate_content")
        # 在图上白色区域盖一层
        texture_content = self.TextureContent(rotated_content, width, height)
        # self.OutputNoImage(texture_content,"textureContent")
        # 回复旋转最小外接矩形的角度
        rotation_matrix = cv2.getRotationMatrix2D((int(width)/2, int(height)/2), - angle, 1)
        rotation_matrix[0, 2] += (bwidth - width) / 2  # 重点在这步，目前不懂为什么加这步
        rotation_matrix[1, 2] += (bheight - height) / 2  # 重点在这步
        second_rotated_content = cv2.warpAffine(texture_content, rotation_matrix, (bwidth, bheight))
        self.OutputNoImage(second_rotated_content,"9.second_rotated_content")

        # 将图片填充回原图
        try:
            self.imageCopy[bly:bly+bheight, blx:blx+bwidth] += second_rotated_content
            self.OutputImage(self.imageCopy,"Origin")
        except Exception as e:
            print(f"An exception occurred: {e}")
            import traceback
            traceback.print_exc()
            print("hello")
            return False
        return True

    def OperateContours(self):  # 输出绘制好覆盖线的图像
        # 查找轮廓
        contours, _ = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            cv2.destroyAllWindows()
            flg = self.OperateSingleContour(contour)
            print("operation flag",flg)
            # self.OutputImage(self.imageCopy,"iteration")

    def OutputNoImage(self, image, name="Pic"):
        cv2.imshow(name, image)
        cv2.resizeWindow(name,600,600)
    def OutputImage(self, image, name="Pic"):
        # cv2.imshow(name, image)
        # cv2.resizeWindow(name,600,600)
        # cv2.waitKey()
        return

if __name__ == '__main__':
    # tmp = OperateImg("binPic.png")
    tmp = OperateImg("local.png")
    tmp.OperateContours()
    tmp.OutputImage(tmp.imageCopy)
    cv2.imshow("dfdf",tmp.imageCopy)
    cv2.waitKey()



# def generate_random_points(self):
#     points = []
#     for _ in range(10):
#         x = random.randint(0, 100)
#         y = random.randint(0, 100)
#         points.append((x, y))
#     return points
#
#
# def draw_min_area_rectangle(self):
#     points = self.generate_random_points()
#
#     # 提取后五个点
#     last_five_points = points[5:]
#
#     # 将点转换为 numpy 数组
#     points_array = np.array(last_five_points, dtype=np.float32)
#
#     # 计算最小外接矩形
#     rect = cv2.minAreaRect(points_array)
#
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#
#     # 绘制点
#     img = np.zeros((120, 120, 3), dtype=np.uint8)
#     for point in points:
#         cv2.circle(img, tuple(point), 2, (255, 0, 0), -1)
#
#     # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
#
#     # 输出矩形信息
#     center = rect[0]
#     width, height = rect[1]
#     angle = rect[2]
#     print(f"Center: {center}")
#     print(f"Width: {width}, Height: {height}")
#     print(f"Angle: {angle}")
#     # self.OutputImage('Points and Rectangle',img)
#     # cv2.destroyAllWindows()
#
#
# # def rotate_image(self,image,angle = 45, savew = 500,saveh =500):
# #     # 旋转图像
# #     rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle-90, 1)
# #     rotated_content = cv2.warpAffine(image, rotation_matrix, (savew, saveh))
# #     return rotated_content
# def get_min_area_rec(self):
#     # 查找轮廓
#     contours, _ = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # 创建一个副本用于绘制
#     # drawn_image = self.image.copy()
#     for contour in contours:
#         # 绘制轮廓
#         # cv2.drawContours(drawn_image, [contour], -1, 128, 2)
#         # 获取轮廓的最小外接矩形
#         rect = cv2.minAreaRect(contour)
#         boundingrect = cv2.boundingRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#
#         # 获取矩形的中心点、角度和尺寸
#         center, (width, height), angle = rect
#         contour_content, rotated_content = self.rotate_image(contour, angle)
#
#         # self.OutputImage(contour_content,'Original Contour Content')
#         # self.OutputImage(rotated_content,'Rotated Contour Content')
#     # 显示原始二值图和带有轮廓及斜线的图像
#     # self.OutputImage(self.image,'Original Binary Image')
#     cv2.destroyAllWindows()
