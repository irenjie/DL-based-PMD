import numpy as np
from numpy.polynomial import legendre
from numpy.polynomial.legendre import leggrid2d
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt
import open3d as o3d
from torchvision.transforms import transforms
from pylab import *
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

np.polynomial.set_default_printstyle('unicode')

clist = ['darkblue', 'blue', 'limegreen', 'orange', 'yellow']
mycmp = LinearSegmentedColormap.from_list('chaos', clist)


def cal_normal(dx_actual, dy_actual):
    """
    计算面每点单位法线
    n=\left[ -\frac{p}{N},-\frac{q}{N},-\frac{1}{N} \right], N=(1+p^2+q^2)^\frac{1}{2}
    """
    N = np.sqrt(1 + np.square(dx_actual) + np.square(dy_actual))
    n = np.stack((-dx_actual / N, -dy_actual / N, 1 / N), axis=-1)  # 结果是单位矢量
    return n


def reflect(camera_center, X, Y, SUT, n, screen_point, screen_normal):
    """
    计算反射光线
    r=i-2(i*n)n
    反射光线与屏幕交点（直线与平面交点）
    𝑡=(𝑣_𝑎 (𝑥_1−𝑥_0 )+𝑣_𝑏 (𝑦_1−𝑦_0)+𝑣_𝑐 (𝑧_1−𝑧_0))/(𝑣_1 𝑣_𝑎+𝑣_2 𝑣_𝑏+𝑣_3 𝑣_𝑐 )

    camera_center: 相机光心坐标
    X, Y: SUT 的 X, Y坐标
    SUT: 所有SUT高度
    n: 所有SUT的每点法线
    screen_point: 屏幕上一点
    screen_normal: 屏幕法线
    """
    # SUT点减去光心得到入射光线
    Xg, Yg = np.meshgrid(X, Y)
    Xg = Xg.T  # 经分析，meshgrid 的坐标转置后才正确
    Yg = Yg.T
    SUT_3D = np.stack((Xg, Yg, SUT), axis=-1)
    I = SUT_3D - camera_center  # 应为减去 camera_center，注意入射光线方向!!!
    temp = (I * n).sum(axis=-1)
    Reflect = I - 2 * np.stack((temp, temp, temp), axis=-1) * n  # 反射光线

    t1 = ((screen_point - SUT_3D) * screen_normal).sum(axis=-1)
    t2 = (screen_normal * Reflect).sum(axis=-1)
    t = t1 / t2
    reflect_screen = np.stack((t, t, t), axis=-1) * Reflect + SUT_3D  # 反射光线与屏幕交点
    return I, Reflect, reflect_screen


def reference(camera_center, Incident, reflect_screen, refer_point, refer_normal):
    """
    计算入射光线与参考平面交点，进而求出 height-slope ambiguity 下的 slope
    """
    t1 = ((refer_point - camera_center) * refer_normal).sum()
    t2 = (refer_normal * Incident).sum(axis=-1)
    t = t1 / t2
    refer_surface = np.stack((t, t, t), axis=-1) * Incident + camera_center  # 入射光线与参考平面交点

    # 反射3点求斜率公式
    dr2c = np.sqrt(np.square(refer_surface - camera_center).sum(axis=-1))
    dr2s = np.sqrt(np.square(refer_surface - reflect_screen).sum(axis=-1))
    xr = refer_surface[..., 0]  # 参考面
    yr = refer_surface[..., 1]
    zr = refer_surface[..., 2]

    xc = camera_center[..., 0]  # 相机
    yc = camera_center[..., 1]
    zc = camera_center[..., 2]

    xs = reflect_screen[..., 0]  # 屏幕
    ys = reflect_screen[..., 1]
    zs = reflect_screen[..., 2]

    denominator = (zc - zr) / dr2c + (zs - zr) / dr2s
    refer_slope_x = ((xr - xc) / dr2c + (xr - xs) / dr2s) / denominator
    refer_slope_y = ((yr - yc) / dr2c + (yr - ys) / dr2s) / denominator

    return refer_surface, refer_slope_x, refer_slope_y


class spherical_surface():
    def __init__(self, SUT_resolution, SUT_X_range, SUT_Y_range, camera_center, screen_point,
                 screen_normal, refer_point, refer_normal):
        self.SUT_resolution = SUT_resolution
        self.SUT_X_range = SUT_X_range
        self.SUT_Y_range = SUT_Y_range
        self.camera_center = camera_center
        self.screen_point = screen_point
        self.screen_normal = screen_normal

        X = np.linspace(SUT_X_range[0], SUT_X_range[1], SUT_resolution[0])
        Y = np.linspace(SUT_Y_range[0], SUT_Y_range[1], SUT_resolution[1])
        X1, Y1 = np.meshgrid(X, Y)
        X1 = X1.T
        Y1 = Y1.T
        R = 0.25  # radis，单位 m
        c = 1.0 / R
        FenZi = c * (np.square(X1) + np.square(Y1))
        FenMu = 1 + np.sqrt(1 - c * c * (np.square(X1) + np.square(Y1)))
        self.Z = FenZi / FenMu  # 高度，单位 m
        # 真实斜率
        self.dx_actual = (2 * c * X1 * FenMu + FenZi / (FenMu - 1) * 2 * c * c * X1) / np.square(FenMu)
        self.dy_actual = (2 * c * Y1 * FenMu + FenZi / (FenMu - 1) * 2 * c * c * Y1) / np.square(FenMu)

        # 真实法线
        self.n = cal_normal(self.dx_actual, self.dy_actual)
        self.Incident, self.Reflect, self.reflect_screen = reflect(camera_center, X, Y, self.Z, self.n, screen_point,
                                                                   screen_normal)

        self.refer_surface, self.refer_slope_x, self.refer_slope_y = reference(camera_center, self.Incident,
                                                                               self.reflect_screen, refer_point,
                                                                               refer_normal)
        self.refer_slope = np.stack((self.refer_slope_x, self.refer_slope_y), axis=-1)

    def getitem(self):
        return self.refer_slope, self.Z

    def __len__(self):
        return len(self.refer_slope)

    def test_surface(self):
        X = np.linspace(self.SUT_X_range[0], self.SUT_X_range[1], self.SUT_resolution[0]) * 1000
        Y = np.linspace(self.SUT_Y_range[0], self.SUT_Y_range[1], self.SUT_resolution[1]) * 1000
        X1, Y1 = np.meshgrid(X, Y)
        X1 = X1.T
        Y1 = Y1.T
        fig = plt.figure()
        ax3d = fig.add_subplot(projection='3d')

        ax3d.plot_surface(X1, Y1, self.Z * 1000, cmap=mycmp)

        ax3d.set_zticks(np.arange(-0.5, 2.5, 0.5))
        ax3d.set_xlabel('X(mm)')
        ax3d.set_ylabel('Y(mm)')
        ax3d.set_zlabel('Z(mm)')
        plt.show(block=True)


def load_sphere():
    f = open('cfg.json')
    cfg = json.load(f)
    f.close()

    camera_center = np.array(cfg['camera_center'])
    screen_point = np.array(cfg['screen_point'])
    screen_normal = np.array(cfg['screen_normal'])
    SUT_resolution = cfg['SUT_resolution']
    SUT_X_range = np.array(cfg['SUT_X_range'])
    SUT_Y_range = np.array(cfg['SUT_X_range'])
    refer_point = np.array((0, 0, 0))
    refer_normal = np.array((0, 0, 1))

    SD = spherical_surface(SUT_resolution, SUT_X_range, SUT_Y_range, camera_center, screen_point,
                           screen_normal, refer_point, refer_normal)

    return SD


if __name__ == '__main__':
    SD = load_sphere()
    SD.test_surface()
