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

# 除了 SUT 高度为 mm，其余为 m
terms = 9  # legendre 多项式项数
Domain_scalar = 0.6


def gen_L2_surface(SUT_num, Xt, Yt):
    """
    用legendre多项式生成指定数量，点数的SUT

    SUT_num: SUT数量
    Xt: SUT X坐标, 转为[-1,1]后
    Yt: 意义同上
    """
    np.random.seed(11)
    C = np.random.rand(SUT_num, terms, terms)
    SUT = np.empty([SUT_num, Xt.shape[0], Yt.shape[0]])
    for i in range(SUT_num):
        SUT[i] = leggrid2d(Xt, Yt, C[i])
    return C, SUT


def cal_normal(C, Xt, Yt, dtdx, dtdy):
    """
    计算 L2 面每点单位法线
    n=\left[ -\frac{p}{N},-\frac{q}{N},-\frac{1}{N} \right], N=(1+p^2+q^2)^\frac{1}{2}

    C: 所有legendre SUT系数.例如[128,4,4]表示128个SUT，4x4 项的 legendre
    Xt: SUT X坐标, 转为[-1,1]后
    Yt: 意义同上
    """
    dx = np.empty([C.shape[0], Xt.shape[0], Yt.shape[0]])
    dy = np.empty(dx.shape)
    for i in range(C.shape[0]):
        dxc = legendre.legder(C[i], axis=0)
        dx[i] = leggrid2d(Xt, Yt, dxc) * dtdx
        dyc = legendre.legder(C[i], axis=1)
        dy[i] = leggrid2d(Xt, Yt, dyc) * dtdy
    maxdx = np.max(dx)
    mindx = np.min(dx)
    maxdy = np.max(dy)
    mindy = np.min(dy)
    N = np.sqrt(1 + np.square(dx) + np.square(dy))
    n = np.stack((-dx / N, -dy / N, 1 / N), axis=-1)  # 结果是单位矢量
    return dx, dy, n


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
    SUT = SUT / 1000  # 单位 mm 转为 m
    SUT_3D = np.stack((np.expand_dims(Xg, 0).repeat(SUT.shape[0], axis=0),
                       np.expand_dims(Yg, 0).repeat(SUT.shape[0], axis=0)
                       , SUT), axis=-1)
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


class SurfaceDataset(Dataset):
    def __init__(self, SUT_resolution, SUT_num, SUT_X_range, SUT_Y_range, camera_center, screen_point,
                 screen_normal, refer_point, refer_normal, transform=None):
        self.transform = transform
        self.SUT_resolution = SUT_resolution
        self.SUT_num = SUT_num
        self.SUT_X_range = SUT_X_range
        self.SUT_Y_range = SUT_Y_range
        self.camera_center = camera_center
        self.screen_point = screen_point
        self.screen_normal = screen_normal

        X = np.linspace(SUT_X_range[0], SUT_X_range[1], SUT_resolution[0])
        Y = np.linspace(SUT_Y_range[0], SUT_Y_range[1], SUT_resolution[1])
        Xt = (SUT_X_range.sum() - 2 * X) / (SUT_X_range[0] - SUT_X_range[1]) * Domain_scalar
        Yt = (SUT_Y_range.sum() - 2 * Y) / (SUT_Y_range[0] - SUT_Y_range[1]) * Domain_scalar

        dtdx = Domain_scalar * -2 / (SUT_X_range[0] - SUT_X_range[1]) / 1000
        dtdy = Domain_scalar * -2 / (SUT_Y_range[0] - SUT_Y_range[1]) / 1000
        self.C, self.SUT = gen_L2_surface(SUT_num, Xt, Yt)
        self.dx, self.dy, self.n = cal_normal(self.C, Xt, Yt, dtdx, dtdy)
        self.Incident, self.Reflect, self.reflect_screen = reflect(camera_center, X, Y, self.SUT, self.n, screen_point,
                                                                   screen_normal)

        self.refer_surface, self.refer_slope_x, self.refer_slope_y = reference(camera_center, self.Incident,
                                                                               self.reflect_screen, refer_point,
                                                                               refer_normal)
        self.refer_slope = np.stack((self.refer_slope_x, self.refer_slope_y), axis=-1)

    def __getitem__(self, index):
        slope, sut = self.refer_slope[index], self.SUT[index]
        if self.transform is not None:
            slope = self.transform(slope)

        return slope, sut

    def __len__(self):
        return len(self.refer_slope)


def loadDataset(is_test=False):
    f = open('cfg.json')
    cfg = json.load(f)
    f.close()

    camera_center = np.array(cfg['camera_center'])
    screen_point = np.array(cfg['screen_point'])
    screen_normal = np.array(cfg['screen_normal'])
    SUT_resolution = cfg['SUT_resolution']
    if is_test:
        SUT_num = 64
    else:
        SUT_num = cfg['SUT_num']
    SUT_X_range = np.array(cfg['SUT_X_range'])
    SUT_Y_range = np.array(cfg['SUT_X_range'])
    x_grad_limit = np.array(cfg['x_grad_limit'])
    y_grad_limit = np.array(cfg['y_grad_limit'])
    refer_point = np.array((0, 0, 0))
    refer_normal = np.array((0, 0, 1))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    SD = SurfaceDataset(SUT_resolution, SUT_num, SUT_X_range, SUT_Y_range, camera_center, screen_point,
                        screen_normal, refer_point, refer_normal, transform)

    return SD

if __name__ == '__main__':
    SD = loadDataset()
    SD.test_system()
