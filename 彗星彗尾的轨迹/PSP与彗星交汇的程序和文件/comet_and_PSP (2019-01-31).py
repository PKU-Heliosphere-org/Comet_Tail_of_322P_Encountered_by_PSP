import math
import mayavi.mlab as mlab
import re
import json
import numpy
import colorsys
from poliastro.twobody import Orbit
from poliastro.frames import get_frame, Planes
from poliastro.bodies import Sun, Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
import poliastro.bodies as bodies
import poliastro.constants as constants
from poliastro import ephem

from astropy import units as u
from astropy import time
from astropy.coordinates import SkyCoord
import astropy.coordinates as coordinates
from astropy.constants import G

import datetime

from pandas import Series
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from jdcal import gcal2jd, jd2gcal
from numpy import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter, attrgetter
from_deg2rad = math.pi/180  # 角度向弧度的转换

'''
定义类型MilkyWay，实现各种坐标转换的功能
定义类型SolarSystem,根据日期实现星历的获取
定义函数get_xyz_eclip_planet，根据所提供的日期时间，给出行星天体在国际天球参考系中位置，进一步通过坐标变换，得到行星天体在日心黄道坐标系中的位置
'''
class MilkyWay:
    def __init__(self):
        self.frame_name = 'heliocentrictrueecliptic'
        self.frame_type = coordinates.Galactic()
        self.gala_coo0 = []
        self.gala_coo1 = []
        self.gala_coo2 = []
        self.gala_coo3 = []
        self.x_multiply = []
        self.y_multiply = []
        self.z_multiply = []

        # galcen_distance = self.distance * units.AU)

    def calculate_coe(self):
        loc = coordinates.SkyCoord(0, 0, 0, unit='AU', frame = self.frame_name, representation='cartesian')
        res = loc.transform_to(self.frame_type)
        self.gala_coo0 = res.cartesian.xyz.to('AU').value

        loc = coordinates.SkyCoord(1, 0, 0, unit='AU', frame = self.frame_name, representation='cartesian')
        res = loc.transform_to(self.frame_type)
        self.gala_coo1 = res.cartesian.xyz.to('AU').value

        loc = coordinates.SkyCoord(0, 1, 0, unit='AU', frame = self.frame_name, representation='cartesian')
        res = loc.transform_to(self.frame_type)
        self.gala_coo2 = res.cartesian.xyz.to('AU').value

        loc = coordinates.SkyCoord(0, 0, 1, unit='AU', frame = self.frame_name, representation='cartesian')
        res = loc.transform_to(self.frame_type)
        # res = loc.transform_to(coordinates.ICRS())
        self.gala_coo3 = res.cartesian.xyz.to('AU').value

        self.x_multiply = self.gala_coo1 - self.gala_coo0
        self.y_multiply = self.gala_coo2 - self.gala_coo0
        self.z_multiply = self.gala_coo3 - self.gala_coo0

    def vector_cal(self, x, y, z):
        return x * self.x_multiply + y * self.y_multiply + z * self.z_multiply + self.gala_coo0

    def trans_to_ecliptic(self, x, y, z):
        return x * self.x_multiply + y * self.y_multiply + z * self.z_multiply + self.gala_coo0

    def gala_to_ecli(self):
        self.frame_name = 'galactic'
        self.frame_type = coordinates.HeliocentricTrueEcliptic()

    def icrs_to_ecli(self):
        self.frame_name = 'icrs'
        self.frame_type = coordinates.HeliocentricTrueEcliptic()

    def icrs_to_gala(self):
        self.frame_name = 'icrs'
        self.frame_type = coordinates.Galactic()

class SolarSystem:
    def __init__(self):
        self.sun = Sun
        self.bodies = [Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune]
        self.color_map = ['black','blue','cyan','blue','green','yellow','red','magenta']*10
        self.locations_dict = {}
        self.loc_val_dict = {}
        self.orbit_dict = {}
        self.n_point_per_orbit = 200
        self.coord_time = None

    def calculate_locations(self, the_time=datetime.datetime(2018, 12, 22, 15, 19, 29)):
        import de405
        from jplephem import Ephemeris

        self.locations_dict = {star : Orbit.from_body_ephem(star, time.Time(the_time)) for star in self.bodies}

        # Add pluto.
        self.bodies.append(Pluto)
        eph405 = Ephemeris(de405)
        pos_pluto, vec_pluto = eph405.position_and_velocity('pluto', time.Time(the_time).jd1)
        pos_pluto = [x[0] for x in pos_pluto]
        vec_pluto = [x[0] for x in vec_pluto]
        units = u
        pluto_orbit = Orbit.from_vectors(Sun, pos_pluto * units.km, vec_pluto * units.km / units.day, epoch=time.Time(the_time))
        self.locations_dict[Pluto] = pluto_orbit

        # Add Chiron.
        chiron_zero_time = time.Time(datetime.datetime(2008, 11, 4, 0, 0, 0))
        chiron_loc = (1.176533659248079E+01, -9.875630634479151E+00, 2.362986001474300E+00) * units.AU
        chiron_speed = (3.569754736654343E-03, 1.774888491950930E-03, 7.769506785978814E-04) * units.AU / units.day
        chiron_orbit = Orbit.from_vectors(Sun, chiron_loc, chiron_speed, epoch=chiron_zero_time)
        chiron_body = bodies.Body(Sun, G * 1e7 * units.kg, 'Chiron')
        self.bodies.append(chiron_body)
        self.locations_dict[chiron_body] = chiron_orbit

        self.loc_val_dict = {star : self.locations_dict[star].state.to_vectors().r.to('AU').value for star in self.bodies}
        self.orbit_dict = {
            star :
                [
                    self.locations_dict[star].
                        propagate(self.locations_dict[star].period / self.n_point_per_orbit * i).
                        state.to_vectors().r.to('AU').value
                    for i in range(1, self.n_point_per_orbit + 2)
                ]
            for star in self.bodies
        }
        self.coord_time = the_time
        print(the_time)

'''
根据所提供的日期时间，给出行星天体在国际天球参考系中位置，进一步通过坐标变换，得到行星天体在日心黄道坐标系中的位置
'''
def get_xyz_eclip_planet(planet=Earth, the_time=datetime.datetime(2018, 6, 22, 15, 19, 29)):
    the_system = SolarSystem()
    # the_time = datetime.datetime(2018, 6, 22, 15, 19, 29) #'''提供要求行星位置的日期时间'''
    the_system.calculate_locations(the_time)
    print(the_system)
    xyz_icrs_Earth = the_system.loc_val_dict[planet] #'''得到行星天体如地球在国际天球参考系icrs中的位置'''
    print(xyz_icrs_Earth)
    icrs_to_ecliptic = MilkyWay()
    icrs_to_ecliptic.icrs_to_ecli()
    icrs_to_ecliptic.calculate_coe()
    xyz_eclip_Earth = icrs_to_ecliptic.vector_cal(xyz_icrs_Earth[0], xyz_icrs_Earth[1], xyz_icrs_Earth[2]) #'''通过坐标变换得到行星天体在日心黄道坐标系中的位置'''
    print('xyz_eclip_Earth in get_xyz_eclip_planet: ', xyz_eclip_Earth)
    return xyz_eclip_Earth


'''
定义函数coo_transform，将轨道坐标系的坐标转换到黄道坐标系下的坐标
定义函数dist_psp_comet，计算指定日期时间下，PSP和comet的距离
定义函数single_beta_tail_particles，回溯number_of_days天，返回一个彗尾尘埃的位置列表
'''
file_dir = '/Users/duo/Desktop/My_Files/太阳风小小组/太阳系小天体/'
f = open(file_dir+'cometels_e_less_than_1.json')
ele_data = json.load(f)
f.close()

f2 = open(file_dir+'modified_psp_ephemeris_data_2019_0601_1201_V2.json')
psp_time_pos = json.load(f2)
f2.close()

f3 = open(file_dir+"PSP_comets_very_close_list.json")
psp_comets_very_close_list = json.load(f3)
f3.close()

def date_to_jd(year, month, day):
    # 将年,月,日转换成儒略日
    # return jd 儒略日
    # jd = gcal2jd(year, month, day)[0] + gcal2jd(year, month, day)[1]
    jd = gcal2jd(year, month, day)[0] + gcal2jd(year, month, day)[1] + day%1
    return jd


def jd_to_date(jdday):
    jd2gcal(2400000.5, jdday - 2400000.5)
    return jd2gcal(2400000.5, jdday - 2400000.5)


def cal_bc_from_ae(a, ecc):
    # 从半长轴和偏心率计算bc
    b = a * math.sqrt(1 - ecc**2)
    c = a * ecc
    return b, c


def cal_xy_from_theta(a, ecc, theta):  # theta为真近点角，计算theta对应的轨道坐标系坐标xy（以焦点为原点）
    theta = theta*from_deg2rad
    b = cal_bc_from_ae(a, ecc)[0]
    c = cal_bc_from_ae(a, ecc)[1]
    denominator = (a**2)*((sin(theta))**2) + (b**2)*((cos(theta))**2)
    x_numerator = -(b**2)*c*((cos(theta))**2) + a*(b**2)*(cos(theta))
    y_numerator = -(b**2)*c*(sin(theta))*(cos(theta)) + a*(b**2)*(sin(theta))
    x = x_numerator/denominator
    y = y_numerator/denominator
    return x, y


def get_orbit_xy(a, ecc, n):
    # n是取角度的数量  得到轨道坐标系中的x y 坐标（以焦点为原点）
    # 把轨道离散化，返回每隔2π/n 的轨道坐标
    b = cal_bc_from_ae(a, ecc)[0]
    c = cal_bc_from_ae(a, ecc)[1]
    x = []
    y = []
    for i in range(n):
        theta = i * (2 * math.pi/n)
        x0 = a*cos(theta)-c
        y0 = b*sin(theta)
        x.append(x0)
        y.append(y0)
    return x, y


def coo_transform(node, i, peri, x, y, z):
    """
    将单个坐标从轨道坐标系转化到黄道坐标系
    :param node:
    :param i:
    :param peri: 以上三项为轨道参数
    :param x:
    :param y:
    :param z: xyz为要转化的坐标，x y z 都为数值（轨道坐标系）
    :return: 转化后的坐标（黄道坐标系）
    """
    x1 = cos(node) * cos(peri) - sin(node) * cos(i) * sin(peri)
    x2 = sin(node) * cos(peri) + cos(node) * cos(i) * sin(peri)
    x3 = sin(i) * sin(peri)
    y1 = - cos(node) * sin(peri) - sin(node) * cos(i) * cos(peri)
    y2 = - sin(node) * sin(peri) + cos(node) * cos(i) * cos(peri)
    y3 = sin(i) * cos(peri)
    z1 = sin(i) * sin(node)
    z2 = - sin(i) * cos(node)
    z3 = cos(i)
    mat_trans = mat([[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]])
    mat_trans_t = mat_trans.I
    coo_ori = mat([[x], [y], [z]])
    coo_transformed = mat_trans_t * coo_ori
    x_trans = coo_transformed[0, 0]
    y_trans = coo_transformed[1, 0]
    z_trans = coo_transformed[2, 0]
    return x_trans, y_trans, z_trans


def frame_trans(node, i, peri, x, y):
    """
    :param node:
    :param i:
    :param peri: 以上三项为某一星体的轨道参数
    :param x:
    :param y: xy为在该星体椭圆轨道上取的点的坐标数组（轨道坐标系）
    :return: 转化到黄道坐标系后的坐标数组
    """
    x_transformed = []
    y_transformed = []
    z_transformed = []
    dot_amount = len(x)
    for j in range(dot_amount):
        [x0, y0, z0] = coo_transform(node, i, peri, x[j], y[j], 0)
        x_transformed.append(x0)
        y_transformed.append(y0)
        z_transformed.append(z0)
    return x_transformed, y_transformed, z_transformed   # 黄道坐标系的坐标数组


def newtons_method(M, e):
    # M 平近点角
    # e 轨道离心率
    #利用牛顿逼近方法得到偏近点角E
    #return E 偏近点角
    # error = 1e-15
    error = 1e-15
    # E = 1.1
    E = 2.0
    delta = abs(0 - (E - e*math.sin(E)-M))
    while delta > error:
        E = E - (E - e*math.sin(E)-M)/(1 - e*math.cos(E))
        delta = abs(0-(E - e*math.sin(E)-M))
    return E


def M2T(M, e):
    """
    :param M: 平近点角
    :param e: 轨道离心率
    :return: True_anomaly 真近点角
    """
    Eccentric_anomaly = newtons_method(M, e)
    interval = (math.cos(Eccentric_anomaly)-e)/(1-e*math.cos(Eccentric_anomaly))
    if Eccentric_anomaly < math.pi:
        True_anomaly = math.acos(interval)
    else:
        True_anomaly = 2*math.pi-math.acos(interval)
    return True_anomaly


def get_pos_of_the_obj(sn_pku, jdday=1.0, date_tuple=(0,0,0)):
    """
    :param sn_pku:
    :param jdday:
    :param date_tuple: date in the future(tuple)(year, month, day)
    :return: position of the star(x, y, z, r, theta, phi)
    """
    if jdday != 1.0:
        jd_day = jdday
    else:
        year, month, day = date_tuple
        jd_day = date_to_jd(year, month, day)
    n = sn_pku - 1
    a = ele_data[n]['a']
    ecc = ele_data[n]['e']
    node = ele_data[n]['Node'] * from_deg2rad
    peri = ele_data[n]['Peri'] * from_deg2rad
    i = ele_data[n]['i'] * from_deg2rad
    daily_motion = ele_data[n]['n']
    # days_from_epoch = date_to_jd(year, month, day) - ele_data[n]['Epoch']
    days_from_perihelion = jd_day - ele_data[n]["jd_day_of_perihelion"]
    # total_motion = daily_motion * days_from_epoch
    total_motion = daily_motion * days_from_perihelion
    """mean_ano = (total_motion + ele_data[n]['M']) % 360  # 给定时间特定星体的平近点角（角度制）"""
    mean_ano = total_motion % 360  # 给定时间特定星体的平近点角（角度制）
    true_ano = M2T(mean_ano * from_deg2rad, ecc) / from_deg2rad  # 真近点角（角度制）
    [ori_x, ori_y] = cal_xy_from_theta(a, ecc, true_ano)  # 利用真近点角计算轨道坐标系坐标
    [trans_x, trans_y, trans_z] = coo_transform(node, i, peri, ori_x, ori_y, 0)  # 转化到黄道坐标系后的坐标
    if trans_x > 0 and trans_y > 0:
        phi = math.atan(trans_y / trans_x)
    elif trans_x < 0 < trans_y:
        phi = math.atan(trans_y / trans_x) + math.pi
    elif trans_x < 0 and trans_y < 0:
        phi = math.atan(trans_y / trans_x) + math.pi
    else:
        phi = math.atan(trans_y / trans_x) + 2 * math.pi
    phi = phi / from_deg2rad
    r = math.sqrt(trans_x**2 + trans_y**2 + trans_z**2)
    theta = math.acos(trans_z / r) / from_deg2rad
    latitude = 90 - theta
    longitude = phi
    return (trans_x, trans_y, trans_z, r, latitude, longitude), true_ano


def dist_psp_comet(comet_num, jdday=1.0, date_tuple=(0, 0, 0)):
    """
    :param date_tuple: 指定一个日期（元组）
    :param comet_num: 彗星的编号（SN_PKU_local）
    :return: 在指定的日期，PSP和指定彗星的距离
    """

    if jdday != 1.0:
        jd_day = jdday
        jd_day_for_psp = jd_day
        comet_coordinates = get_pos_of_the_obj(comet_num, jdday=jd_day)[0]
        # print("comet", comet_coordinates)
        comet_x, comet_y, comet_z = comet_coordinates[:3]
        # print(comet_coordinates)
    else:
        year, month, day = date_tuple
        jd_day = date_to_jd(year, month, day)
        jd_day_for_psp = jd_day - day%1
        comet_coordinates = get_pos_of_the_obj(comet_num, jdday=jd_day)[0]
        # print("comet", comet_coordinates)
        comet_x, comet_y, comet_z = comet_coordinates[:3]

    psp_xyz = psp_time_pos[str(jd_day_for_psp)][1:4]
    # print("PSP", psp_xyz)
    psp_x, psp_y, psp_z = psp_xyz
    distance = math.sqrt((comet_x - psp_x) ** 2 + (comet_y - psp_y) ** 2 + (comet_z - psp_z) ** 2)
    # print("distance between this comet and PSP: ", distance)
    return distance, (comet_x, comet_y, comet_z), (psp_x, psp_y, psp_z)


def single_beta_tail_particles(comet_sn_pku_local, number_of_days, number_of_time_blocks, beta, date_tuple=(0,0,0), time_tuple=(0,0),
                               use_jdday=False, jdday=0, use_timesn=False, timesn=-1):
    """
    从date_tuple或jdday开始，回溯number_of_days天，返回一个彗尾尘埃的位置列表
    :param comet_sn_pku_local:
    :param date_tuple:
    :param time_tuple:
    :param use_jdday:
    :param jdday:
    :param number_of_days:
    :param number_of_time_blocks:
    :param beta:
    :return:
    """
    length_of_time_block = number_of_days / number_of_time_blocks
    if use_timesn:
        jd_day = psp_time_pos[int(timesn)]["jd_day"]
    elif use_jdday:
        jd_day = jdday
    else:
        year, month, day = date_tuple
        time_sn = (date_to_jd(year, month, day) - 2458635.5) * 144 + time_tuple[0] * 6 + time_tuple[1] / 10
        time_sn = int(time_sn)
        jd_day = psp_time_pos[time_sn]["jd_day"]  # jd_day是当前这一天的儒略日
    particles_present_pos_lst = []

    for i_day2 in range(number_of_time_blocks + 1):
        release_day = jd_day - number_of_days + i_day2 * length_of_time_block # release_day是释放当天的儒略日

        particle_release_r = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day)[0][:3] # 尘埃（彗星）在释放时刻的位置
        short_time = 0.01
        particle_release_r2 = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day + short_time)[0][:3]
        particle_release_r3 = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day - short_time)[0][:3]
        particle_release_v = [(particle_release_r2[i] - particle_release_r3[i]) / (2 * short_time) for i in range(3)] # 尘埃（彗星）在释放时刻的速度
        particle_release_r = [particle_release_r[i] for i in range(3)] * u.AU
        particle_release_v = [particle_release_v[i] for i in range(3)] * u.AU / u.day
        particle_orbit = Orbit.from_vectors(Sun, particle_release_r, particle_release_v, plane=Planes.EARTH_ECLIPTIC)

        particle_orbit.attractor.k = particle_orbit.attractor.k * (1 - beta)
        particle_present_pos = particle_orbit.propagate((number_of_days - i_day2 * length_of_time_block) * u.day).state.to_vectors().r.to('AU').value # 在release_day释放的粒子的当前坐标xyz
        particles_present_pos_lst.append(particle_present_pos)
        particle_orbit.attractor.k = particle_orbit.attractor.k / (1 - beta)
    return particles_present_pos_lst


def x_axis2vector_angular_dist(the_vector):
    if the_vector[0] > 0 and the_vector[1] > 0:
        angular_dist = math.atan(the_vector[1] / the_vector[0])
    elif the_vector[0] < 0 < the_vector[1]:
        angular_dist = math.atan(the_vector[1] / the_vector[0]) + math.pi
    elif the_vector[0] < 0 and the_vector[1] < 0:
        angular_dist = math.atan(the_vector[1] / the_vector[0]) + math.pi
    elif the_vector[1] < 0 < the_vector[0]:
        angular_dist = math.atan(the_vector[1] / the_vector[0]) + math.pi*2
    elif the_vector[0] == 0 and the_vector[1] > 0:
        angular_dist = math.pi/2
    elif the_vector[0] == 0 and the_vector[1] < 0:
        angular_dist = math.pi*3/2
    elif the_vector[1] == 0 and the_vector[0] > 0:
        angular_dist = 0
    else:
        angular_dist = math.pi
    return angular_dist


def angular_dist_psp_particle(comet_sn_pku_local, number_of_days, number_of_time_blocks, beta, date_tuple=(0,0,0), time_tuple=(0,0),
                               use_jdday=False, jdday=0, use_timesn=False, timesn=-1):
    if use_timesn:
        particles_present_pos_lst = single_beta_tail_particles(comet_sn_pku_local, number_of_days,
                                                               number_of_time_blocks, beta, use_timesn=True, timesn=timesn)
        time_sn = timesn
    elif use_jdday:
        particles_present_pos_lst = single_beta_tail_particles(comet_sn_pku_local, number_of_days,
                                                               number_of_time_blocks, beta, use_jdday=True, jdday=jdday)
        time_sn = (jdday - 2458635.5) * 144
        time_sn = int(time_sn)
    else:
        particles_present_pos_lst = single_beta_tail_particles(comet_sn_pku_local, number_of_days,
                                                               number_of_time_blocks, beta, date_tuple=date_tuple, time_tuple=time_tuple)
        year, month, day = date_tuple
        time_sn = (date_to_jd(year, month, day) - 2458635.5) * 144 + time_tuple[0] * 6 + time_tuple[1] / 10
        time_sn = int(time_sn)
    x_psp, y_psp, z_psp = psp_time_pos[time_sn]["x"], psp_time_pos[time_sn]["y"], psp_time_pos[time_sn]["z"]
    psp2sun_vector = (-x_psp, -y_psp, -z_psp)
    angular_dist_lst = []
    for i_block in range(number_of_time_blocks + 1):
        block_of_back = number_of_time_blocks - i_block
        single_pos = particles_present_pos_lst[i_block]
        psp2particle_vector = (single_pos[0] - x_psp, single_pos[1] - y_psp, single_pos[2] - z_psp)
        ps2pa_angular_dist = x_axis2vector_angular_dist(psp2sun_vector) \
                             - x_axis2vector_angular_dist(psp2particle_vector) # 从psp2sun_vector顺时针转到psp2particle_vector转过的角度
        if ps2pa_angular_dist < 0:
            ps2pa_angular_dist = ps2pa_angular_dist + math.pi*2
        ps2pa_angular_dist = ps2pa_angular_dist / from_deg2rad
        # ps2pa_angular_dist = ps2pa_angular_dist - 13.5 # particle在WISPR视场中的radial角度
        ps2pa_lat_angular_dist = math.atan(psp2particle_vector[-1] /
                                           math.sqrt(psp2particle_vector[0]**2 + psp2particle_vector[1]**2)) / from_deg2rad
        angular_dist_lst.append({"block of back":block_of_back, "particle_in_WISPR_radial":ps2pa_angular_dist,
                                 "psrticle_in_WISPR_transverse":ps2pa_lat_angular_dist})
    return angular_dist_lst

"""
    obj = ele_data[comet_sn_pku_local - 1]
    new_tail_particle_pos_lst = []
    new_orbit_distance_lst = []
    a = obj['a'] * u.AU
    ecc = obj['e'] * u.one
    inc = obj['i'] * u.deg
    raan = obj['Node'] * u.deg
    argp = obj['Peri'] * u.deg
    nu = get_pos_of_the_obj(comet_sn_pku_local, date_tuple=date_tuple)[1] * u.deg # 得到真近点角
    obj_orbit = Orbit.from_classical(Sun, a, ecc, inc, raan, argp, nu, plane=Planes.EARTH_ECLIPTIC)

    # print(obj_orbit.state.to_vectors().r.to('AU').value)
    # print('get_pos')
    # print(get_pos_of_the_obj(comet_sn_pku_local, date_tuple=date_tuple))
    # obj_orbit_minus_31 = obj_orbit.propagate(-31 * u.day)
    # print(obj_orbit_minus_31.state.to_vectors().r.to('AU').value)
    # print('get_pos2')
    # print(get_pos_of_the_obj(comet_sn_pku_local, date_tuple=(2023,10,1)))

    obj_orbit.attractor.k = obj_orbit.attractor.k * (1 - beta)
    for i_day in range(number_of_days + 1):
        new_obj_orbit = obj_orbit.propagate((-i_day) * u.day)
        new_obj_r = new_obj_orbit.state.to_vectors().r.to('AU').value # 回溯i_day+1天时的彗尾尘埃的位置
        new_tail_particle_pos_lst.append(new_obj_r)
        new_obj_dist = math.sqrt(new_obj_r[0]**2 + new_obj_r[1]**2 + new_obj_r[2]**2)
        new_orbit_distance_lst.append(new_obj_dist) # 计算与太阳的距离
    print(obj_orbit.attractor.k)
    print(new_orbit_distance_lst)
    obj_orbit.attractor.k = obj_orbit.attractor.k / (1 - beta)
    return new_tail_particle_pos_lst
    """


def plot_comet_tail_particles_in_wispr(beta_lst, comet_sn_pku_local, number_of_days, number_of_time_blocks, date_tuple=(0,0,0), time_tuple=(0,0),
                               use_jdday=False, jdday=0, use_timesn=False, timesn=-1):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rect = plt.Rectangle((13.5, -29), 94.5, 58, fill=False)
    ax.add_patch(rect)
    size_flag = True
    i_beta = -1
    for beta in beta_lst:
        i_beta = i_beta + 1
        if use_timesn:
            single_beta_angular_dist_lst = angular_dist_psp_particle(comet_sn_pku_local, number_of_days, number_of_time_blocks, beta, use_timesn=True, timesn=timesn)
        elif use_jdday:
            single_beta_angular_dist_lst = angular_dist_psp_particle(comet_sn_pku_local, number_of_days,
                                                                     number_of_time_blocks, beta, use_jdday=True, jdday=jdday)
        else:
            single_beta_angular_dist_lst = angular_dist_psp_particle(comet_sn_pku_local, number_of_days,
                                                                     number_of_time_blocks, beta, date_tuple=date_tuple, time_tuple=time_tuple)
        radial_angle_tuple = tuple(item["particle_in_WISPR_radial"] for item in single_beta_angular_dist_lst)
        transverse_angle_tuple = tuple(item["psrticle_in_WISPR_transverse"] for item in single_beta_angular_dist_lst)
        if size_flag:
            plt.scatter(radial_angle_tuple[-1], transverse_angle_tuple[-1],
                        c=generate_hsv_color_array(beta_lst, number_of_time_blocks,
                                                   270. / 360., 0.0, 1.0, 0.2)[i_beta][-1], s=50, marker="*")
            size_flag = False
        else:
            plt.scatter(radial_angle_tuple, transverse_angle_tuple,
                        c=generate_hsv_color_array(beta_lst, number_of_time_blocks,
                                                   270. / 360., 0.0, 1.0, 0.2)[i_beta], s=0.5)
    plt.axis([0, 360, -90, 90])
    plt.legend(["FOV of WISPR", "comet present position", "beta = "+str(beta_lst[1]), "beta = "+str(beta_lst[2]), "beta = "+str(beta_lst[3])])
    plt.title("Comet Particles in WISPR "+str(date_tuple)+" "+str(time_tuple))
    plt.xlabel("radial angle / deg")
    plt.ylabel("transverse angle / deg")
    plt.grid()
    plt.show()
    plt.close()


def generate_hsv_color_array(beta_lst, number_of_time_blocks_comet, h_of_hsv_max, h_of_hsv_min, s_of_hsv_max, s_of_hsv_min):
    beta_vect = np.array(beta_lst)
    beta_min = min(beta_vect)
    beta_max = max(beta_vect)
    h_of_hsv_lst = h_of_hsv_min + (beta_vect - beta_min) / (beta_max - beta_min) * (h_of_hsv_max - h_of_hsv_min)
    day_vect_comet = np.arange(number_of_time_blocks_comet + 1)
    s_of_hsv_lst = s_of_hsv_max - (day_vect_comet) / number_of_time_blocks_comet * (s_of_hsv_max - s_of_hsv_min)
    s_of_hsv_lst = s_of_hsv_lst[::-1]
    v_of_hsv_lst = beta_vect * 0.0 + 1.0

    color_array = np.zeros([len(beta_lst), number_of_time_blocks_comet+1], dtype=object)
    i_beta = -1
    for beta in beta_lst:
        i_beta = i_beta + 1
        for i_block in range(number_of_time_blocks_comet + 1):
            h_of_hsv_tmp = h_of_hsv_lst[i_beta]
            s_of_hsv_tmp = s_of_hsv_lst[i_block]
            v_of_hsv_tmp = v_of_hsv_lst[i_beta]
            color_hsv = [h_of_hsv_tmp, s_of_hsv_tmp, v_of_hsv_tmp]
            color_rgb = colorsys.hsv_to_rgb(color_hsv[0], color_hsv[1], color_hsv[2])
            color_array[i_beta][i_block] = color_rgb
    return color_array


def plot_comet_tail(comet_sn_pku_local, number_of_days_comet, number_of_time_blocks_comet, number_of_days_psp, beta_lst, date_tuple=(0,0,0), use_jdday=False, jdday=0):
    """

    :param comet_sn_pku_local: 彗星的编号
    :param date_tuple: 日期
    :param use_jdday:
    :param jdday:
    :param number_of_days_comet: 彗星从date_tuple开始回溯的天数
    :param number_of_time_blocks_comet:
    :param number_of_days_psp: psp从date_tuple开始回溯的天数
    :param beta_lst: beta的列表
    :return:
    """
    if use_jdday:
        jd_day = jdday
        jd_day_for_psp = jd_day
    else:
        jd_day = date_to_jd(date_tuple[0], date_tuple[1], date_tuple[2])  # jd_day是当前这一天的儒略日
        jd_day_for_psp = jd_day - date_tuple[2]%1

    mlab.figure(bgcolor=(0.,0.,0.),size=(1000, 800))
    dot_size = 0.01

    mlab.view(azimuth=45, elevation=45, distance=10., focalpoint=[0.,0.,0.])

    '''画太阳'''
    mlab.points3d(0, 0, 0, scale_factor=0.03, color=(1, 1, 0)) # 太阳

    """加画从太阳指向地球的箭头"""
    date_tuple_tmp = jd_to_date(jd_day)
    xyz_eclip_Earth = get_xyz_eclip_planet(planet=Earth,
                                           the_time=datetime.datetime(date_tuple_tmp[0], date_tuple_tmp[1], date_tuple_tmp[2], 15, 19, 29))
    u = numpy.array([[xyz_eclip_Earth[0]]])
    v = numpy.array([[xyz_eclip_Earth[1]]])
    w = numpy.array([[xyz_eclip_Earth[2]]])
    x, y, z = numpy.zeros_like(u), numpy.zeros_like(u), numpy.zeros_like(u)
    mlab.quiver3d(x, y, z, u, v, w, line_width=0.3, scale_factor=0.5, mode='arrow', color=(138 / 255, 43 / 255, 226 / 255))
    mlab.points3d(xyz_eclip_Earth[0], xyz_eclip_Earth[1], xyz_eclip_Earth[2], scale_factor=0.03, color=(0.1, 0.1, 1.0))  # 地球

    color_i = 0
    color_lst = [(192/255, 192/255, 192/255), (128/255, 118/255, 105/255), (128/255, 42/255, 42/255)]

    # 何建森添加 (2019-01-11)
    # 根据beta_lst数组 创建 h_of_hsv_lst，使得小beta具有红色值，大beta值具有紫色值
    # 根据number_of_days_comet 创建 s_of_hsv_lst，使得回溯的天数越大s越小，回溯的天数越小s越大
    beta_vect = np.array(beta_lst)
    beta_min = min(beta_vect)
    beta_max = max(beta_vect)
    h_of_hsv_min = 0.0
    h_of_hsv_max = 270./360.
    h_of_hsv_lst = h_of_hsv_min + (beta_vect-beta_min)/(beta_max-beta_min)*(h_of_hsv_max-h_of_hsv_min)
    s_of_hsv_min = 0.2
    s_of_hsv_max = 1.0
    day_vect_comet = np.arange(number_of_time_blocks_comet+1)
    s_of_hsv_lst = s_of_hsv_max - (day_vect_comet)/number_of_time_blocks_comet*(s_of_hsv_max-s_of_hsv_min)
    s_of_hsv_lst = s_of_hsv_lst[::-1]
    v_of_hsv_lst = beta_vect*0.0+1.0


    i_beta = -1
    for beta in beta_lst:
        i_beta = i_beta+1
        single_beta_particles_pos_lst = single_beta_tail_particles(comet_sn_pku_local, number_of_days_comet, number_of_time_blocks_comet, beta, use_jdday=True, jdday=jd_day)
        # print(single_beta_particles_pos_lst)
        i_day_comet = -1
        for single_particle_pos in single_beta_particles_pos_lst:
            i_day_comet = i_day_comet + 1
            h_of_hsv_tmp = h_of_hsv_lst[i_beta]
            if i_day_comet == 80:
                print(single_beta_particles_pos_lst)

            s_of_hsv_tmp = s_of_hsv_lst[i_day_comet]
            v_of_hsv_tmp = v_of_hsv_lst[i_beta]
            color_hsv = [h_of_hsv_tmp, s_of_hsv_tmp, v_of_hsv_tmp]
            color_rgb = colorsys.hsv_to_rgb(color_hsv[0],color_hsv[1],color_hsv[2])
            # print(color_rgb)
            # mlab.points3d(single_particle_pos[0], single_particle_pos[1], single_particle_pos[2], scale_factor=dot_size, color=color_lst[color_i])
            mlab.points3d(single_particle_pos[0], single_particle_pos[1], single_particle_pos[2], scale_factor=dot_size, color=color_rgb)
        color_i = color_i + 1

    for i_day in range(number_of_days_psp):
        release_day = jd_day_for_psp - number_of_days_psp + i_day
        psp_xyz = psp_time_pos[str(release_day)][1:4]
        mlab.points3d(psp_xyz[0], psp_xyz[1], psp_xyz[2], scale_factor=dot_size)
    mlab.points3d(psp_time_pos[str(jd_day_for_psp)][1], psp_time_pos[str(jd_day_for_psp)][2], psp_time_pos[str(jd_day_for_psp)][3], scale_factor=3 * dot_size)

    for i_day_3 in range(number_of_days_psp + 1):
        release_day = jd_day - number_of_days_psp + i_day_3
        comet_xyz = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day)[0][:3]
        mlab.points3d(comet_xyz[0], comet_xyz[1], comet_xyz[2], scale_factor=1*dot_size, color=(1, 0, 0))
        if i_day_3 == number_of_days_psp:
            mlab.points3d(comet_xyz[0], comet_xyz[1], comet_xyz[2], scale_factor=3 * dot_size, color=(1, 0, 0))

    """保存某一天的comet的位置、若干天之内释放的tail particle在该天位置、以及该天的PSP的位置"""
    """涉及到坐标系的转换
    目标：从日心黄道坐标系转换到日心卡林顿坐标系
    SunPy能提供HEEQ、日心卡林顿坐标系、Heliocentric
    要从日心黄道坐标系转换到HEEQ：
    要知道HEEQ的x/y/z轴在日心黄道坐标系中的矢量表达
    日心黄道坐标系有时候也叫HAE(Heliocentric Aris Ecliptic coordinate)坐标系
    """

    """
    epoch = time.Time(str(date_tuple[0]) + "-" + str(date_tuple[1]) + "-" + str(date_tuple[2]) + " 00:00")
    earth_orbit = Orbit.from_body_ephem(Earth, epoch)
    print(earth_orbit)
    earth_r = earth_orbit.state.to_vectors().r.to('AU').value # icrs坐标系
    print(earth_r)
    earth_icrs_coord = SkyCoord(earth_r[0], earth_r[1], earth_r[2], unit='AU', frame = "icrs", representation='cartesian')
    earth_r_ecliptic = earth_icrs_coord.transform_to(coordinates.HeliocentricTrueEcliptic())
    print(earth_r_ecliptic)
    """

    date_tmp = jd_to_date(jd_day)
    date_tmp_str = str(date_tmp)
    mlab.savefig(file_dir + "plot_comet_tail(6, 80, 80, [0.2, 0.4, 0.6], date_tuple="+date_tmp_str+").png")
    mlab.show()
    # mlab.savefig(file_dir+'fig.png')
    # mlab.savefig(file_dir+"plot_comet_tail(6, 80, 80, [0.2, 0.4, 0.6], date_tuple=(2019, 9, 3)).png")

"""崔博添加函数dist_psp_particle
用于计算彗星释放的不同尘埃与psp相距最近的距离，相距最近的回溯天数，以及beta值
增加了参数number_of_time_blocks_comet（时间块数，对总回溯天数所分的段数）"""
def dist_psp_particle(comet_sn_pku_local, number_of_days_comet, number_of_time_blocks_comet, beta_lst, date_tuple=(0,0,0), time_tuple=(0,0)):
    """
    :param comet_sn_pku_local:
    :param number_of_days_comet:
    :param number_of_time_blocks_comet: 时间块数，对总回溯天数所分的段数
    :param beta_lst:
    :param date_tuple:
    :param time_tuple: 元组(hour, minute)
    :param use_jdday:
    :param jdday:
    :return: 在某一天(date_tuple)与psp相距最近的尘埃的释放时间（回溯的天数），该尘埃的beta值，相距最近的距离，以及二维数组psp_particle_dist_arr
    """
    length_of_time_block = number_of_days_comet / number_of_time_blocks_comet

    year, month, day = date_tuple
    time_sn = (date_to_jd(year, month, day) - 2458635.5) * 144 + time_tuple[0] * 6 + time_tuple[1] / 10
    time_sn = int(time_sn)
    jd_day = psp_time_pos[time_sn]["jd_day"]

    psp_particle_dist_arr = np.zeros((len(beta_lst), number_of_time_blocks_comet+1))
    # psp_particle_dist_arr是一个二维数组，行从上到下为不同的beta，列从左到右为回溯时间块数由小到大对应的尘埃和psp的距离
    i_beta = 0
    for beta in beta_lst:
        particles_present_pos = single_beta_tail_particles(comet_sn_pku_local, number_of_days_comet, number_of_time_blocks_comet, beta, use_jdday=True, jdday=jd_day) # list
        i_day = 0
        for single_pos in particles_present_pos:
            dist = math.sqrt((psp_time_pos[time_sn]['x'] - single_pos[0])**2 + (psp_time_pos[time_sn]['y'] - single_pos[1])**2 +
                             (psp_time_pos[time_sn]['z'] - single_pos[2])**2)
            psp_particle_dist_arr[i_beta][-i_day - 1] = dist
            i_day = i_day + 1
        i_beta = i_beta + 1
    min_dist_psp_particle = np.min(psp_particle_dist_arr)
    min_dist_iblock = np.where(psp_particle_dist_arr == np.min(psp_particle_dist_arr))[1][0]
    min_dist_ibeta = np.where(psp_particle_dist_arr == np.min(psp_particle_dist_arr))[0][0]

    # min_dist_psp_r = (psp_time_pos[time_sn]['x'], psp_time_pos[time_sn]['y'], psp_time_pos[time_sn]['z'])
    min_dist_psp_v = (psp_time_pos[time_sn]['vx'] * 1736.111111111111, psp_time_pos[time_sn]['vy'] * 1736.111111111111, psp_time_pos[time_sn]['vz'] * 1736.111111111111)

    release_day = jd_day - min_dist_iblock * number_of_days_comet / number_of_time_blocks_comet  # release_day是释放当天的儒略日

    particle_release_r = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day)[0][:3]  # 尘埃（彗星）在释放时刻的位置
    short_time = 0.01
    particle_release_r2 = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day + short_time)[0][:3]
    particle_release_r3 = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day - short_time)[0][:3]
    particle_release_v = [(particle_release_r2[i] - particle_release_r3[i]) / (2 * short_time) for i in
                          range(3)]  # 尘埃（彗星）在释放时刻的速度
    particle_release_r = [particle_release_r[i] for i in range(3)] * u.AU
    particle_release_v = [particle_release_v[i] for i in range(3)] * u.AU / u.day
    particle_orbit = Orbit.from_vectors(Sun, particle_release_r, particle_release_v, plane=Planes.EARTH_ECLIPTIC)

    particle_orbit.attractor.k = particle_orbit.attractor.k * (1 - beta_lst[min_dist_ibeta])
    particle_present_pos = particle_orbit.propagate(
        (min_dist_iblock * number_of_days_comet / number_of_time_blocks_comet) * u.day).state.to_vectors().v.to(
        'km/s').value  # 在release_day释放的粒子的当前坐标xyz
    particle_orbit.attractor.k = particle_orbit.attractor.k / (1 - beta_lst[min_dist_ibeta])

    min_dist_particle_v = particle_present_pos
    relative_v_psp_particle = min_dist_particle_v - min_dist_psp_v
    return min_dist_iblock, min_dist_ibeta, min_dist_psp_particle, relative_v_psp_particle, min_dist_psp_v, min_dist_particle_v, psp_particle_dist_arr


def plot_dist_psp_particle(beta_lst, total_lookback_in_day, num_of_lookback, beginning_time, end_time, time_step_in_min, draw_angle=False):
    """

    :param beta_lst: 假设的beta的列表，对其中的每个beta画一幅图
    :param total_lookback_in_day: 对每一个当前时间的总回溯天数
    :param num_of_lookback: 对每一个当前时间，回溯的时间块数（尘埃粒子数）
    :param beginning_time: 当前时间的开始时间，如 "2019-08-25 00:00:00"
    :param end_time: 当前时间的结束时间，如 "2019-09-10 00:00:00"
    :param time_step_in_min: 当前时间的时间步长，以分钟为单位（要求为10min的整数倍）
    :return:
    """
    beginning_year = int(beginning_time[:4])
    beginning_month = int(beginning_time[5:7])
    beginning_day = int(beginning_time[8:10])
    beginning_hour = int(beginning_time[11:13])
    beginning_min = int(beginning_time[14:16])
    beginning_time_sn = (date_to_jd(beginning_year, beginning_month, beginning_day) - 2458635.5) * 144 + beginning_hour * 6 + beginning_min / 10
    end_year = int(end_time[:4])
    end_month = int(end_time[5:7])
    end_day = int(end_time[8:10])
    end_hour = int(end_time[11:13])
    end_min = int(end_time[14:16])
    end_time_sn = (date_to_jd(end_year, end_month,
                                    end_day) - 2458635.5) * 144 + end_hour * 6 + end_min / 10
    num_of_time_node = int((end_time_sn - beginning_time_sn) // (time_step_in_min / 10))

    for beta in beta_lst:
        coord_x_for_plot = []
        coord_y_for_plot = []
        total_dist_lst_for_plot = []
        total_dist_arr_for_plot = np.array(total_dist_lst_for_plot)
        for i_node in range(num_of_time_node):
            present_time_sn = int(beginning_time_sn + i_node * time_step_in_min / 10)
            present_node_back_lst = single_beta_tail_particles(6, total_lookback_in_day, num_of_lookback, beta, use_timesn=True, timesn=present_time_sn)
            present_dist_lst = [math.sqrt((a[0] - psp_time_pos[present_time_sn]['x'])**2 + (a[1] - psp_time_pos[present_time_sn]['y'])**2 +
                                          (a[2] - psp_time_pos[present_time_sn]['z'])**2) for a in present_node_back_lst]
            total_dist_lst_for_plot = total_dist_lst_for_plot + present_dist_lst
            total_dist_arr_for_plot = np.array(total_dist_lst_for_plot)
            coord_x_for_plot = coord_x_for_plot + [present_time_sn] * (num_of_lookback + 1)
            coord_y_for_plot = coord_y_for_plot + [total_lookback_in_day - i*total_lookback_in_day/num_of_lookback for i in range(num_of_lookback + 1)]
        c_m = plt.cm.get_cmap('RdYlBu')
        # sc = plt.scatter(coord_x_for_plot, coord_y_for_plot, c=np.log10(total_dist_arr_for_plot), vmin=0, vmax=0.65, s=35, cmap=cm, marker="s")
        sc = plt.scatter(coord_x_for_plot, coord_y_for_plot, c=np.log10(total_dist_arr_for_plot), vmin=-3,
                         s=35, cmap=c_m, marker="s")
        plt.colorbar(sc)
        x_tick_pos_lst = [coord_x_for_plot[i] for i in range(0, num_of_time_node*(num_of_lookback + 1), num_of_time_node//3*(num_of_lookback + 1))]
        x_tick_label_lst = [psp_time_pos[a]["date"] + " " + psp_time_pos[a]["time"][:-8] for a in x_tick_pos_lst]
        plt.xticks(x_tick_pos_lst, x_tick_label_lst)
        plt.xlabel("present day")
        plt.ylabel("days of look back")
        plt.title("distance between PSP and comet tail particles in different time and days of look back (beta = "  + str(beta) + ")" + "\nfrom "
                  + beginning_time + " to " +  end_time)
        plt.show()
        plt.close()


if __name__ == '__main__':
    # plot_dist_psp_particle([0, 0.2, 0.4, 0.6, 0.8], 5, 60, "2019-09-02 00:00:00", "2019-09-03 00:00:00", 10)

    # print(single_comet_tail_particle(775, (2020, 6, 25), 30, 0))

    # plot_comet_tail(771, (2020, 6, 25), 80, 80, [0, 0.2])
    # print(get_pos_of_the_obj(771, date_tuple=(2020, 6, 25)))

    # plot_comet_tail(845, (2019, 9, 3), 80, 80, [0.2, 0.3, 0.5])
    # print(get_pos_of_the_obj(845, date_tuple=(2019, 9, 3)))

    # plot_comet_tail(79, (2023, 11, 1), 80, 80, [0.2, 0.3, 0.5])
    # print(get_pos_of_the_obj(79, date_tuple=(2023, 11, 1)))

    # print(single_beta_tail_particles(6, 80, 320, 0.2, (2019, 9, 2), time_tuple=(17, 40)))

    # plot_dist_psp_particle([0.2], 5, 60, "2019-09-02 00:00:00", "2019-09-03 00:00:00", 10)

    # print(dist_psp_particle(6, 80, 320, [0.2, 0.4, 0.6], date_tuple=(2019, 9, 2), time_tuple=(8, 20)))

    """print(len(angular_dist_psp_particle(6, 30, 120, 0.2, (2019, 9, 2), (8, 20))))
    print(angular_dist_psp_particle(6, 30, 120, 0.2, (2019, 9, 2), (8, 20)))"""

    """color_array = generate_hsv_color_array([0.2, 0.4, 0.6], 30, 270./360., 0.0, 1.0, 0.2)
    print(color_array)"""

    plot_comet_tail_particles_in_wispr([0,0.2,0.4,0.6], 6, 30, 4320, (2019, 9, 2), (7, 50))

    """for i_hour in range(7, 13):
        for i_minute in range(0, 60, 10):
            print("2019.09.02--" + str(i_hour) + ":" + str(i_minute))
            print(dist_psp_particle(6, 80, 320, [0.2, 0.4, 0.6], date_tuple=(2019, 9, 2), time_tuple=(i_hour, i_minute))[:3])
            print("\n")"""


    # print(date_to_jd(2018, 8, 13))
    # print(date_to_jd(2019, 9, 3.1))

    """date_tuple_closest = (2019,9,3)
    jd_day_closest = date_to_jd(date_tuple_closest[0], date_tuple_closest[1], date_tuple_closest[2])
    jd_day_plot = jd_day_closest + 0.0
    date_plot = jd_to_date(jd_day_plot)
    print('date_plot: ',date_plot)
    plot_comet_tail(6, 80, 80, [0.2, 0.4, 0.6], use_jdday=True, jdday=jd_day_plot)"""


    # print(get_pos_of_the_obj(6, date_tuple=(2019, 9, 3)))

    # plot_comet_tail(845, 60, 60, [0.2, 0.4, 0.6], use_jdday=True, jdday=2458729.5 + 100)

    """for obj in psp_comets_very_close_list:
        plot_comet_tail(obj["comet_SN_PKU_local"], 60, 60, [0.2, 0.4, 0.6], use_jdday=True, jdday=obj["closest_jdday"] - 40)
        plot_comet_tail(obj["comet_SN_PKU_local"], 60, 60, [0.2, 0.4, 0.6], use_jdday=True,
                        jdday=obj["closest_jdday"] + 30)
        plot_comet_tail(obj["comet_SN_PKU_local"], 60, 60, [0.2, 0.4, 0.6], use_jdday=True,
                        jdday=obj["closest_jdday"] + 100)"""

    # print(date_to_jd(2020,1,2))
    # print(jd_to_date(2458849.6))
    """close_lst = [] # 元组的列表，存储与PSP足够近的天体的编号，距离小于1AU时的时间，当天的间距，准备存储更多信息
    for obj in ele_data:
        dist1 = 1
        the_day = 0
        return_tuple = (0, 0, 0)
        if obj["Perihelion_dist"] < 1:
            flag = True
            for day_number in range(2458343, 2460918, 1):
                jdday = day_number + 0.5
                return_tuple_2 = dist_psp_comet(obj["SN_PKU_local"], jdday=jdday)
                dist2 = return_tuple_2[0]
                if dist2 <= dist1:
                    flag = False
                    dist1 = dist2
                    the_day = jdday
                    return_tuple = return_tuple_2
            if not flag:
                comet_close_dict = {}
                comet_close_dict["comet_SN_PKU_local"] = obj["SN_PKU_local"]
                comet_close_dict["Designation_and_name"] = obj["Designation_and_name"]
                comet_close_dict["closest_jdday"] = the_day
                comet_close_dict["closest_date"] = (jd_to_date(the_day)[0], jd_to_date(the_day)[1], jd_to_date(the_day)[2])
                comet_close_dict["closest_distance"] = dist1
                comet_close_dict["comet_position"] = return_tuple[1]
                comet_sun_distance = math.sqrt(return_tuple[1][0]**2 + return_tuple[1][1]**2 + return_tuple[1][2]**2)
                comet_close_dict["comet_sun_distance_at_the_day"] = comet_sun_distance
                comet_close_dict["PSP_position"] = return_tuple[2]
                psp_sun_distance = math.sqrt(return_tuple[2][0]**2 + return_tuple[2][1]**2 + return_tuple[2][2]**2)
                comet_close_dict["PSP_sun_distance_at_the_day"] = psp_sun_distance
                if comet_sun_distance < psp_sun_distance:
                    comet_close_dict["comet_closer_than_PSP"] = True
                else:
                    comet_close_dict["comet_closer_than_PSP"] = False
                close_lst.append(comet_close_dict)
                # close_lst.append((obj["SN_PKU_local"], the_day, dist1, return_tuple[1], return_tuple[2]))
                print(obj["SN_PKU_local"], the_day, dist1, return_tuple[1], return_tuple[2])
    print(close_lst)
    print(len(close_lst))"""


    

    """jStr = json.dumps(close_lst, indent=2)
    with open('PSP_comets_close_distance_list_V7_latest.json', 'w') as f:
        f.write(jStr)"""
    # 108 2460720.5 0.24044048975315005
    """flag = False
    for day_number in range(2458348, 2460918, 1):
        jdday = day_number + 0.5
        distance = dist_psp_comet(597, jdday=jdday)
        if distance < 1:
            print(jdday, distance)
            flag = True
            # break
    print(flag)"""

    """dist1 = 10
    jdday2 = 0
    for day_number in range(2458343, 2460918, 1):
        jdday = day_number + 0.5
        dist2 = dist_psp_comet(597, jdday=jdday)[0]
        print(jdday, dist2)
        if dist2 < dist1:
            dist1 = dist2
            jdday2 = jdday
            # print(jdday2, dist1)"""

    # print(get_pos_of_the_obj(6, 2459127.5))

    """i = 0
    for obj in ele_data:
        if obj["Perihelion_dist"] < 1:
            i = i + 1
            print(obj["SN_PKU_local"])"""


    # print(dist_psp_comet(597, 2459830.5))
    # print(dist_psp_comet(530, 2458709.5))
    # dist_psp_comet(356, date_tuple=(2020,1,1))
    # print(get_pos_of_the_obj(463, (2011, 1, 1)))


"""i = 0
for obj in ele_data:
    if obj["Perihelion_dist"] < 1:
        i = i + 1
print(i)"""
