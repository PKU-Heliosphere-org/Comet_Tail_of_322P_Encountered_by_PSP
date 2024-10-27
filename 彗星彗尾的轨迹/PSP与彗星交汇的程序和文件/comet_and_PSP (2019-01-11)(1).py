import math
import mayavi.mlab as mlab
import re
import json
import numpy
import colorsys
from poliastro.twobody import Orbit
from poliastro.frames import get_frame, Planes
from poliastro.bodies import Mercury, Venus, Earth, Mars, Sun
from poliastro import ephem

from astropy import units as u
from astropy import time
from astropy.coordinates import SkyCoord
import astropy.coordinates as coordinates

from pandas import Series
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from jdcal import gcal2jd, jd2gcal
from numpy import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter, attrgetter
from_deg2rad = math.pi/180  # 角度向弧度的转换

f = open('/E-Work/学生科研/侯传鹏 & 崔博/PSP&Comet/cometels_e_less_than_1.json')
ele_data = json.load(f)
f.close()

f2 = open('/E-Work/学生科研/侯传鹏 & 崔博/PSP&Comet/modified_PSP_ephemeris_data.json')
psp_time_pos = json.load(f2)
f2.close()

f3 = open("/E-Work/学生科研/侯传鹏 & 崔博/PSP&Comet/PSP_comets_very_close_list.json")
psp_comets_very_close_list = json.load(f3)
f3.close()

def date_to_jd(year, month, day):
    # 将年,月,日转换成儒略日
    # return jd 儒略日
    jd = gcal2jd(year, month, day)[0] + gcal2jd(year, month, day)[1]
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
        comet_coordinates = get_pos_of_the_obj(comet_num, jdday=jd_day)[0]
        # print("comet", comet_coordinates)
        comet_x, comet_y, comet_z = comet_coordinates[:3]
        # print(comet_coordinates)
    else:
        year, month, day = date_tuple
        jd_day = date_to_jd(year, month, day)
        comet_coordinates = get_pos_of_the_obj(comet_num, jdday=jd_day)[0]
        # print("comet", comet_coordinates)
        comet_x, comet_y, comet_z = comet_coordinates[:3]

    psp_xyz = psp_time_pos[str(jd_day)][1:4]
    # print("PSP", psp_xyz)
    psp_x, psp_y, psp_z = psp_xyz
    distance = math.sqrt((comet_x - psp_x) ** 2 + (comet_y - psp_y) ** 2 + (comet_z - psp_z) ** 2)
    # print("distance between this comet and PSP: ", distance)
    return distance, (comet_x, comet_y, comet_z), (psp_x, psp_y, psp_z)


def single_beta_tail_particles(comet_sn_pku_local, number_of_days, beta, date_tuple=(0,0,0), use_jdday=False, jdday=0):
    """
    从date_tuple或jdday开始，回溯number_of_days天，返回一个彗尾尘埃的位置列表
    :param comet_sn_pku_local:
    :param date_tuple:
    :param use_jdday:
    :param jdday:
    :param number_of_days:
    :param beta:
    :return:
    """
    if use_jdday:
        jd_day = jdday
    else:
        jd_day = date_to_jd(date_tuple[0], date_tuple[1], date_tuple[2])  # jd_day是当前这一天的儒略日
    particles_present_pos_lst = []

    for i_day2 in range(number_of_days + 1):
        release_day = jd_day - number_of_days + i_day2 # release_day是释放当天的儒略日

        particle_release_r = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day)[0][:3] # 尘埃（彗星）在释放时刻的位置
        short_time = 0.01
        particle_release_r2 = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day + short_time)[0][:3]
        particle_release_r3 = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day - short_time)[0][:3]
        particle_release_v = [(particle_release_r2[i] - particle_release_r3[i]) / (2 * short_time) for i in range(3)] # 尘埃（彗星）在释放时刻的速度
        particle_release_r = [particle_release_r[i] for i in range(3)] * u.AU
        particle_release_v = [particle_release_v[i] for i in range(3)] * u.AU / u.day
        particle_orbit = Orbit.from_vectors(Sun, particle_release_r, particle_release_v, plane=Planes.EARTH_ECLIPTIC)

        particle_orbit.attractor.k = particle_orbit.attractor.k * (1 - beta)
        particle_present_pos = particle_orbit.propagate((number_of_days - i_day2) * u.day).state.to_vectors().r.to('AU').value # 在release_day释放的粒子的当前坐标xyz
        particles_present_pos_lst.append(particle_present_pos)
        particle_orbit.attractor.k = particle_orbit.attractor.k / (1 - beta)
    return particles_present_pos_lst


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


def plot_comet_tail(comet_sn_pku_local, number_of_days_comet, number_of_days_psp, beta_lst, date_tuple=(0,0,0), use_jdday=False, jdday=0):
    """

    :param comet_sn_pku_local: 彗星的编号
    :param date_tuple: 日期
    :param use_jdday:
    :param jdday:
    :param number_of_days_comet: 彗星从date_tuple开始回溯的天数
    :param number_of_days_psp: psp从date_tuple开始回溯的天数
    :param beta_lst: beta的列表
    :return:
    """
    if use_jdday:
        jd_day = jdday
    else:
        jd_day = date_to_jd(date_tuple[0], date_tuple[1], date_tuple[2])  # jd_day是当前这一天的儒略日

    figure1 = mlab.figure()
    dot_size = 0.01

    mlab.points3d(0, 0, 0, scale_factor=0.03, color=(1, 1, 0)) # 太阳
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
    day_vect_comet = np.arange(number_of_days_comet+1)
    s_of_hsv_lst = s_of_hsv_max - (day_vect_comet)/number_of_days_comet*(s_of_hsv_max-s_of_hsv_min)
    v_of_hsv_lst = beta_vect*0.0+1.0

    i_beta = -1
    for beta in beta_lst:
        i_beta = i_beta+1
        single_beta_particles_pos_lst = single_beta_tail_particles(comet_sn_pku_local, number_of_days_comet, beta, use_jdday=True, jdday=jd_day)
        print(single_beta_particles_pos_lst)
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
            print(color_rgb)
            # mlab.points3d(single_particle_pos[0], single_particle_pos[1], single_particle_pos[2], scale_factor=dot_size, color=color_lst[color_i])
            mlab.points3d(single_particle_pos[0], single_particle_pos[1], single_particle_pos[2], scale_factor=dot_size, color=color_rgb)
        color_i = color_i + 1

    for i_day in range(number_of_days_psp):
        release_day = jd_day - number_of_days_psp + i_day
        psp_xyz = psp_time_pos[str(release_day)][1:4]
        mlab.points3d(psp_xyz[0], psp_xyz[1], psp_xyz[2], scale_factor=dot_size)
    mlab.points3d(psp_time_pos[str(jd_day)][1], psp_time_pos[str(jd_day)][2], psp_time_pos[str(jd_day)][3], scale_factor=3 * dot_size)

    for i_day_3 in range(number_of_days_psp + 1):
        release_day = jd_day - number_of_days_psp + i_day_3
        comet_xyz = get_pos_of_the_obj(comet_sn_pku_local, jdday=release_day)[0][:3]
        mlab.points3d(comet_xyz[0], comet_xyz[1], comet_xyz[2], scale_factor=dot_size, color=(1, 0, 0))

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

    mlab.show()
    # mlab.savefig('/Users/duo/Desktop/回溯30天的彗尾尘埃轨迹.pdf')


if __name__ == '__main__':
    # print(single_comet_tail_particle(775, (2020, 6, 25), 30, 0))

    # plot_comet_tail(771, (2020, 6, 25), 80, 80, [0, 0.2])
    # print(get_pos_of_the_obj(771, date_tuple=(2020, 6, 25)))

    # plot_comet_tail(845, (2019, 9, 3), 80, 80, [0.2, 0.3, 0.5])
    # print(get_pos_of_the_obj(845, date_tuple=(2019, 9, 3)))

    # plot_comet_tail(79, (2023, 11, 1), 80, 80, [0.2, 0.3, 0.5])
    # print(get_pos_of_the_obj(79, date_tuple=(2023, 11, 1)))

    plot_comet_tail(6, 80, 80, [0.2, 0.4, 0.6], date_tuple=(2019, 9, 3))
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
