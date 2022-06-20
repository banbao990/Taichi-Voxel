from scene import Scene
import taichi as ti
from taichi.math import *

# config
exposure = 2
floor = -40

scene = Scene(exposure=exposure)
scene.set_floor(floor/64.0, (1.0, 1.0, 1.0)) # position, color
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1.5, 1.5, 1.5), 0.2, vec3(1.0, 1.0, 1.0) / exposure)

@ti.func
def get_uint(n): return ti.round(ti.random() * n)

@ti.func
def get_int(n): return -n + get_uint(n*2)

@ti.func
# pn: positive/negative
def get_range(n1, n2): return (n2 - n1) + n1


@ti.func
def sphere(center, radius, color):
    r = ti.ceil(radius*1.5)
    r2 = radius*radius
    for ijk in ti.grouped(ti.ndrange((-r, r), (-r, r), (-r, r))):
        if(ijk.dot(ijk) <= r2):
            scene.set_voxel(ijk+center, 1, color)


@ti.func
def stem(pos1, pos2, pos3):
    color = vec3(90, 169, 23)/255.0
    steps = 150
    mid = ti.random()*0.5 + 0.25
    a = 1/(mid)
    b = 1/((1 - mid)*(-mid))
    c = 1/(1 - mid)
    for i in ti.ndrange(steps):
        t = 1.0*i/steps
        # 拉格朗日插值
        p = a*(mid-t)*(1-t)*pos1 + b*(1-t)*(-t)*pos2 + c*(mid-t)*(-t)*pos3
        # p = (1-t)*(1-t)*pos1 + (1-t)*2*t*pos2 + t*t*pos3
        for i in ti.grouped(ti.ndrange((-1, 1), (-1, 1), (-1, 1))):
            scene.set_voxel(p+i, 1, color)
    sphere(pos3, 3, color)


@ti.func
def flower(center, radius):
    branches = 80 + 5 * radius
    steps = radius + 10; 
    inv_steps = 1.0/steps
    for i in ti.ndrange(branches):
        v = vec3(ti.random()*2-1, ti.random()*2-1, ti.random()*2-1).normalized()
        for j in ti.ndrange(steps):
            j2 = j*inv_steps
            pos = center + j2*radius*v
            color = mix(vec3(0.9, 0.8, 0.5), vec3(1.0, 1.0, 1.0), j2)
            scene.set_voxel(pos, 1, color)

@ti.func
def vase(offset, color, size):
    # x^2+z^2 = (r*sin(y)+R)^2
    r = 5; R = 10
    steps = ti.cast(120*size, ti.i16)
    steps_inv_2pi = 2*pi/steps
    height = 40*size
    omega_inv = 1.0/(6*size)
    for y in ti.ndrange(height):
        radius = (r*ti.sin(omega_inv*y) + R)*size
        for t in ti.ndrange(steps):
            theta = t*steps_inv_2pi
            x = ti.cos(theta)*radius
            z = ti.sin(theta)*radius
            scene.set_voxel(vec3(x, y, z) + offset, 1, color)


@ti.kernel
def initialize_voxels():
    offset = vec3(0, floor, 0)
    size = 1.2
    # 花瓶
    color_vase = vec3(173, 216, 230)/255.0
    vase(offset, color_vase, size)

    # 花花
    size /= 1.4
    for i in ti.ndrange(2):
        for j in ti.ndrange(2):
            d = [((1 if ti.random() > 0.5 else -1) / (i + 1)) for k in ti.ndrange(2)]
            p1 = offset + size * vec3(0, -floor/2, 0)
            p2 = offset + size * vec3(get_range(2, 10)*d[0], get_range(25, 55)+i*25, get_range(5, 10)*d[1])
            p3 = offset + size * vec3(get_range(2, 30)*d[0], get_range(45, 75)+i*25, get_range(5, 30)*d[1])
            r = ti.round(size * (15 + get_uint(10)) / (i*0.5+1))
            stem(p1, p2, p3); flower(p3, r)

initialize_voxels()

scene.finish()
