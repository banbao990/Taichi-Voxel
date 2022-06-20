from scene import Scene
import taichi as ti
from taichi.math import *

# config
exposure = 2
floor = -40

scene = Scene(exposure=exposure)
scene.set_floor(floor/64.0, (1.0, 1.0, 1.0)) # position, color
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1.0, 1.5, 1.0), 0.2, vec3(1.0, 1.0, 1.0) / exposure)

# set_voxel:
# mat: 0(empty), 1(normal), 2(light)
# idx: [-64, 64)

@ti.func
def bar(width, offset, color=vec3(0.8, 0.4, 0.1), noise_color=vec3(0.2)):
    w2 = width*width
    for ijk in ti.grouped(ti.ndrange(
                    (-width, width), (floor, offset[1]), (-width, width))):
        v = ijk + 0.51
        if(v[0]*v[0] + v[2]*v[2] < w2):
            scene.set_voxel(vec3(ijk[0] + offset[0], ijk[1], ijk[2] + offset[2]),
                            1, color + noise_color*ti.random())

@ti.func
def heart(offset, color, size):
    # 使用极坐标的形式, SDF
    # 如果计算得到的距离和 r*size 比较判断内外
    # -size, size
    scale = 5.0/size
    bar(3, vec3(offset[0], offset[1] - 2.5/scale, offset[2]))
    for ijk in ti.grouped(ti.ndrange((-size, size), (-size, size))):
        x = (ijk[0] + 0.51) * scale
        y = (ijk[1] + 0.51) * scale - 2.5 # 避免除零
        r = ti.sqrt(x*x + y*y)
        x = ti.sqrt(x*x)  # x >= 0
        c = x/r
        s = y/r
        r2 =  3 - 3*s + s*ti.sqrt(c)/(s + 1.6)
        if(r < r2):
            x2 = offset[0] + ijk[0]
            y2 = offset[1] + ijk[1]
            z2 = offset[2]
            scene.set_voxel(ivec3(x2, y2, z2), 1, color)

@ti.func
def flower8(offset, color, size, num, f1_exp, f1_size):
    scale = 1.01/size
    o_t = ti.random()*pi*2 # offset_theta
    for ijk in ti.grouped(ti.ndrange((-size, size), (-size, size), (-size, size))):
        x = (ijk[0] + 0.51) * scale
        y = (ijk[2] + 0.51) * scale
        theta = ti.atan2(y, x)
        r = x*x + y*y
        r2 = ti.sin(num*theta+o_t) # num: 叶子的片数
        r2 *= r2
        if(r < r2):
            zoff = ti.round(r**f1_exp*f1_size*size)
            z = ijk[1] + 0.51
            if(ti.abs(z - zoff) < 1.0):
                scene.set_voxel(ijk + offset + 0.51, 1, color)

@ti.func
def f8(offset):
    # 注意顺序, 应该是从里到外
    c1 = vec3(255.0, 192.0, 203.0)/255.0 # color
    c2 = vec3(255.0, 0,     0    )/255.0
    c3 = c1 - c2
    r = 5 # range
    s = 1.0/r # step
    bar(2, offset)
    for i in ti.ndrange(r):
        j = r - 1 - i
        flower8(offset, c3*s*j+c2, 20-j*3, 6+j*2, 2-0.3*j, 0.25+0.4*j)
    flower8(offset, vec3(0.1,0.8,0.1), 22, 3, 2, -0.25)

@ti.kernel
def initialize_voxels():
    # [1] 爱心
    heart(vec3(20, 0, 0), vec3(0.80, 0.10, 0.20), 20)
    
    # [2] sin(theta) 花花
    f8(vec3(-20, 0, 0))


initialize_voxels()

scene.finish()
