from scene import Scene
import taichi as ti
from taichi.math import *

# config
exposure = 10

scene = Scene(exposure=exposure)
scene.set_floor(-0.95, (1.0, 1.0, 1.0)) # position, color
scene.set_background_color((0.5, 0.5, 0.4))
scene.set_directional_light((1.0, 1.5, 1.0), 0.2, vec3(1.0, 1.0, 1.0) / exposure)

# set_voxel:
# mat: 0(empty), 1(normal), 2(light)
# idx: [-64, 64)

@ti.func
def heart(offset, color, size):
    # 使用极坐标的形式, SDF
    # 如果计算得到的距离和 r*size 比较判断内外
    # -size, size
    scale = 5.0/size
    for ijk in ti.grouped(ti.ndrange((-size, size), (-size, size))):
        x = ijk[0] * scale
        y = ijk[1] * scale - 2.51 # 避免除零
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
    for ijk in ti.grouped(ti.ndrange((-size, size), (-size, size))):
        x = (ijk[0] + 0.51) * scale
        y = (ijk[1] + 0.51) * scale
        theta = ti.atan2(y, x)
        r = x*x + y*y
        r2 = ti.sin(num*theta) # num: 叶子的片数
        r2 *= r2
        if(r < r2):
            zoff = ti.round(r**f1_exp*f1_size*size)
            x2 = offset[0] + ijk[0]
            y2 = offset[1] + ijk[1]
            z2 = offset[2] + zoff
            scene.set_voxel(ivec3(x2, z2, y2), 1, color)

@ti.kernel
def initialize_voxels():
    # heart(ivec3(20, 20, 0), vec3(0.80, 0.10, 0.20), 20)
    # 注意顺序, 应该是从里到外
    offset_f8 = ivec3(0, 0, 0)
    # c1_f8 = vec3(255.0, 212.0, 125.0)/255.0
    c1_f8 = vec3(255.0, 192.0, 203.0)/255.0
    c2_f8 = vec3(255.0, 0,     0    )/255.0
    c3_f8 = c1_f8 - c2_f8
    flower8(offset_f8, c3_f8 * 0.2 + c2_f8, 20, 6, 2, 0.25)
    flower8(offset_f8, c3_f8 * 0.4 + c2_f8, 18, 8, 1.5, 0.6)
    flower8(offset_f8, c3_f8 * 0.6 + c2_f8, 15, 10, 1.3, 0.8)
    flower8(offset_f8, c3_f8 * 0.8 + c2_f8, 12, 12, 1.1, 1.0)
    flower8(offset_f8, c3_f8 * 1.0 + c2_f8, 10, 20, 1.0, 1.5)


initialize_voxels()

scene.finish()
