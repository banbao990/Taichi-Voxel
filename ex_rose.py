# -*- coding: utf-8 -*-
"""
Created on Tue May 17 06:15:30 2022
@author: https://github.com/lzhou-xyz
"""
# https://github.com/lzhou-xyz/voxel-challenge-2022-rosecurve/blob/main/main.py

from scene import Scene
import taichi as ti
from taichi.math import *

PI = 3.141592654

scene = Scene(voxel_edges=0, exposure=2)
scene.set_floor(-0.85, (1.0, 1.0, 1.0))
scene.set_background_color((192.0/255, 192.0/255, 192.0/255))
scene.set_directional_light((1, 1, -1), 0.2, (255.0/255, 200.0/255, 230.0/255))

@ti.func
def createbody(pos=vec3(0,0,0), r=2, height=50, color=vec3(50.0/255, 180.0/255, 30.0/255)):
    for x,y,z in ti.ndrange((-r, r), (-r, r), (-height, 0)):
        if x**2 + y ** 2 <=  r** 2:
            scene.set_voxel(pos + vec3(x, z, y), 1, color)


@ti.func
def createFloor(pos=vec3(0,-55,0),n=2,r=60):
    for theta_i in range(1000):
        theta=PI*2*((theta_i)/1000)
        x=ti.round(r*ti.sin(theta*n)*ti.cos(theta))
        y=ti.round(r*ti.sin(theta*n)*ti.sin(theta))
        scene.set_voxel(pos+vec3(x, 0, y), 1, vec3(1.0, 0.0, 0.0))

@ti.func
def creatHearta():
    ploy=[-0.0166584002935799,0.412310623807180,-0.706940917295783,2.12986774231448]
    for x,z in ti.ndrange((0,20),(-64, 64)):
        zz=ti.round(ploy[0]*ti.pow(x,3)+ploy[1]*ti.pow(x,2)+ploy[2]*x+ploy[3])
        if z==zz:
            scene.set_voxel(vec3(x,z,0),1, vec3(1.0, 0.0, 0.0))
            
@ti.func
def creatHeartout(pos=vec3(0,0,0),n=1,width=20,height=30,theta0=0,theta_range=360,color=vec3(1.0, 0.0, 0.0)):
    
    for theta_i in range(theta_range*100/360):
        theta=PI*2*theta_i/100
        for r_i in range(100):
            r=r_i*20/100
            #rr=ti.sin(n*theta)
            rr=r*ti.abs(ti.sin(n*theta))
            #rr=r*(ti.sin(n*theta))
            ploy=[-0.0166584002935799,0.412310623807180,-0.706940917295783,2.12986774231448]
            zz=(ploy[0]*ti.pow(rr,3)+ploy[1]*ti.pow(rr,2)+ploy[2]*rr+ploy[3])/30*height
            x=ti.round(rr*ti.cos(theta+theta+theta0/360*2*PI)/20*width)
            y=ti.round(rr*ti.sin(theta+theta+theta0/360*2*PI)/20*width)
            z=ti.round(zz)
            scene.set_voxel(pos+vec3(x,z,y),1, color)
            
@ti.func
def creatHeartin(pos=vec3(0,0,0),n=1,width=20,height=30,theta0=0,theta_range=360,color=vec3(1.0, 0.0, 0.0)):
    for theta_i in range(theta_range*100/360):
        theta=PI*2*((theta_i)/100)
        for r_i in range(100):
            r=r_i*20/100
            #rr=ti.sin(n*theta)
            rr=r*ti.abs(ti.sin(n*theta))
            #rr=r*(ti.sin(n*theta))
            zz=(ti.pow(rr,4)/4000/40*height)
            x=ti.round(rr*ti.cos(theta+theta0/360*2*PI)/20*width)
            y=ti.round(rr*ti.sin(theta+theta0/360*2*PI)/20*width)
            z=ti.round(zz)
            scene.set_voxel(pos+vec3(x,z,y),1, color)
            

@ti.kernel
def initialize_voxels():
    createbody()
    createFloor()
    creatHeartout(vec3(0,0,0),3,20,20,0,360,vec3(1.0, 0.0, 0.0)) #最外层
    creatHeartout(vec3(0,0,0),2,20,20,40,360,vec3(1.0, 0.0, 0.0)) #最外层2
    creatHeartin(vec3(0,0,0),1.5,10,25,0,360,vec3(1.0, 0.0, 0.0)) #内层
    creatHeartin(vec3(0,0,0),1.5,12,26,60,360,vec3(1.0, 0.0, 0.0)) #内层
    creatHeartin(vec3(0,0,0),2,8,27,30,360,vec3(1.0, 0.0, 0.0)) #内层
    creatHeartin(vec3(0,0,0),3,6,26,130,360,vec3(1.0, 0.0, 0.0)) #内层
    creatHeartin(vec3(0,0,0),4,12,15,0,360,vec3(5.0/255, 27.0/255, 18.0/255)) #叶托
    creatHeartin(vec3(0,-12,0),3,30,13,25,60,vec3(8.0/255, 30.0/255, 21.0/255)) #独叶子
    creatHeartin(vec3(0,-8,0),3,28,12,130,60,vec3(4.0/255, 28.0/255, 19.0/255)) #独叶子
    creatHeartin(vec3(0,-25,0),3,32,10,100,60,vec3(6.0/255, 35.0/255, 20.0/255)) #独叶子

initialize_voxels()
scene.finish()