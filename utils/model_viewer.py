import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import time
from PIL import Image, ImageTk
import tempfile

# 添加3D预览所需的库
from OpenGL import GL, GLU
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import trimesh
from pyopengltk import OpenGLFrame

# 导入客户端类
from plugin.hunyuan.hunyuan_client import Hunyuan3DClient

class ModelViewer(OpenGLFrame):
    """使用OpenGL的3D模型预览组件"""
    
    def __init__(self, parent, **kw):
        # 确保width和height参数存在
        if 'width' not in kw:
            kw['width'] = 400
        if 'height' not in kw:
            kw['height'] = 300
            
        # 调用父类初始化
        OpenGLFrame.__init__(self, parent, **kw)
        
        # 标记OpenGL上下文是否已初始化
        self._gl_initialized = False
        
        # 初始化视图参数
        self.init_x = self.init_y = 0
        self.x = self.y = 0
        self.rotate_x = self.rotate_y = 0
        self.distance = -10.0
        self.model = None
        self.model_vertices = None
        self.model_faces = None
        self.model_normals = None
        self.model_colors = None
        self.light_position = [1.0, 1.0, 1.0, 0.0]
        self.model_info = {"filename": None, "format": None}
        
        # 设置鼠标事件处理
        self.bind("<Button-1>", self.mouse_down)
        self.bind("<B1-Motion>", self.mouse_move)
        self.bind("<MouseWheel>", self.mouse_wheel)
        self.animate = 0
        
        # 使用after方法确保在Tkinter完全初始化后创建OpenGL上下文
        self.after(100, self._ensure_gl_init)
    
    def _ensure_gl_init(self):
        """确保OpenGL上下文被初始化"""
        if self.winfo_ismapped():
            try:
                # 强制创建OpenGL上下文
                self.tkCreateContext()
                self.initgl()
                self._gl_initialized = True
                self.redraw()
            except Exception as e:
                print(f"初始化OpenGL上下文失败: {str(e)}")
                # 如果失败，稍后再试
                self.after(500, self._ensure_gl_init)
        else:
            # 如果窗口还没有映射，稍后再尝试
            self.after(500, self._ensure_gl_init)
    
    def redraw(self, *args):
        """重绘场景"""
        # 先检查窗口是否已映射以及OpenGL上下文是否已初始化
        if not self.winfo_ismapped():
            return
            
        try:
            # 尝试使上下文成为当前上下文
            self.tkMakeCurrent()
        except Exception as e:
            print(f"设置OpenGL上下文失败: {str(e)}")
            # 如果失败，尝试重新初始化
            self.after(100, self._ensure_gl_init)
            return
            
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        
        # 设置更好的透视投影
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect = self.winfo_width() / max(1, self.winfo_height())
        GLU.gluPerspective(45, aspect, 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        
        # 设置相机位置和旋转
        GL.glTranslatef(0.0, 0.0, self.distance)
        GL.glRotatef(self.rotate_x, 1.0, 0.0, 0.0)
        GL.glRotatef(self.rotate_y, 0.0, 1.0, 0.0)
        
        # 渲染模型
        if self.model_vertices is not None and self.model_faces is not None:
            GL.glBegin(GL.GL_TRIANGLES)
            for face in self.model_faces:
                for idx in face:
                    if self.model_normals is not None:
                        GL.glNormal3fv(self.model_normals[idx])
                    GL.glColor3f(0.8, 0.8, 0.8)  # 默认颜色
                    GL.glVertex3fv(self.model_vertices[idx])
            GL.glEnd()
        else:
            # 如果没有模型，绘制一个坐标系
            self._draw_axes()
        
        # 确保缓冲区交换
        try:
            self.tkSwapBuffers()
        except Exception as e:
            print(f"交换缓冲区失败: {str(e)}")
        
    def _mapped_callback(self, event):
        """当窗口映射到屏幕上后调用"""
        if self.winfo_ismapped():
            self.initgl()
            self.redraw()
        
    def mouse_down(self, event):
        """鼠标按下事件处理"""
        self.init_x = event.x
        self.init_y = event.y
        
    def mouse_move(self, event):
        """鼠标移动事件处理"""
        self.x, self.y = event.x, event.y
        self.rotate_x += (self.y - self.init_y) / 5.0
        self.rotate_y += (self.x - self.init_x) / 5.0
        self.init_x, self.init_y = self.x, self.y
        self.redraw()
        
    def mouse_wheel(self, event):
        """鼠标滚轮事件处理"""
        # Windows平台上可能没有delta属性，而是使用其他方式
        try:
            delta = event.delta
        except AttributeError:
            # 尝试使用Windows特定的滚轮属性
            if event.num == 4:
                delta = 120
            elif event.num == 5:
                delta = -120
            else:
                delta = 0
        
        # 缩放处理
        if delta > 0:
            self.distance += 0.5
        elif delta < 0:
            self.distance -= 0.5
        
        self.redraw()
        
    def initgl(self):
        """初始化OpenGL设置"""
        # 确保上下文已创建
        try:
            GL.glClearColor(0.2, 0.2, 0.2, 1.0)
            GL.glShadeModel(GL.GL_SMOOTH)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_LIGHT0)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, self.light_position)
            GL.glEnable(GL.GL_COLOR_MATERIAL)
            GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
            return True
        except Exception as e:
            print(f"OpenGL初始化错误: {str(e)}")
            return False
            
    def _draw_axes(self):
        """绘制坐标轴"""
        GL.glBegin(GL.GL_LINES)
        # X轴 - 红色
        GL.glColor3f(1.0, 0.0, 0.0)
        GL.glVertex3f(0.0, 0.0, 0.0)
        GL.glVertex3f(1.0, 0.0, 0.0)
        # Y轴 - 绿色
        GL.glColor3f(0.0, 1.0, 0.0)
        GL.glVertex3f(0.0, 0.0, 0.0)
        GL.glVertex3f(0.0, 1.0, 0.0)
        # Z轴 - 蓝色
        GL.glColor3f(0.0, 0.0, 1.0)
        GL.glVertex3f(0.0, 0.0, 0.0)
        GL.glVertex3f(0.0, 0.0, 1.0)
        GL.glEnd()
            
    def load_model(self, model_path):
        """加载3D模型"""
        try:
            print(f"开始加载模型: {model_path}")
            
            # 存储模型信息
            self.model_info["filename"] = os.path.basename(model_path)
            self.model_info["format"] = os.path.splitext(model_path)[1].lower().replace(".", "")
            
            # 使用trimesh加载模型
            mesh = trimesh.load(model_path)
            
            # 检查是否为复合模型
            if isinstance(mesh, trimesh.Scene):
                print("检测到模型是场景(Scene)，尝试合并所有网格")
                mesh = mesh.dump(concatenate=True)
            
            # 获取顶点、面和法线
            self.model_vertices = np.array(mesh.vertices, dtype=np.float32)
            self.model_faces = np.array(mesh.faces, dtype=np.int32)
            
            print(f"模型加载成功: 顶点数={len(self.model_vertices)}, 面数={len(self.model_faces)}")
            
            # 尝试获取顶点法线，如果不存在则计算
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                self.model_normals = np.array(mesh.vertex_normals, dtype=np.float32)
                print("使用模型提供的顶点法线")
            else:
                print("计算顶点法线...")
                # 计算法线
                self.model_normals = np.zeros_like(self.model_vertices)
                for face in self.model_faces:
                    # 计算三角形面法线
                    v0 = self.model_vertices[face[0]]
                    v1 = self.model_vertices[face[1]]
                    v2 = self.model_vertices[face[2]]
                    normal = np.cross(v1 - v0, v2 - v0)
                    normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0,0,1])
                    
                    # 将面法线加到顶点法线
                    self.model_normals[face[0]] += normal
                    self.model_normals[face[1]] += normal
                    self.model_normals[face[2]] += normal
            
            # 归一化顶点法线
            norms = np.linalg.norm(self.model_normals, axis=1)
            norms[norms == 0] = 1  # 避免除零
            self.model_normals = self.model_normals / norms[:, np.newaxis]
            
            # 居中和缩放模型
            print("居中和缩放模型...")
            self._center_and_scale_model()
            
            # 确保OpenGL上下文正确初始化
            self._ensure_gl_init()
            
            # 更新显示
            print("重绘模型...")
            self.redraw()
            return True
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            return False
            
    def _center_and_scale_model(self):
        """居中和缩放模型以适应视图"""
        if self.model_vertices is None:
            return
            
        # 计算边界框
        min_coords = np.min(self.model_vertices, axis=0)
        max_coords = np.max(self.model_vertices, axis=0)
        
        # 计算中心点
        center = (min_coords + max_coords) / 2.0
        
        # 平移顶点到中心
        self.model_vertices -= center
        
        # 计算缩放因子
        scale = 1.0 / max(max_coords - min_coords)
        
        # 缩放模型
        self.model_vertices *= scale * 5.0  # 放大一点以便更好地查看
        
    def clear_model(self):
        """清除当前模型"""
        self.model_vertices = None
        self.model_faces = None
        self.model_normals = None
        self.redraw()
        
    def show_test_cube(self):
        """显示测试用立方体，验证渲染是否正常工作"""
        # 定义一个简单的立方体
        vertices = np.array([
            # 前面
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            # 后面
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5]
        ], dtype=np.float32)
        
        # 简单的面定义
        faces = np.array([
            [0, 1, 2], [2, 3, 0],  # 前面
            [1, 5, 6], [6, 2, 1],  # 右面
            [5, 4, 7], [7, 6, 5],  # 后面
            [4, 0, 3], [3, 7, 4],  # 左面
            [3, 2, 6], [6, 7, 3],  # 顶面
            [4, 5, 1], [1, 0, 4]   # 底面
        ], dtype=np.int32)
        
        # 设置模型数据
        self.model_vertices = vertices
        self.model_faces = faces
        
        # 计算简单的法线
        self.model_normals = np.zeros_like(vertices)
        for face in faces:
            # 计算面法线
            v0, v1, v2 = vertices[face]
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0,0,1])
            
            # 将面法线加到顶点法线
            for idx in face:
                self.model_normals[idx] += normal
        
        # 归一化顶点法线
        norms = np.linalg.norm(self.model_normals, axis=1)
        norms[norms == 0] = 1  # 避免除零
        self.model_normals = self.model_normals / norms[:, np.newaxis]
        
        print("测试立方体已加载")
        
        # 确保模型居中和缩放
        self._center_and_scale_model()
        self.redraw()
        return True