import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import math
import os
import json
import trimesh
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image, ImageTk
import time
import threading
from utils.export import MODEL_LIBRARY, MATERIAL_LIBRARY
from pyopengltk import OpenGLFrame

class ModelViewerFrame(OpenGLFrame):
    """使用PyOpenGLTk的OpenGL视图框架，用于替代Pygame渲染"""
    
    def __init__(self, parent, **kw):
        # 设置默认尺寸
        if 'width' not in kw:
            kw['width'] = 800
        if 'height' not in kw:
            kw['height'] = 600
            
        # 调用父类初始化
        OpenGLFrame.__init__(self, parent, **kw)
        
        # 初始化视图参数
        self.camera_distance = 5.0
        self.camera_rotation_x = 45.0
        self.camera_rotation_y = 45.0
        self.wireframe_mode = False
        self.show_grid = True
        self.show_axes = True
        
        # 模型数据
        self.models = []
        self.selected_model_index = -1
        
        # 鼠标事件绑定
        self.bind("<Button-1>", self.on_mouse_button_down)
        self.bind("<ButtonRelease-1>", self.on_mouse_button_up)
        self.bind("<B1-Motion>", self.on_mouse_motion)
        self.bind("<MouseWheel>", self.on_mouse_wheel)
        self.dragging = False
        self.mouse_x = 0
        self.mouse_y = 0
        
        # 初始化OpenGL
        self.init_complete = False
        self.after(100, self.check_init)
    
    def check_init(self):
        """检查并确保初始化完成"""
        if self.winfo_ismapped():
            try:
                self.tkCreateContext()
                self.initgl()
                self.init_complete = True
                self.redraw()
            except Exception as e:
                print(f"OpenGL初始化失败: {str(e)}")
                self.after(500, self.check_init)
        else:
            self.after(500, self.check_init)
    
    def initgl(self):
        """初始化OpenGL设置"""
        glClearColor(0.3, 0.3, 0.3, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 1.0, 2.0, 0.0])
        glEnable(GL_COLOR_MATERIAL)
        glShadeModel(GL_SMOOTH)
        glDisable(GL_CULL_FACE)
    
    def redraw(self, *args):
        """重绘场景"""
        if not self.winfo_ismapped() or not hasattr(self, 'init_complete') or not self.init_complete:
            return
            
        # 确保OpenGL上下文是当前的
        try:
            self.tkMakeCurrent()
        except Exception as e:
            print(f"设置OpenGL上下文失败: {str(e)}")
            self.after(100, self.check_init)
            return
            
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置透视投影
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width = self.winfo_width()
        height = max(1, self.winfo_height())  # 避免除零
        gluPerspective(45, width / height, 0.1, 100.0)
        
        # 设置模型视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 设置相机位置
        glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_rotation_x, 1, 0, 0)
        glRotatef(self.camera_rotation_y, 0, 1, 0)
        
        # 渲染场景内容
        if self.show_grid:
            self.draw_grid()
        
        if self.show_axes:
            self.draw_axes()
        
        # 设置线框模式
        if self.wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            
        # 绘制模型
        self.draw_models()
        
        # 恢复默认渲染模式
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # 交换缓冲区
        try:
            self.tkSwapBuffers()
        except Exception as e:
            print(f"交换缓冲区失败: {str(e)}")
    
    def draw_grid(self):
        """绘制网格"""
        glDisable(GL_LIGHTING)
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(1.0)
        
        grid_size = 10
        grid_step = 1.0
        
        glBegin(GL_LINES)
        for i in range(-grid_size, grid_size + 1):
            # 沿X轴的线
            glVertex3f(i * grid_step, 0, -grid_size * grid_step)
            glVertex3f(i * grid_step, 0, grid_size * grid_step)
            
            # 沿Z轴的线
            glVertex3f(-grid_size * grid_step, 0, i * grid_step)
            glVertex3f(grid_size * grid_step, 0, i * grid_step)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_axes(self):
        """绘制坐标轴"""
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        
        # X轴 (红色)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(2.0, 0, 0)
        glEnd()
        
        # Y轴 (绿色)
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 2.0, 0)
        glEnd()
        
        # Z轴 (蓝色)
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 2.0)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_models(self):
        """绘制所有模型"""
        # 设置线框模式
        if self.wireframe_mode:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # 绘制每个模型
        for i, model in enumerate(self.models):
            try:
                # 保存当前矩阵
                glPushMatrix()
                
                # 应用模型变换
                glTranslatef(model['position'][0], model['position'][1], model['position'][2])
                glRotatef(model['rotation'][0], 1, 0, 0)
                glRotatef(model['rotation'][1], 0, 1, 0)
                glRotatef(model['rotation'][2], 0, 0, 1)
                glScalef(model['scale'][0], model['scale'][1], model['scale'][2])
                
                # 如果是选中的模型，绘制选择框
                if i == self.selected_model_index:
                    self.draw_selection_box(model)
                
                # 应用材质
                material = model.get('material', {})
                diffuse_color = material.get('diffuse_color', [0.8, 0.8, 0.8])
                specular = material.get('specular', 0.5)
                
                glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, diffuse_color + [1.0])
                glMaterialfv(GL_FRONT, GL_SPECULAR, [specular, specular, specular, 1.0])
                glMaterialf(GL_FRONT, GL_SHININESS, (1.0 - material.get('roughness', 0.5)) * 128)
                
                # 绘制网格
                self.draw_mesh(model['mesh_data'])
                
                # 恢复矩阵
                glPopMatrix()
            except Exception as e:
                # 确保即使出错也能平衡矩阵栈
                print(f"绘制模型时出错: {str(e)}")
                # 尝试恢复矩阵栈平衡
                try:
                    glPopMatrix()
                except:
                    pass
        
        # 恢复默认渲染模式
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    
    def draw_selection_box(self, model):
        """绘制选择框"""
        # 使用原来的draw_selection_box方法逻辑...
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 0.0)  # 黄色
        glLineWidth(2.0)
        
        if 'bounds' in model:
            min_bound, max_bound = model['bounds']
            
            # 绘制边界框
            glBegin(GL_LINE_LOOP)
            glVertex3f(min_bound[0], min_bound[1], min_bound[2])
            glVertex3f(max_bound[0], min_bound[1], min_bound[2])
            glVertex3f(max_bound[0], min_bound[1], max_bound[2])
            glVertex3f(min_bound[0], min_bound[1], max_bound[2])
            glEnd()
            
            # 其余边界框线条绘制逻辑...
            glBegin(GL_LINE_LOOP)
            glVertex3f(min_bound[0], max_bound[1], min_bound[2])
            glVertex3f(max_bound[0], max_bound[1], min_bound[2])
            glVertex3f(max_bound[0], max_bound[1], max_bound[2])
            glVertex3f(min_bound[0], max_bound[1], max_bound[2])
            glEnd()
            
            glBegin(GL_LINES)
            # 连接上下边界框
            glVertex3f(min_bound[0], min_bound[1], min_bound[2])
            glVertex3f(min_bound[0], max_bound[1], min_bound[2])
            
            glVertex3f(max_bound[0], min_bound[1], min_bound[2])
            glVertex3f(max_bound[0], max_bound[1], min_bound[2])
            
            glVertex3f(max_bound[0], min_bound[1], max_bound[2])
            glVertex3f(max_bound[0], max_bound[1], max_bound[2])
            
            glVertex3f(min_bound[0], min_bound[1], max_bound[2])
            glVertex3f(min_bound[0], max_bound[1], max_bound[2])
            glEnd()
        
        glEnable(GL_LIGHTING)
    
    def draw_mesh(self, mesh_data):
        """绘制网格数据"""
        if not mesh_data:
            return
        
        vertices = mesh_data.get('vertices', [])
        normals = mesh_data.get('normals', [])
        faces = mesh_data.get('faces', [])
        
        glBegin(GL_TRIANGLES)
        for face in faces:
            for i, vertex_idx in enumerate(face):
                if normals and len(normals) > vertex_idx:
                    normal = normals[vertex_idx]
                    glNormal3f(normal[0], normal[1], normal[2])
                
                if len(vertices) > vertex_idx:
                    vertex = vertices[vertex_idx]
                    glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()
    
    # 添加必要的鼠标处理方法
    def on_mouse_button_down(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.dragging = True
    
    def on_mouse_button_up(self, event):
        self.dragging = False
    
    def on_mouse_motion(self, event):
        if self.dragging:
            dx = event.x - self.mouse_x
            dy = event.y - self.mouse_y
            self.mouse_x = event.x
            self.mouse_y = event.y
            
            # 旋转视图
            self.camera_rotation_y += dx * 0.5
            self.camera_rotation_x += dy * 0.5
            
            # 限制X轴旋转角度
            self.camera_rotation_x = max(-90, min(90, self.camera_rotation_x))
            
            self.redraw()
    
    def on_mouse_wheel(self, event):
        delta = event.delta
        
        if delta < 0:
            self.camera_distance += 0.5
        else:
            self.camera_distance -= 0.5
        
        # 限制缩放范围
        self.camera_distance = max(2.0, min(50.0, self.camera_distance))
        
        self.redraw()

class Model3DEditor:
    """3D模型编辑器类，提供与Blender类似的基本3D建模功能"""
    
    def __init__(self, master, map_data=None, logger=None):
        self.master = master
        self.map_data = map_data
        self.logger = logger
        
        # 不再需要pygame.init()
        
        # 创建主对话框窗口
        self.dialog = tk.Toplevel(master)
        self.dialog.title("3D模型编辑器")
        self.dialog.geometry("1200x800")
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 模型数据存储
        self.models = []
        self.selected_model = None
        self.selected_model_index = -1
        
        # 编辑状态
        self.edit_mode = "select"  # select, move, rotate, scale
        self.edit_axis = "xyz"     # x, y, z, xyz
        
        # 模型库
        self.model_library = MODEL_LIBRARY.copy()
        self.material_library = MATERIAL_LIBRARY.copy()
        
        # 确保状态栏在最开始初始化
        self.status_bar = ttk.Frame(self.dialog)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="就绪")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.coords_label = ttk.Label(self.status_bar, text="")
        self.coords_label.pack(side=tk.RIGHT, padx=5)
        
        # 创建UI
        self.setup_ui()
        
        # 加载地图数据中的模型（如果有）
        if map_data is not None:
            self.load_models_from_map_data()

    def update_gl_models(self):
        """将当前模型列表同步到GL视图框架"""
        if hasattr(self, 'gl_frame'):
            self.gl_frame.models = self.models
            self.gl_frame.selected_model_index = self.selected_model_index
            self.gl_frame.redraw()
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主分隔窗格
        self.main_paned = ttk.PanedWindow(self.dialog, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        self.control_frame = ttk.Frame(self.main_paned, width=300)
        self.main_paned.add(self.control_frame, weight=1)
        
        # 右侧3D视图面板
        self.view_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.view_frame, weight=3)
        
        # 设置左侧控制面板内容
        self.setup_control_panel()
        
        # 设置右侧3D视图面板
        self.setup_view_panel()
        
        # 设置底部状态栏
        self.setup_status_bar()
    
    def setup_control_panel(self):
        """设置控制面板"""
        # 创建Notebook选项卡
        self.control_notebook = ttk.Notebook(self.control_frame)
        self.control_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 模型列表选项卡
        self.models_tab = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.models_tab, text="模型")
        
        # 编辑选项卡
        self.edit_tab = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.edit_tab, text="编辑")
        
        # 材质选项卡
        self.materials_tab = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.materials_tab, text="材质")
        
        # 设置各选项卡的内容
        self.setup_models_tab()
        self.setup_edit_tab()
        self.setup_materials_tab()
    
    def setup_models_tab(self):
        """设置模型列表选项卡"""
        # 模型列表框架
        models_frame = ttk.LabelFrame(self.models_tab, text="模型列表")
        models_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建模型列表
        self.models_listbox = tk.Listbox(models_frame, height=15)
        self.models_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.models_listbox.bind("<<ListboxSelect>>", self.on_model_select)
        
        # 添加滚动条
        models_scrollbar = ttk.Scrollbar(self.models_listbox, orient="vertical")
        models_scrollbar.config(command=self.models_listbox.yview)
        self.models_listbox.config(yscrollcommand=models_scrollbar.set)
        models_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 按钮框架
        buttons_frame = ttk.Frame(models_frame)
        buttons_frame.pack(fill=tk.X, pady=5)
        
        # 添加模型按钮
        add_model_button = ttk.Button(buttons_frame, text="添加模型", command=self.add_model_dialog)
        add_model_button.pack(side=tk.LEFT, padx=2)
        
        # 删除模型按钮
        delete_model_button = ttk.Button(buttons_frame, text="删除模型", command=self.delete_selected_model)
        delete_model_button.pack(side=tk.LEFT, padx=2)
        
        # 复制模型按钮
        duplicate_model_button = ttk.Button(buttons_frame, text="复制模型", command=self.duplicate_selected_model)
        duplicate_model_button.pack(side=tk.LEFT, padx=2)
        
        # 导入模型框架
        import_frame = ttk.LabelFrame(self.models_tab, text="导入模型")
        import_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # 导入OBJ模型按钮
        import_obj_button = ttk.Button(import_frame, text="导入OBJ文件", command=self.import_obj_file)
        import_obj_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 从模型库导入按钮
        import_library_button = ttk.Button(import_frame, text="从模型库导入", command=self.import_from_library)
        import_library_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 导出模型框架
        export_frame = ttk.LabelFrame(self.models_tab, text="导出模型")
        export_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # 导出全部按钮
        export_all_button = ttk.Button(export_frame, text="导出全部模型", command=self.export_all_models)
        export_all_button.pack(fill=tk.X, padx=5, pady=5)
        
        # 导出选中模型按钮
        export_selected_button = ttk.Button(export_frame, text="导出选中模型", command=self.export_selected_model)
        export_selected_button.pack(fill=tk.X, padx=5, pady=5)
    
    def setup_edit_tab(self):
        """设置编辑选项卡"""
        # 变换工具框架
        transform_frame = ttk.LabelFrame(self.edit_tab, text="变换工具")
        transform_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 编辑模式选择
        mode_frame = ttk.Frame(transform_frame)
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mode_frame, text="编辑模式:").pack(side=tk.LEFT)
        
        self.edit_mode_var = tk.StringVar(value="select")
        mode_select = ttk.Combobox(mode_frame, textvariable=self.edit_mode_var, 
                                   values=["select", "move", "rotate", "scale"],
                                   state="readonly", width=10)
        mode_select.pack(side=tk.LEFT, padx=5)
        mode_select.bind("<<ComboboxSelected>>", lambda e: self.set_edit_mode(self.edit_mode_var.get()))
        
        # 轴向选择
        axis_frame = ttk.Frame(transform_frame)
        axis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(axis_frame, text="编辑轴向:").pack(side=tk.LEFT)
        
        self.edit_axis_var = tk.StringVar(value="xyz")
        axis_select = ttk.Combobox(axis_frame, textvariable=self.edit_axis_var, 
                                   values=["x", "y", "z", "xyz"],
                                   state="readonly", width=10)
        axis_select.pack(side=tk.LEFT, padx=5)
        axis_select.bind("<<ComboboxSelected>>", lambda e: self.set_edit_axis(self.edit_axis_var.get()))
        
        # 位置编辑框架
        position_frame = ttk.LabelFrame(self.edit_tab, text="位置")
        position_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # X坐标
        x_frame = ttk.Frame(position_frame)
        x_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(x_frame, text="X:").pack(side=tk.LEFT)
        self.position_x_var = tk.DoubleVar(value=0.0)
        x_entry = ttk.Spinbox(x_frame, from_=-1000, to=1000, increment=0.1, textvariable=self.position_x_var, width=10)
        x_entry.pack(side=tk.LEFT, padx=5)
        self.position_x_var.trace_add("write", lambda n, i, m: self.update_model_position())
        
        # Y坐标
        y_frame = ttk.Frame(position_frame)
        y_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(y_frame, text="Y:").pack(side=tk.LEFT)
        self.position_y_var = tk.DoubleVar(value=0.0)
        y_entry = ttk.Spinbox(y_frame, from_=-1000, to=1000, increment=0.1, textvariable=self.position_y_var, width=10)
        y_entry.pack(side=tk.LEFT, padx=5)
        self.position_y_var.trace_add("write", lambda n, i, m: self.update_model_position())
        
        # Z坐标
        z_frame = ttk.Frame(position_frame)
        z_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(z_frame, text="Z:").pack(side=tk.LEFT)
        self.position_z_var = tk.DoubleVar(value=0.0)
        z_entry = ttk.Spinbox(z_frame, from_=-1000, to=1000, increment=0.1, textvariable=self.position_z_var, width=10)
        z_entry.pack(side=tk.LEFT, padx=5)
        self.position_z_var.trace_add("write", lambda n, i, m: self.update_model_position())
        
        # 旋转编辑框架
        rotation_frame = ttk.LabelFrame(self.edit_tab, text="旋转 (度)")
        rotation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # X旋转
        rx_frame = ttk.Frame(rotation_frame)
        rx_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(rx_frame, text="X:").pack(side=tk.LEFT)
        self.rotation_x_var = tk.DoubleVar(value=0.0)
        rx_entry = ttk.Spinbox(rx_frame, from_=-360, to=360, increment=1, textvariable=self.rotation_x_var, width=10)
        rx_entry.pack(side=tk.LEFT, padx=5)
        self.rotation_x_var.trace_add("write", lambda n, i, m: self.update_model_rotation())
        
        # Y旋转
        ry_frame = ttk.Frame(rotation_frame)
        ry_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(ry_frame, text="Y:").pack(side=tk.LEFT)
        self.rotation_y_var = tk.DoubleVar(value=0.0)
        ry_entry = ttk.Spinbox(ry_frame, from_=-360, to=360, increment=1, textvariable=self.rotation_y_var, width=10)
        ry_entry.pack(side=tk.LEFT, padx=5)
        self.rotation_y_var.trace_add("write", lambda n, i, m: self.update_model_rotation())
        
        # Z旋转
        rz_frame = ttk.Frame(rotation_frame)
        rz_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(rz_frame, text="Z:").pack(side=tk.LEFT)
        self.rotation_z_var = tk.DoubleVar(value=0.0)
        rz_entry = ttk.Spinbox(rz_frame, from_=-360, to=360, increment=1, textvariable=self.rotation_z_var, width=10)
        rz_entry.pack(side=tk.LEFT, padx=5)
        self.rotation_z_var.trace_add("write", lambda n, i, m: self.update_model_rotation())
        
        # 缩放编辑框架
        scale_frame = ttk.LabelFrame(self.edit_tab, text="缩放")
        scale_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 统一缩放
        uniform_scale_frame = ttk.Frame(scale_frame)
        uniform_scale_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(uniform_scale_frame, text="统一缩放:").pack(side=tk.LEFT)
        self.uniform_scale_var = tk.BooleanVar(value=True)
        uniform_check = ttk.Checkbutton(uniform_scale_frame, variable=self.uniform_scale_var)
        uniform_check.pack(side=tk.LEFT, padx=5)
        
        # X缩放
        sx_frame = ttk.Frame(scale_frame)
        sx_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sx_frame, text="X:").pack(side=tk.LEFT)
        self.scale_x_var = tk.DoubleVar(value=1.0)
        sx_entry = ttk.Spinbox(sx_frame, from_=0.01, to=100, increment=0.1, textvariable=self.scale_x_var, width=10)
        sx_entry.pack(side=tk.LEFT, padx=5)
        self.scale_x_var.trace_add("write", lambda n, i, m: self.update_model_scale(axis='x'))
        
        # Y缩放
        sy_frame = ttk.Frame(scale_frame)
        sy_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sy_frame, text="Y:").pack(side=tk.LEFT)
        self.scale_y_var = tk.DoubleVar(value=1.0)
        sy_entry = ttk.Spinbox(sy_frame, from_=0.01, to=100, increment=0.1, textvariable=self.scale_y_var, width=10)
        sy_entry.pack(side=tk.LEFT, padx=5)
        self.scale_y_var.trace_add("write", lambda n, i, m: self.update_model_scale(axis='y'))
        
        # Z缩放
        sz_frame = ttk.Frame(scale_frame)
        sz_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sz_frame, text="Z:").pack(side=tk.LEFT)
        self.scale_z_var = tk.DoubleVar(value=1.0)
        sz_entry = ttk.Spinbox(sz_frame, from_=0.01, to=100, increment=0.1, textvariable=self.scale_z_var, width=10)
        sz_entry.pack(side=tk.LEFT, padx=5)
        self.scale_z_var.trace_add("write", lambda n, i, m: self.update_model_scale(axis='z'))
        
        # 高级编辑框架
        advanced_frame = ttk.LabelFrame(self.edit_tab, text="高级编辑")
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 镜像按钮
        mirror_frame = ttk.Frame(advanced_frame)
        mirror_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(mirror_frame, text="X轴镜像", command=lambda: self.mirror_model('x')).pack(side=tk.LEFT, padx=2)
        ttk.Button(mirror_frame, text="Y轴镜像", command=lambda: self.mirror_model('y')).pack(side=tk.LEFT, padx=2)
        ttk.Button(mirror_frame, text="Z轴镜像", command=lambda: self.mirror_model('z')).pack(side=tk.LEFT, padx=2)
        
        # 重置变换按钮
        reset_frame = ttk.Frame(advanced_frame)
        reset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(reset_frame, text="重置位置", command=self.reset_position).pack(side=tk.LEFT, padx=2)
        ttk.Button(reset_frame, text="重置旋转", command=self.reset_rotation).pack(side=tk.LEFT, padx=2)
        ttk.Button(reset_frame, text="重置缩放", command=self.reset_scale).pack(side=tk.LEFT, padx=2)
    
    def setup_materials_tab(self):
        """设置材质选项卡"""
        # 当前材质框架
        current_material_frame = ttk.LabelFrame(self.materials_tab, text="当前材质")
        current_material_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 材质名称
        name_frame = ttk.Frame(current_material_frame)
        name_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(name_frame, text="名称:").pack(side=tk.LEFT)
        self.material_name_var = tk.StringVar(value="默认材质")
        name_entry = ttk.Entry(name_frame, textvariable=self.material_name_var, width=20)
        name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 颜色选择框架
        color_frame = ttk.Frame(current_material_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 漫反射颜色 - 使用tk.Button而不是ttk.Button
        diffuse_frame = ttk.Frame(color_frame)
        diffuse_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(diffuse_frame, text="漫反射颜色:").pack(side=tk.LEFT)
        self.diffuse_color_button = tk.Button(diffuse_frame, text="   ", width=3, background='#cccccc')
        self.diffuse_color_button.pack(side=tk.LEFT, padx=5)
        self.diffuse_color_button.configure(command=self.pick_diffuse_color)
        
        # 反射率
        specular_frame = ttk.Frame(current_material_frame)
        specular_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(specular_frame, text="反射率:").pack(side=tk.LEFT)
        self.specular_var = tk.DoubleVar(value=0.5)
        specular_scale = ttk.Scale(specular_frame, from_=0.0, to=1.0, variable=self.specular_var, orient=tk.HORIZONTAL)
        specular_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.specular_var.trace_add("write", lambda n, i, m: self.update_material_properties())
        
        # 粗糙度
        roughness_frame = ttk.Frame(current_material_frame)
        roughness_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(roughness_frame, text="粗糙度:").pack(side=tk.LEFT)
        self.roughness_var = tk.DoubleVar(value=0.5)
        roughness_scale = ttk.Scale(roughness_frame, from_=0.0, to=1.0, variable=self.roughness_var, orient=tk.HORIZONTAL)
        roughness_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.roughness_var.trace_add("write", lambda n, i, m: self.update_material_properties())
        
        # 金属度
        metallic_frame = ttk.Frame(current_material_frame)
        metallic_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(metallic_frame, text="金属度:").pack(side=tk.LEFT)
        self.metallic_var = tk.DoubleVar(value=0.0)
        metallic_scale = ttk.Scale(metallic_frame, from_=0.0, to=1.0, variable=self.metallic_var, orient=tk.HORIZONTAL)
        metallic_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.metallic_var.trace_add("write", lambda n, i, m: self.update_material_properties())
        
        # 透明度
        transparency_frame = ttk.Frame(current_material_frame)
        transparency_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(transparency_frame, text="透明度:").pack(side=tk.LEFT)
        self.transparency_var = tk.DoubleVar(value=0.0)
        transparency_scale = ttk.Scale(transparency_frame, from_=0.0, to=1.0, variable=self.transparency_var, orient=tk.HORIZONTAL)
        transparency_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.transparency_var.trace_add("write", lambda n, i, m: self.update_material_properties())
        
        # 纹理框架
        texture_frame = ttk.LabelFrame(self.materials_tab, text="纹理贴图")
        texture_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 漫反射贴图
        diffuse_map_frame = ttk.Frame(texture_frame)
        diffuse_map_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(diffuse_map_frame, text="漫反射贴图:").pack(side=tk.LEFT)
        self.diffuse_map_var = tk.StringVar(value="")
        diffuse_map_entry = ttk.Entry(diffuse_map_frame, textvariable=self.diffuse_map_var, width=15)
        diffuse_map_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(diffuse_map_frame, text="浏览", command=lambda: self.browse_texture_file("diffuse")).pack(side=tk.LEFT)
        
        # 法线贴图
        normal_map_frame = ttk.Frame(texture_frame)
        normal_map_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(normal_map_frame, text="法线贴图:").pack(side=tk.LEFT)
        self.normal_map_var = tk.StringVar(value="")
        normal_map_entry = ttk.Entry(normal_map_frame, textvariable=self.normal_map_var, width=15)
        normal_map_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(normal_map_frame, text="浏览", command=lambda: self.browse_texture_file("normal")).pack(side=tk.LEFT)
        
        # 粗糙度贴图
        roughness_map_frame = ttk.Frame(texture_frame)
        roughness_map_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(roughness_map_frame, text="粗糙度贴图:").pack(side=tk.LEFT)
        self.roughness_map_var = tk.StringVar(value="")
        roughness_map_entry = ttk.Entry(roughness_map_frame, textvariable=self.roughness_map_var, width=15)
        roughness_map_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(roughness_map_frame, text="浏览", command=lambda: self.browse_texture_file("roughness")).pack(side=tk.LEFT)
        
        # 材质库按钮
        library_frame = ttk.Frame(self.materials_tab)
        library_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(library_frame, text="从材质库加载", command=self.load_from_material_library).pack(side=tk.LEFT, padx=2)
        ttk.Button(library_frame, text="保存到材质库", command=self.save_to_material_library).pack(side=tk.LEFT, padx=2)
        ttk.Button(library_frame, text="应用到选中模型", command=self.apply_material_to_selected).pack(side=tk.LEFT, padx=2)
    
    def setup_view_panel(self):
        """设置3D视图面板"""
        # 创建工具栏
        toolbar_frame = ttk.Frame(self.view_frame)
        toolbar_frame.pack(fill=tk.X)
        
        # 视图控制按钮
        ttk.Button(toolbar_frame, text="顶视图", command=lambda: self.set_camera_view("top")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="前视图", command=lambda: self.set_camera_view("front")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="侧视图", command=lambda: self.set_camera_view("side")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar_frame, text="透视图", command=lambda: self.set_camera_view("perspective")).pack(side=tk.LEFT, padx=2)
        
        # 显示控制
        ttk.Label(toolbar_frame, text="   显示:").pack(side=tk.LEFT)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar_frame, text="网格", variable=self.show_grid_var, 
                    command=self._update_display_options).pack(side=tk.LEFT)
        
        self.show_axes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar_frame, text="坐标轴", variable=self.show_axes_var,
                    command=self._update_display_options).pack(side=tk.LEFT)
        
        self.wireframe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar_frame, text="线框模式", variable=self.wireframe_var,
                    command=self._update_display_options).pack(side=tk.LEFT)
        
        # 创建OpenGL视图框架
        self.gl_frame = ModelViewerFrame(self.view_frame)
        self.gl_frame.pack(fill=tk.BOTH, expand=True)
        
        # 确保控件可以接收键盘焦点
        self.gl_frame.focus_set()
        
        # 键盘快捷键绑定到gl_frame
        self.gl_frame.bind("<Key-1>", lambda e: self.set_camera_view("top"))
        self.gl_frame.bind("<Key-2>", lambda e: self.set_camera_view("front"))
        self.gl_frame.bind("<Key-3>", lambda e: self.set_camera_view("side"))
        self.gl_frame.bind("<Key-4>", lambda e: self.set_camera_view("perspective"))
        self.gl_frame.bind("<Key-r>", lambda e: self.reset_camera())

    def _update_display_options(self):
        """更新显示选项"""
        if hasattr(self, 'gl_frame'):
            self.gl_frame.show_grid = self.show_grid_var.get()
            self.gl_frame.show_axes = self.show_axes_var.get()
            self.gl_frame.wireframe_mode = self.wireframe_var.get()
            self.gl_frame.redraw()

    def set_camera_view(self, view_type):
        """设置摄像机视图"""
        if not hasattr(self, 'gl_frame'):
            return
            
        if view_type == "top":
            self.gl_frame.camera_rotation_x = 90.0
            self.gl_frame.camera_rotation_y = 0.0
        elif view_type == "front":
            self.gl_frame.camera_rotation_x = 0.0
            self.gl_frame.camera_rotation_y = 0.0
        elif view_type == "side":
            self.gl_frame.camera_rotation_x = 0.0
            self.gl_frame.camera_rotation_y = 90.0
        elif view_type == "perspective":
            self.gl_frame.camera_rotation_x = 20.0
            self.gl_frame.camera_rotation_y = 30.0
        
        self.gl_frame.redraw()

    def reset_camera(self):
        """重置相机位置"""
        if hasattr(self, 'gl_frame'):
            self.gl_frame.camera_distance = 5.0
            self.gl_frame.camera_rotation_x = 20.0
            self.gl_frame.camera_rotation_y = 30.0
            self.gl_frame.redraw()

    def _setup_opengl_pygame_directx(self):
        """使用 DirectX 驱动初始化 OpenGL"""
        pygame.init()
        os.environ['SDL_VIDEODRIVER'] = 'directx'
        pygame.display.init()
        self.screen_width = self.gl_frame.winfo_width() or 800
        self.screen_height = self.gl_frame.winfo_height() or 600
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )

    def _setup_opengl_pygame_default(self):
        """使用默认驱动初始化 OpenGL"""
        pygame.init()
        if 'SDL_VIDEODRIVER' in os.environ:
            del os.environ['SDL_VIDEODRIVER']
        pygame.display.quit()
        pygame.display.init()
        self.screen_width = self.gl_frame.winfo_width() or 800
        self.screen_height = self.gl_frame.winfo_height() or 600
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )

    def _setup_opengl_pygame_windib(self):
        """使用 windib 驱动初始化 OpenGL"""
        pygame.init()
        os.environ['SDL_VIDEODRIVER'] = 'windib'
        pygame.display.quit()
        pygame.display.init()
        self.screen_width = self.gl_frame.winfo_width() or 800
        self.screen_height = self.gl_frame.winfo_height() or 600
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )

    def _setup_opengl_standalone(self):
        """创建独立的 OpenGL 窗口"""
        try:
            pygame.init()
            if 'SDL_WINDOWID' in os.environ:
                del os.environ['SDL_WINDOWID']
            if 'SDL_VIDEODRIVER' in os.environ:
                del os.environ['SDL_VIDEODRIVER']
            pygame.display.quit()
            pygame.display.init()
            self.screen_width = 800
            self.screen_height = 600
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height),
                pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
            pygame.display.set_caption("3D Model Editor - OpenGL Window")
            
            # 确保OpenGL上下文有效
            glClearColor(0.2, 0.2, 0.2, 1.0)
            
            # 通知用户使用独立窗口
            messagebox.showinfo(
                "OpenGL 窗口",
                "由于技术限制，OpenGL 视图将在独立窗口中打开。\n请不要关闭该窗口，直到完成编辑。"
            )
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"创建独立OpenGL窗口失败: {str(e)}")
            return False
    
    def setup_status_bar(self):
        """设置状态栏"""
        self.status_bar = ttk.Frame(self.dialog)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="就绪")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.coords_label = ttk.Label(self.status_bar, text="")
        self.coords_label.pack(side=tk.RIGHT, padx=5)
    
    def init_opengl(self):
        """初始化OpenGL设置"""
        # 设置背景色 - 改为更亮的颜色以便于调试
        glClearColor(0.3, 0.3, 0.3, 1.0)
        
        # 启用深度测试
        glEnable(GL_DEPTH_TEST)
        
        # 设置光照 - 增强光照以便更容易看到模型
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # 增强环境光
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        
        # 设置更强的漫反射光
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        
        # 调整光源位置，使其照亮正面
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 1.0, 2.0, 0.0])
        
        # 设置渲染质量
        glEnable(GL_COLOR_MATERIAL)
        glShadeModel(GL_SMOOTH)
        
        # 禁用面剔除以确保模型可见
        glDisable(GL_CULL_FACE)  # 临时禁用，确认模型可见后可以重新启用
        
        # 重置视图
        self.reset_camera()
    
    def reset_camera(self):
        """重置相机位置和方向，调整距离以便更好地查看模型"""
        # 调整相机距离，确保能看到模型
        self.camera_distance = 3.0  # 减小距离，让模型更大
        self.camera_rotation_x = 20.0
        self.camera_rotation_y = 30.0
    
    def render_loop(self):
        """渲染循环"""
        while self.running:
            try:
                self.render_scene()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"渲染错误: {str(e)}")
                # 如果渲染出错，暂停一会再试
                time.sleep(0.5)
            # 限制帧率
            time.sleep(1/60)
    
    def render_scene(self):
        """渲染场景"""
        # 清除缓冲区
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置透视投影
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # 设置视野角度、纵横比、近裁剪面和远裁剪面
        gluPerspective(45, self.screen_width / self.screen_height, 0.1, 100.0)
        
        # 设置模型视图矩阵
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # 根据摄像机参数设置视图
        # 先退后camera_distance
        glTranslatef(0, 0, -self.camera_distance)
        # 然后围绕X和Y轴旋转
        glRotatef(self.camera_rotation_x, 1, 0, 0)
        glRotatef(self.camera_rotation_y, 0, 1, 0)
        
        # 绘制网格
        if self.show_grid_var.get():
            self.draw_grid()
        
        # 绘制坐标轴
        if self.show_axes_var.get():
            self.draw_axes()
        
        # 绘制所有模型
        self.draw_models()
        
        # 更新显示
        pygame.display.flip()
    
    def draw_grid(self):
        """绘制网格"""
        # 禁用光照以便绘制线条
        glDisable(GL_LIGHTING)
        
        # 设置线条颜色和宽度
        glColor3f(0.5, 0.5, 0.5)
        glLineWidth(1.0)
        
        # 绘制网格线
        grid_size = 10
        grid_step = 1.0
        
        glBegin(GL_LINES)
        for i in range(-grid_size, grid_size + 1):
            # 沿X轴的线
            glVertex3f(i * grid_step, 0, -grid_size * grid_step)
            glVertex3f(i * grid_step, 0, grid_size * grid_step)
            
            # 沿Z轴的线
            glVertex3f(-grid_size * grid_step, 0, i * grid_step)
            glVertex3f(grid_size * grid_step, 0, i * grid_step)
        glEnd()
        
        # 重新启用光照
        glEnable(GL_LIGHTING)
    
    def draw_axes(self):
        """绘制坐标轴"""
        # 禁用光照以便绘制线条
        glDisable(GL_LIGHTING)
        
        # 设置线条宽度
        glLineWidth(3.0)
        
        # 绘制X轴 (红色)
        glColor3f(1.0, 0.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(2.0, 0, 0)
        glEnd()
        
        # 绘制Y轴 (绿色)
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 2.0, 0)
        glEnd()
        
        # 绘制Z轴 (蓝色)
        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 2.0)
        glEnd()
        
        # 重新启用光照
        glEnable(GL_LIGHTING)
    
    def draw_models(self):
        """绘制所有模型"""
        # 设置线框模式
        if self.wireframe_var.get():
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # 绘制每个模型
        for i, model in enumerate(self.models):
            # 保存当前矩阵
            glPushMatrix()
            
            # 应用模型变换
            glTranslatef(model['position'][0], model['position'][1], model['position'][2])
            
            # 应用旋转 (OpenGL使用角度，而不是弧度)
            glRotatef(model['rotation'][0], 1, 0, 0)
            glRotatef(model['rotation'][1], 0, 1, 0)
            glRotatef(model['rotation'][2], 0, 0, 1)
            
            # 应用缩放
            glScalef(model['scale'][0], model['scale'][1], model['scale'][2])
            
            # 如果是选中的模型，绘制选择框
            if i == self.selected_model_index:
                self.draw_selection_box(model)
            
            # 应用材质属性
            material = model.get('material', {})
            diffuse_color = material.get('diffuse_color', [0.8, 0.8, 0.8])
            specular = material.get('specular', 0.5)
            
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, diffuse_color + [1.0])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [specular, specular, specular, 1.0])
            glMaterialf(GL_FRONT, GL_SHININESS, (1.0 - material.get('roughness', 0.5)) * 128)
            
            # 绘制网格数据
            self.draw_mesh(model['mesh_data'])
            
            # 恢复矩阵
            glPopMatrix()
        
        # 恢复默认渲染模式
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    
    def draw_selection_box(self, model):
        """绘制选择框"""
        # 禁用光照
        glDisable(GL_LIGHTING)
        
        # 设置线条颜色和宽度
        glColor3f(1.0, 1.0, 0.0)  # 黄色
        glLineWidth(2.0)
        
        # 获取模型的边界框
        if 'bounds' in model:
            min_bound, max_bound = model['bounds']
            
            # 绘制边界框
            glBegin(GL_LINE_LOOP)
            glVertex3f(min_bound[0], min_bound[1], min_bound[2])
            glVertex3f(max_bound[0], min_bound[1], min_bound[2])
            glVertex3f(max_bound[0], min_bound[1], max_bound[2])
            glVertex3f(min_bound[0], min_bound[1], max_bound[2])
            glEnd()
            
            glBegin(GL_LINE_LOOP)
            glVertex3f(min_bound[0], max_bound[1], min_bound[2])
            glVertex3f(max_bound[0], max_bound[1], min_bound[2])
            glVertex3f(max_bound[0], max_bound[1], max_bound[2])
            glVertex3f(min_bound[0], max_bound[1], max_bound[2])
            glEnd()
            
            glBegin(GL_LINES)
            glVertex3f(min_bound[0], min_bound[1], min_bound[2])
            glVertex3f(min_bound[0], max_bound[1], min_bound[2])
            
            glVertex3f(max_bound[0], min_bound[1], min_bound[2])
            glVertex3f(max_bound[0], max_bound[1], min_bound[2])
            
            glVertex3f(max_bound[0], min_bound[1], max_bound[2])
            glVertex3f(max_bound[0], max_bound[1], max_bound[2])
            
            glVertex3f(min_bound[0], min_bound[1], max_bound[2])
            glVertex3f(min_bound[0], max_bound[1], max_bound[2])
            glEnd()
        
        # 重新启用光照
        glEnable(GL_LIGHTING)
    
    def draw_mesh(self, mesh_data):
        """绘制网格数据"""
        # 如果没有网格数据，返回
        if not mesh_data:
            return
        
        # 获取顶点、法线和面
        vertices = mesh_data.get('vertices', [])
        normals = mesh_data.get('normals', [])
        faces = mesh_data.get('faces', [])
        
        # 打印调试信息
        print(f"绘制网格: {len(vertices)}个顶点, {len(faces)}个面")
        
        # 临时改变颜色，确保模型可见
        glColor3f(1.0, 0.0, 0.0)  # 设置为红色
        
        # 绘制每个面
        glBegin(GL_TRIANGLES)
        for face in faces:
            for i, vertex_idx in enumerate(face):
                # 如果有法线，使用法线
                if normals and len(normals) > vertex_idx:
                    normal = normals[vertex_idx]
                    glNormal3f(normal[0], normal[1], normal[2])
                
                # 使用顶点
                if len(vertices) > vertex_idx:
                    vertex = vertices[vertex_idx]
                    glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()
    
    def on_resize(self, event):
        """处理窗口大小调整事件"""
        # 确保我们获取的是 gl_frame 的宽高
        width, height = self.gl_frame.winfo_width(), self.gl_frame.winfo_height()
        if width <= 1 or height <= 1:
            return  # 忽略非常小的调整

        # 更新屏幕尺寸
        self.screen_width = width
        self.screen_height = height
        
        # 调整 OpenGL 视口
        glViewport(0, 0, width, height)
        
        # 重新创建 Pygame 窗口
        if hasattr(self, 'screen'):
            try:
                self.screen = pygame.display.set_mode((width, height), 
                                                pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
            except pygame.error:
                pass  # 如果调整失败，保留原有窗口
            
    def on_mouse_button_down(self, event):
        """处理鼠标按下事件"""
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.dragging = True
    
    def on_mouse_button_up(self, event):
        """处理鼠标释放事件"""
        self.dragging = False
    
    def on_mouse_motion(self, event):
        """处理鼠标移动事件"""
        if self.dragging:
            # 计算移动差异
            dx = event.x - self.mouse_x
            dy = event.y - self.mouse_y
            
            # 更新鼠标位置
            self.mouse_x = event.x
            self.mouse_y = event.y
            
            # 使用左键拖动时
            if event.state & 0x100:  # 检查是否按下鼠标左键
                # 根据编辑模式处理拖动
                if self.edit_mode == "move" and self.selected_model_index >= 0:
                    self.move_selected_model(dx, dy)
                elif self.edit_mode == "rotate" and self.selected_model_index >= 0:
                    self.rotate_selected_model(dx, dy)
                elif self.edit_mode == "scale" and self.selected_model_index >= 0:
                    self.scale_selected_model(dx, dy)
                else:
                    # 默认行为：旋转视图
                    self.camera_rotation_y += dx * 0.5
                    self.camera_rotation_x += dy * 0.5
                    
                    # 限制X轴旋转角度
                    self.camera_rotation_x = max(-90, min(90, self.camera_rotation_x))
    
    def on_mouse_wheel(self, event):
        """处理鼠标滚轮事件"""
        # 获取滚轮移动方向 (Windows)
        delta = event.delta
        
        # 根据事件类型确定用于缩放的系数
        if delta < 0:
            self.camera_distance += 0.5
        else:
            self.camera_distance -= 0.5
        
        # 限制缩放范围
        self.camera_distance = max(2.0, min(50.0, self.camera_distance))
    
    def on_model_select(self, event):
        """处理模型选择事件"""
        selection = self.models_listbox.curselection()
        if selection:
            idx = selection[0]
            if 0 <= idx < len(self.models):
                self.selected_model_index = idx
                self.selected_model = self.models[idx]
                self.update_property_editors()
                self.status_label.config(text=f"已选中模型: {self.models[idx].get('name', '未命名')}")
            else:
                self.selected_model_index = -1
                self.selected_model = None
                self.status_label.config(text="未选中模型")
                
        # 同步到GL视图（更新选中模型高亮）
        self.update_gl_models()
    
    def update_property_editors(self):
        """更新属性编辑器的值"""
        if self.selected_model_index >= 0 and self.selected_model:
            # 确保选中的模型数据格式正确
            self._normalize_model_data(self.selected_model)
            
            # 更新UI控件显示值
            self.position_x_var.set(self.selected_model['position'][0])
            self.position_y_var.set(self.selected_model['position'][1])
            self.position_z_var.set(self.selected_model['position'][2])
            
            self.rotation_x_var.set(self.selected_model['rotation'][0])
            self.rotation_y_var.set(self.selected_model['rotation'][1])
            self.rotation_z_var.set(self.selected_model['rotation'][2])
            
            self.scale_x_var.set(self.selected_model['scale'][0])
            self.scale_y_var.set(self.selected_model['scale'][1])
            self.scale_z_var.set(self.selected_model['scale'][2])
            
            # 更新材质
            material = self.selected_model.get('material', {})
            self.material_name_var.set(material.get('name', '默认材质'))
            
            # 设置颜色按钮背景
            diffuse_color = material.get('diffuse_color', [0.8, 0.8, 0.8])
            # 确保diffuse_color是列表且有三个元素
            if not isinstance(diffuse_color, list):
                diffuse_color = [0.8, 0.8, 0.8]
            elif len(diffuse_color) < 3:
                diffuse_color = diffuse_color + [0.8] * (3 - len(diffuse_color))
                
            hex_color = self.rgb_to_hex(diffuse_color)
            try:
                # 对于tk.Button可以直接设置background
                self.diffuse_color_button.configure(background=hex_color)
            except tk.TclError:
                # 如果颜色格式错误，使用默认颜色
                self.diffuse_color_button.configure(background='#cccccc')
            
            # 更新材质参数
            self.specular_var.set(material.get('specular', 0.5))
            self.roughness_var.set(material.get('roughness', 0.5))
            self.metallic_var.set(material.get('metallic', 0.0))
            self.transparency_var.set(material.get('transparency', 0.0))
            
            # 更新纹理路径
            self.diffuse_map_var.set(material.get('diffuse', ''))
            self.normal_map_var.set(material.get('normal', ''))
            self.roughness_map_var.set(material.get('roughness_map', ''))
        
    def rgb_to_hex(self, rgb):
        """将RGB颜色值转换为十六进制颜色代码"""
        try:
            r, g, b = rgb
            # 确保值在0-1之间
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        except (ValueError, TypeError):
            # 如果转换失败，返回默认灰色
            return '#cccccc'
    
    def hex_to_rgb(self, hex_color):
        """将十六进制颜色代码转换为RGB颜色值"""
        hex_color = hex_color.lstrip('#')
        return [int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)]
    
    def update_model_position(self):
        """更新选中模型的位置"""
        if self.selected_model_index >= 0:
            self.models[self.selected_model_index]['position'] = [
                self.position_x_var.get(),
                self.position_y_var.get(),
                self.position_z_var.get()
            ]
            
        # 同步到GL视图
        self.update_gl_models()
    
    def update_model_rotation(self):
        """更新选中模型的旋转"""
        if self.selected_model_index >= 0:
            self.models[self.selected_model_index]['rotation'] = [
                self.rotation_x_var.get(),
                self.rotation_y_var.get(),
                self.rotation_z_var.get()
            ]

        # 同步到GL视图
        self.update_gl_models()
    
    def update_model_scale(self, axis=None):
        """更新选中模型的缩放"""
        if self.selected_model_index >= 0:
            if self.uniform_scale_var.get() and axis:
                # 统一缩放 - 根据被修改的轴来设置所有轴
                value = getattr(self, f'scale_{axis}_var').get()
                self.scale_x_var.set(value)
                self.scale_y_var.set(value)
                self.scale_z_var.set(value)
            
            self.models[self.selected_model_index]['scale'] = [
                self.scale_x_var.get(),
                self.scale_y_var.get(),
                self.scale_z_var.get()
            ]

        # 同步到GL视图
        self.update_gl_models()
    
    def update_material_properties(self):
        """更新选中模型的材质属性"""
        if self.selected_model_index >= 0:
            material = self.models[self.selected_model_index].get('material', {})
            
            material['name'] = self.material_name_var.get()
            material['specular'] = self.specular_var.get()
            material['roughness'] = self.roughness_var.get()
            material['metallic'] = self.metallic_var.get()
            material['transparency'] = self.transparency_var.get()
            
            # 更新纹理路径
            material['diffuse'] = self.diffuse_map_var.get()
            material['normal'] = self.normal_map_var.get()
            material['roughness_map'] = self.roughness_map_var.get()
            
            self.models[self.selected_model_index]['material'] = material

        # 同步到GL视图
        self.update_gl_models()
    
    def set_edit_mode(self, mode):
        """设置编辑模式"""
        self.edit_mode = mode
        self.status_label.config(text=f"编辑模式: {mode}")
    
    def set_edit_axis(self, axis):
        """设置编辑轴向"""
        self.edit_axis = axis
        self.status_label.config(text=f"编辑轴向: {axis}")
    
    def move_selected_model(self, dx, dy):
        """移动选中的模型"""
        if self.selected_model_index < 0:
            return
        
        # 映射到3D空间
        # 这是一个简化的实现，实际上应该考虑摄像机方向和视图矩阵
        speed = 0.02
        
        if 'x' in self.edit_axis:
            self.models[self.selected_model_index]['position'][0] += dx * speed
        if 'y' in self.edit_axis:
            self.models[self.selected_model_index]['position'][1] -= dy * speed
        
        # 更新属性编辑器
        self.update_property_editors()
        
        # 同步到GL视图
        self.update_gl_models()
        
    def rotate_selected_model(self, dx, dy):
        """旋转选中的模型"""
        if self.selected_model_index < 0:
            return
        
        speed = 0.5
        
        if 'x' in self.edit_axis:
            self.models[self.selected_model_index]['rotation'][0] += dy * speed
        if 'y' in self.edit_axis:
            self.models[self.selected_model_index]['rotation'][1] += dx * speed
        
        # 更新属性编辑器
        self.update_property_editors()
        
        # 同步到GL视图
        self.update_gl_models()
    
    def scale_selected_model(self, dx, dy):
        """缩放选中的模型"""
        if self.selected_model_index < 0:
            return
        
        speed = 0.01
        scale_factor = 1.0 + (dx + dy) * speed
        
        if self.uniform_scale_var.get():
            # 统一缩放
            self.models[self.selected_model_index]['scale'] = [
                s * scale_factor for s in self.models[self.selected_model_index]['scale']
            ]
        else:
            # 轴向缩放
            if 'x' in self.edit_axis:
                self.models[self.selected_model_index]['scale'][0] *= scale_factor
            if 'y' in self.edit_axis:
                self.models[self.selected_model_index]['scale'][1] *= scale_factor
            if 'z' in self.edit_axis:
                self.models[self.selected_model_index]['scale'][2] *= scale_factor
        
        # 更新属性编辑器
        self.update_property_editors()
        
        # 同步到GL视图
        self.update_gl_models()
    
    def mirror_model(self, axis):
        """镜像选中的模型"""
        if self.selected_model_index < 0:
            return
        
        axis_index = {'x': 0, 'y': 1, 'z': 2}[axis]
        
        # 镜像缩放系数
        self.models[self.selected_model_index]['scale'][axis_index] *= -1
        
        # 更新属性编辑器
        self.update_property_editors()
        
        # 同步到GL视图
        self.update_gl_models()
    
    def reset_position(self):
        """重置选中模型的位置"""
        if self.selected_model_index < 0:
            return
        
        self.models[self.selected_model_index]['position'] = [0.0, 0.0, 0.0]
        
        # 更新属性编辑器
        self.update_property_editors()
    
    def reset_rotation(self):
        """重置选中模型的旋转"""
        if self.selected_model_index < 0:
            return
        
        self.models[self.selected_model_index]['rotation'] = [0.0, 0.0, 0.0]
        
        # 更新属性编辑器
        self.update_property_editors()
        
        # 同步到GL视图
        self.update_gl_models()
    
    def reset_scale(self):
        """重置选中模型的缩放"""
        if self.selected_model_index < 0:
            return
        
        self.models[self.selected_model_index]['scale'] = [1.0, 1.0, 1.0]
        
        # 更新属性编辑器
        self.update_property_editors()
        
        # 同步到GL视图
        self.update_gl_models()
    
    def add_model_dialog(self):
        """显示添加模型对话框"""
        dialog = tk.Toplevel(self.dialog)
        dialog.title("添加模型")
        dialog.geometry("300x300")
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        # 模型类型
        type_frame = ttk.Frame(dialog)
        type_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(type_frame, text="模型类型:").pack(side=tk.LEFT)
        
        model_type_var = tk.StringVar(value="cube")
        model_types = ["cube", "sphere", "cylinder", "cone", "plane"]
        type_combo = ttk.Combobox(type_frame, textvariable=model_type_var, 
                                 values=model_types, state="readonly", width=15)
        type_combo.pack(side=tk.LEFT, padx=5)
        
        # 模型名称
        name_frame = ttk.Frame(dialog)
        name_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(name_frame, text="名称:").pack(side=tk.LEFT)
        
        name_var = tk.StringVar(value="新模型")
        name_entry = ttk.Entry(name_frame, textvariable=name_var, width=20)
        name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 模型尺寸
        size_frame = ttk.LabelFrame(dialog, text="尺寸")
        size_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # X尺寸
        x_size_frame = ttk.Frame(size_frame)
        # 继续实现模型尺寸设置部分
        x_size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(x_size_frame, text="X尺寸:").pack(side=tk.LEFT)
        x_size_var = tk.DoubleVar(value=1.0)
        x_size_entry = ttk.Spinbox(x_size_frame, from_=0.1, to=10, increment=0.1, textvariable=x_size_var, width=10)
        x_size_entry.pack(side=tk.LEFT, padx=5)
        
        # Y尺寸
        y_size_frame = ttk.Frame(size_frame)
        y_size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(y_size_frame, text="Y尺寸:").pack(side=tk.LEFT)
        y_size_var = tk.DoubleVar(value=1.0)
        y_size_entry = ttk.Spinbox(y_size_frame, from_=0.1, to=10, increment=0.1, textvariable=y_size_var, width=10)
        y_size_entry.pack(side=tk.LEFT, padx=5)
        
        # Z尺寸
        z_size_frame = ttk.Frame(size_frame)
        z_size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(z_size_frame, text="Z尺寸:").pack(side=tk.LEFT)
        z_size_var = tk.DoubleVar(value=1.0)
        z_size_entry = ttk.Spinbox(z_size_frame, from_=0.1, to=10, increment=0.1, textvariable=z_size_var, width=10)
        z_size_entry.pack(side=tk.LEFT, padx=5)
        
        # 按钮框架
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="添加", command=lambda: self.add_model(
            model_type_var.get(), 
            name_var.get(), 
            [x_size_var.get(), y_size_var.get(), z_size_var.get()]
        )).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def add_model(self, model_type, name, size):
        """添加新模型"""
        # 根据模型类型创建网格数据
        mesh_data = self.create_primitive_mesh(model_type, size)
        
        # 创建模型时，将其放置在视野中心位置
        model = {
            'name': name,
            'type': model_type,
            'position': [0.0, 0.0, 0.0],  # 保持在原点
            'rotation': [0.0, 0.0, 0.0],
            'scale': [1.0, 1.0, 1.0],
            'mesh_data': mesh_data,
            'material': {
                'name': '默认材质',
                'diffuse_color': [0.8, 0.8, 0.8],
                'specular': 0.5,
                'roughness': 0.5,
                'metallic': 0.0,
                'transparency': 0.0
            },
            'bounds': self.calculate_mesh_bounds(mesh_data)
        }
        
        # 更新模型列表显示
        self.update_models_listbox()
        
        # 添加到模型列表
        self.models.append(model)
        
        # 自动选择新添加的模型
        self.selected_model_index = len(self.models) - 1
        self.selected_model = model
        self.models_listbox.selection_clear(0, tk.END)
        self.models_listbox.selection_set(self.selected_model_index)
        self.update_property_editors()
        
        # 添加模型后重置相机视角，确保可以看到模型
        self.reset_camera()
        
        # 设置为透视视图以便更好地查看模型
        self.set_camera_view("perspective")
        
        # 添加调试信息
        print(f"添加模型: {name}, 类型: {model_type}, 位置: {model['position']}")
        print(f"模型列表长度: {len(self.models)}")
        print(f"边界框: {model['bounds']}")
        
        # 更新状态栏
        self.status_label.config(text=f"已添加模型: {name}")
        
        # 同步到GL视图
        self.update_gl_models()
    
    def create_primitive_mesh(self, primitive_type, size):
        """创建基本几何体的网格数据"""
        if primitive_type == "cube":
            return self.create_cube_mesh(size)
        elif primitive_type == "sphere":
            return self.create_sphere_mesh(size)
        elif primitive_type == "cylinder":
            return self.create_cylinder_mesh(size)
        elif primitive_type == "cone":
            return self.create_cone_mesh(size)
        elif primitive_type == "plane":
            return self.create_plane_mesh(size)
        else:
            return None
    
    def create_cube_mesh(self, size):
        """创建立方体网格"""
        x_size, y_size, z_size = size
        half_x = x_size / 2
        half_y = y_size / 2
        half_z = z_size / 2
        
        # 定义顶点
        vertices = [
            # 前面
            [-half_x, -half_y, half_z],
            [half_x, -half_y, half_z],
            [half_x, half_y, half_z],
            [-half_x, half_y, half_z],
            # 后面
            [-half_x, -half_y, -half_z],
            [half_x, -half_y, -half_z],
            [half_x, half_y, -half_z],
            [-half_x, half_y, -half_z]
        ]
        
        # 定义法线
        normals = [
            [0, 0, 1],  # 前
            [0, 0, -1], # 后
            [1, 0, 0],  # 右
            [-1, 0, 0], # 左
            [0, 1, 0],  # 上
            [0, -1, 0]  # 下
        ]
        
        # 定义面 (三角形)
        faces = [
            # 前面
            [0, 1, 2], [0, 2, 3],
            # 后面
            [4, 6, 5], [4, 7, 6],
            # 右面
            [1, 5, 6], [1, 6, 2],
            # 左面
            [4, 0, 3], [4, 3, 7],
            # 上面
            [3, 2, 6], [3, 6, 7],
            # 下面
            [0, 4, 5], [0, 5, 1]
        ]
        
        return {
            'vertices': vertices,
            'normals': [normals[i//2] for i in range(12)],
            'faces': faces
        }
    
    def create_sphere_mesh(self, size):
        """创建球体网格"""
        radius = min(size) / 2
        segments = 16
        rings = 16
        
        vertices = []
        normals = []
        faces = []
        
        # 生成顶点和法线
        for i in range(rings + 1):
            v = i / rings
            phi = v * math.pi
            
            for j in range(segments):
                u = j / segments
                theta = u * 2 * math.pi
                
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                
                vertices.append([x, y, z])
                normals.append([x/radius, y/radius, z/radius])
        
        # 生成面
        for i in range(rings):
            for j in range(segments):
                next_j = (j + 1) % segments
                
                v1 = i * segments + j
                v2 = i * segments + next_j
                v3 = (i + 1) * segments + next_j
                v4 = (i + 1) * segments + j
                
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])
        
        return {
            'vertices': vertices,
            'normals': normals,
            'faces': faces
        }
    
    def create_cylinder_mesh(self, size):
        """创建圆柱体网格"""
        radius = min(size[0], size[2]) / 2
        height = size[1]
        segments = 16  # 圆周分段数
        
        vertices = []
        normals = []
        faces = []
        
        # 生成顶部和底部圆形的顶点
        vertices.append([0, height/2, 0])  # 顶部中心点
        vertices.append([0, -height/2, 0])  # 底部中心点
        
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            # 顶部圆周点
            vertices.append([x, height/2, z])
            normals.append([0, 1, 0])
            
            # 底部圆周点
            vertices.append([x, -height/2, z])
            normals.append([0, -1, 0])
            
            # 侧面法线
            normals.append([x/radius, 0, z/radius])
        
        # 生成顶部面
        for i in range(segments):
            v1 = 0  # 顶部中心点
            v2 = 2 + (i * 2)
            v3 = 2 + (((i + 1) % segments) * 2)
            faces.append([v1, v2, v3])
        
        # 生成底部面
        for i in range(segments):
            v1 = 1  # 底部中心点
            v2 = 3 + (((i + 1) % segments) * 2)
            v3 = 3 + (i * 2)
            faces.append([v1, v2, v3])
        
        # 生成侧面
        for i in range(segments):
            v1 = 2 + (i * 2)
            v2 = 3 + (i * 2)
            v3 = 3 + (((i + 1) % segments) * 2)
            v4 = 2 + (((i + 1) % segments) * 2)
            
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
        
        return {
            'vertices': vertices,
            'normals': normals,
            'faces': faces
        }

    def create_cone_mesh(self, size):
        """创建圆锥体网格"""
        radius = min(size[0], size[2]) / 2
        height = size[1]
        segments = 16  # 圆周分段数
        
        vertices = []
        normals = []
        faces = []
        
        # 顶点和底部中心点
        vertices.append([0, height/2, 0])  # 顶点
        vertices.append([0, -height/2, 0])  # 底部中心点
        
        # 计算侧面法线的角度
        side_normal_angle = math.atan2(height, radius)
        side_normal_y = math.sin(side_normal_angle)
        side_normal_rad = math.cos(side_normal_angle)
        
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            
            # 底部圆周点
            vertices.append([x, -height/2, z])
            
            # 底部法线
            normals.append([0, -1, 0])
            
            # 侧面法线
            nx = side_normal_rad * math.cos(angle)
            nz = side_normal_rad * math.sin(angle)
            normals.append([nx, side_normal_y, nz])
        
        # 生成底部面
        for i in range(segments):
            v1 = 1  # 底部中心点
            v2 = 2 + ((i + 1) % segments)
            v3 = 2 + i
            faces.append([v1, v2, v3])
        
        # 生成侧面
        for i in range(segments):
            v1 = 0  # 顶点
            v2 = 2 + i
            v3 = 2 + ((i + 1) % segments)
            faces.append([v1, v2, v3])
        
        return {
            'vertices': vertices,
            'normals': normals,
            'faces': faces
        }

    def create_plane_mesh(self, size):
        """创建平面网格"""
        width = size[0]
        depth = size[2]
        half_width = width / 2
        half_depth = depth / 2
        
        # 定义顶点
        vertices = [
            [-half_width, 0, -half_depth],
            [half_width, 0, -half_depth],
            [half_width, 0, half_depth],
            [-half_width, 0, half_depth]
        ]
        
        # 定义法线
        normals = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ]
        
        # 定义面 (两个三角形组成一个平面)
        faces = [
            [0, 1, 2],
            [0, 2, 3]
        ]
        
        return {
            'vertices': vertices,
            'normals': normals,
            'faces': faces
        }

    def calculate_mesh_bounds(self, mesh_data):
        """计算网格的边界框"""
        if not mesh_data or not mesh_data.get('vertices'):
            return [[-1, -1, -1], [1, 1, 1]]  # 默认边界
        
        vertices = mesh_data['vertices']
        if not vertices:
            return [[-1, -1, -1], [1, 1, 1]]
        
        # 初始化最小和最大坐标为第一个顶点的坐标
        min_x = max_x = vertices[0][0]
        min_y = max_y = vertices[0][1]
        min_z = max_z = vertices[0][2]
        
        # 遍历所有顶点，更新最小和最大坐标
        for vertex in vertices:
            min_x = min(min_x, vertex[0])
            max_x = max(max_x, vertex[0])
            min_y = min(min_y, vertex[1])
            max_y = max(max_y, vertex[1])
            min_z = min(min_z, vertex[2])
            max_z = max(max_z, vertex[2])
        
        return [[min_x, min_y, min_z], [max_x, max_y, max_z]]

    def update_models_listbox(self):
        """更新模型列表显示"""
        self.models_listbox.delete(0, tk.END)
        for model in self.models:
            self.models_listbox.insert(tk.END, model.get('name', '未命名'))

    def delete_selected_model(self):
        """删除选中的模型"""
        if self.selected_model_index < 0 or self.selected_model_index >= len(self.models):
            messagebox.showinfo("提示", "请先选择一个模型")
            return
        
        # 确认删除
        result = messagebox.askyesno("确认", f"确定要删除模型 '{self.models[self.selected_model_index].get('name', '未命名')}' 吗？")
        if not result:
            return
        
        # 删除模型
        self.models.pop(self.selected_model_index)
        
        # 更新列表显示
        self.update_models_listbox()
        
        # 重置选择
        self.selected_model_index = -1
        self.selected_model = None
        
        # 更新状态栏
        self.status_label.config(text="已删除模型")
        
        # 同步到GL视图
        self.update_gl_models()

    def duplicate_selected_model(self):
        """复制选中的模型"""
        if self.selected_model_index < 0 or self.selected_model_index >= len(self.models):
            messagebox.showinfo("提示", "请先选择一个模型")
            return
        
        # 复制模型数据
        original_model = self.models[self.selected_model_index]
        new_model = original_model.copy()
        
        # 修改名称和稍微偏移位置以区分
        new_model['name'] = f"{original_model.get('name', '未命名')}_复制"
        new_model['position'] = [
            original_model['position'][0] + 0.5,
            original_model['position'][1],
            original_model['position'][2] + 0.5
        ]
        
        # 添加到模型列表
        self.models.append(new_model)
        
        # 更新列表显示
        self.update_models_listbox()
        
        # 选择新复制的模型
        self.selected_model_index = len(self.models) - 1
        self.selected_model = new_model
        self.models_listbox.selection_clear(0, tk.END)
        self.models_listbox.selection_set(self.selected_model_index)
        
        # 更新属性编辑器
        self.update_property_editors()
        
        # 更新状态栏
        self.status_label.config(text=f"已复制模型: {new_model['name']}")
        
        # 同步到GL视图
        self.update_gl_models()

    def import_obj_file(self):
        """导入OBJ模型文件"""
        # 打开文件对话框选择OBJ文件
        filename = filedialog.askopenfilename(
            title="选择OBJ文件",
            filetypes=[("OBJ文件", "*.obj"), ("所有文件", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # 使用trimesh加载OBJ文件
            mesh = trimesh.load(filename)
            
            # 提取文件名作为模型名称
            base_name = os.path.basename(filename)
            name = os.path.splitext(base_name)[0]
            
            # 提取网格数据
            vertices = mesh.vertices.tolist()
            faces = mesh.faces.tolist()
            normals = mesh.vertex_normals.tolist() if hasattr(mesh, 'vertex_normals') else []
            
            # 创建模型
            model = {
                'name': name,
                'type': 'imported',
                'position': [0.0, 0.0, 0.0],
                'rotation': [0.0, 0.0, 0.0],
                'scale': [1.0, 1.0, 1.0],
                'mesh_data': {
                    'vertices': vertices,
                    'normals': normals,
                    'faces': faces
                },
                'material': {
                    'name': '默认材质',
                    'diffuse_color': [0.8, 0.8, 0.8],
                    'specular': 0.5,
                    'roughness': 0.5,
                    'metallic': 0.0,
                    'transparency': 0.0
                }
            }
            
            # 计算边界框
            model['bounds'] = self.calculate_mesh_bounds(model['mesh_data'])
            
            # 添加到模型列表
            self.models.append(model)
            
            # 更新列表显示
            self.update_models_listbox()
            
            # 选择新导入的模型
            self.selected_model_index = len(self.models) - 1
            self.selected_model = model
            self.models_listbox.selection_clear(0, tk.END)
            self.models_listbox.selection_set(self.selected_model_index)
            
            # 更新属性编辑器
            self.update_property_editors()
            
            # 更新状态栏
            self.status_label.config(text=f"已导入模型: {name}")
            
            # 同步到GL视图
            self.update_gl_models()
            
        except Exception as e:
            messagebox.showerror("导入失败", f"导入OBJ文件失败: {str(e)}")

    def import_from_library(self):
        """从模型库导入模型"""
        # 如果模型库为空
        if not self.model_library:
            messagebox.showinfo("提示", "模型库为空")
            return
        
        # 创建选择对话框
        dialog = tk.Toplevel(self.dialog)
        dialog.title("从模型库导入")
        dialog.geometry("400x400")
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        # 创建模型列表
        ttk.Label(dialog, text="选择要导入的模型:").pack(padx=10, pady=5, anchor=tk.W)
        
        model_listbox = tk.Listbox(dialog, height=15)
        model_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(model_listbox, orient="vertical")
        scrollbar.config(command=model_listbox.yview)
        model_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充模型列表
        for name in self.model_library.keys():
            model_listbox.insert(tk.END, name)
        
        # 添加按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def on_import():
            selection = model_listbox.curselection()
            if not selection:
                messagebox.showinfo("提示", "请选择一个模型")
                return
            
            model_name = model_listbox.get(selection[0])
            model_data = self.model_library[model_name].copy()
            
            # 确保模型数据格式正确
            self._normalize_model_data(model_data)
            
            # 设置名称和初始位置
            model_data['name'] = f"{model_name}_导入"
            model_data['position'] = [0.0, 0.0, 0.0]
            
            # 添加到模型列表
            self.models.append(model_data)
            self.update_models_listbox()
            self.selected_model_index = len(self.models) - 1
            self.selected_model = model_data
            self.models_listbox.selection_clear(0, tk.END)
            self.models_listbox.selection_set(self.selected_model_index)
            self.update_property_editors()
            
            dialog.destroy()
            self.status_label.config(text=f"已从库导入模型: {model_name}")
            
            # 同步到GL视图
            self.update_gl_models()
        
        ttk.Button(button_frame, text="导入", command=on_import).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def export_all_models(self):
        """导出所有模型"""
        if not self.models:
            messagebox.showinfo("提示", "没有模型可导出")
            return
        
        # 选择导出目录
        directory = filedialog.askdirectory(title="选择导出目录")
        if not directory:
            return
        
        try:
            for i, model in enumerate(self.models):
                model_name = model.get('name', f'model_{i}')
                # 导出为JSON格式
                with open(os.path.join(directory, f"{model_name}.json"), 'w') as f:
                    json.dump(model, f, indent=2)
            
            messagebox.showinfo("导出成功", f"成功导出 {len(self.models)} 个模型到 {directory}")
        except Exception as e:
            messagebox.showerror("导出失败", f"导出模型失败: {str(e)}")

    def export_selected_model(self):
        """导出选中的模型"""
        if self.selected_model_index < 0 or self.selected_model_index >= len(self.models):
            messagebox.showinfo("提示", "请先选择一个模型")
            return
        
        # 获取选中的模型
        model = self.models[self.selected_model_index]
        model_name = model.get('name', 'model')
        
        # 选择保存文件
        filename = filedialog.asksaveasfilename(
            title="保存模型",
            initialfile=f"{model_name}.json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # 确保文件有扩展名
            if not filename.endswith('.json'):
                filename += '.json'
            
            # 导出为JSON格式
            with open(filename, 'w') as f:
                json.dump(model, f, indent=2)
            
            messagebox.showinfo("导出成功", f"成功导出模型到 {filename}")
        except Exception as e:
            messagebox.showerror("导出失败", f"导出模型失败: {str(e)}")

    def pick_diffuse_color(self):
        """选择漫反射颜色"""
        # 获取当前颜色
        current_color = [0.8, 0.8, 0.8]
        if self.selected_model_index >= 0:
            material = self.models[self.selected_model_index].get('material', {})
            current_color = material.get('diffuse_color', current_color)
        
        # 转换为hex颜色
        hex_color = self.rgb_to_hex(current_color)
        
        # 打开颜色选择器
        from tkinter import colorchooser
        color = colorchooser.askcolor(hex_color)
        
        if color and color[0]:  # color是一个元组 ((r,g,b), hex_string)
            try:
                # 更新按钮颜色
                self.diffuse_color_button.configure(background=color[1])
                
                # 更新模型材质
                if self.selected_model_index >= 0:
                    material = self.models[self.selected_model_index].get('material', {})
                    material['diffuse_color'] = [c/255 for c in color[0]]  # 转换为0-1范围
                    self.models[self.selected_model_index]['material'] = material
            except tk.TclError as e:
                if self.logger:
                    self.logger.error(f"设置颜色失败: {str(e)}")

    def browse_texture_file(self, texture_type):
        """浏览纹理文件"""
        filename = filedialog.askopenfilename(
            title=f"选择{texture_type}纹理文件",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tga"),
                ("所有文件", "*.*")
            ]
        )
        
        if not filename:
            return
        
        # 根据纹理类型更新对应的变量
        if texture_type == "diffuse":
            self.diffuse_map_var.set(filename)
        elif texture_type == "normal":
            self.normal_map_var.set(filename)
        elif texture_type == "roughness":
            self.roughness_map_var.set(filename)
        
        # 如果有选中的模型，更新其材质
        if self.selected_model_index >= 0:
            material = self.models[self.selected_model_index].get('material', {})
            material[texture_type] = filename
            self.models[self.selected_model_index]['material'] = material
            self.update_material_properties()

    def load_from_material_library(self):
        """从材质库加载材质"""
        # 如果材质库为空
        if not self.material_library:
            messagebox.showinfo("提示", "材质库为空")
            return
        
        # 创建选择对话框
        dialog = tk.Toplevel(self.dialog)
        dialog.title("从材质库加载")
        dialog.geometry("400x400")
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        # 创建材质列表
        ttk.Label(dialog, text="选择要加载的材质:").pack(padx=10, pady=5, anchor=tk.W)
        
        material_listbox = tk.Listbox(dialog, height=15)
        material_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(material_listbox, orient="vertical")
        scrollbar.config(command=material_listbox.yview)
        material_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充材质列表
        for name in self.material_library.keys():
            material_listbox.insert(tk.END, name)
        
        # 添加按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def on_load():
            selection = material_listbox.curselection()
            if not selection:
                messagebox.showinfo("提示", "请选择一个材质")
                return
            
            material_name = material_listbox.get(selection[0])
            material_data = self.material_library[material_name]
            
            # 复制材质数据到当前编辑器
            self.material_name_var.set(material_name)
            
            if 'diffuse_color' in material_data:
                self.diffuse_color_button.configure(background=self.rgb_to_hex(material_data['diffuse_color']))
            
            if 'specular' in material_data:
                self.specular_var.set(material_data['specular'])
            
            if 'roughness' in material_data:
                self.roughness_var.set(material_data['roughness'])
            
            if 'metallic' in material_data:
                self.metallic_var.set(material_data['metallic'])
            
            if 'transparency' in material_data:
                self.transparency_var.set(material_data['transparency'])
            
            if 'diffuse' in material_data:
                self.diffuse_map_var.set(material_data['diffuse'])
            
            if 'normal' in material_data:
                self.normal_map_var.set(material_data['normal'])
            
            if 'roughness_map' in material_data:
                self.roughness_map_var.set(material_data['roughness_map'])
            
            dialog.destroy()
            self.status_label.config(text=f"已加载材质: {material_name}")
        
        ttk.Button(button_frame, text="加载", command=on_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def save_to_material_library(self):
        """保存当前材质到材质库"""
        # 创建保存对话框
        dialog = tk.Toplevel(self.dialog)
        dialog.title("保存到材质库")
        dialog.geometry("300x120")
        dialog.transient(self.dialog)
        dialog.grab_set()
        
        # 创建名称输入框
        ttk.Label(dialog, text="材质名称:").pack(padx=10, pady=5, anchor=tk.W)
        
        name_var = tk.StringVar(value=self.material_name_var.get())
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.pack(fill=tk.X, padx=10, pady=5)
        name_entry.select_range(0, tk.END)
        name_entry.focus()
        
        # 添加按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def on_save():
            material_name = name_var.get().strip()
            if not material_name:
                messagebox.showinfo("提示", "请输入材质名称")
                return
            
            # 收集当前材质属性
            material = {
                'name': material_name,
                'diffuse_color': self.hex_to_rgb(self.diffuse_color_button.cget('background')),
                'specular': self.specular_var.get(),
                'roughness': self.roughness_var.get(),
                'metallic': self.metallic_var.get(),
                'transparency': self.transparency_var.get(),
                'diffuse': self.diffuse_map_var.get(),
                'normal': self.normal_map_var.get(),
                'roughness_map': self.roughness_map_var.get()
            }
            
            # 检查是否覆盖已有材质
            if material_name in self.material_library:
                result = messagebox.askyesno("确认", f"材质 '{material_name}' 已存在，是否覆盖？")
                if not result:
                    return
            
            # 保存到材质库
            self.material_library[material_name] = material
            
            dialog.destroy()
            self.status_label.config(text=f"已保存材质: {material_name}")
        
        ttk.Button(button_frame, text="保存", command=on_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def load_models_from_map_data(self):
        """从地图数据加载模型"""
        if not self.map_data:
            return
        
        # 初始化模型数据列表
        models_data = []
        
        try:
            # 根据 map_data 类型获取模型数据
            if isinstance(self.map_data, dict):
                models_data = self.map_data.get('models', [])
            elif hasattr(self.map_data, 'models'):
                # 直接获取模型列表属性
                if hasattr(self.map_data.models, '__iter__') and not isinstance(self.map_data.models, str):
                    models_data = list(self.map_data.models)
                else:
                    self.status_label.config(text="地图数据中的模型格式不正确")
                    return
        except Exception as e:
            if self.logger:
                self.logger.error(f"获取模型数据失败: {str(e)}")
            self.status_label.config(text=f"加载模型失败: {str(e)}")
            return
        
        if not models_data:
            return
        
        # 处理每个模型数据，确保格式正确
        for model_data in models_data:
            if not isinstance(model_data, dict):
                continue
                
            # 确保有必要的属性
            if 'mesh_data' not in model_data:
                continue
            
            # 处理和标准化模型属性
            self._normalize_model_data(model_data)
            
            # 添加到模型列表
            self.models.append(model_data)
        
        # 更新UI
        self.update_models_listbox()
        
        # 选择第一个模型
        if self.models:
            self.selected_model_index = 0
            self.selected_model = self.models[0]
            self.models_listbox.selection_set(0)
            self.update_property_editors()
            
            self.status_label.config(text=f"已从地图加载 {len(self.models)} 个模型")

    def _normalize_model_data(self, model_data):
        """确保模型数据的格式正确"""
        # 标准化位置
        if 'position' not in model_data:
            model_data['position'] = [0.0, 0.0, 0.0]
        elif not isinstance(model_data['position'], list):
            try:
                # 尝试转换为浮点数
                value = float(model_data['position'])
                model_data['position'] = [value, 0.0, 0.0]
            except (ValueError, TypeError):
                model_data['position'] = [0.0, 0.0, 0.0]
        
        # 确保位置列表有三个元素
        if len(model_data['position']) < 3:
            model_data['position'] = model_data['position'] + [0.0] * (3 - len(model_data['position']))
        
        # 标准化旋转
        if 'rotation' not in model_data:
            model_data['rotation'] = [0.0, 0.0, 0.0]
        elif not isinstance(model_data['rotation'], list):
            try:
                value = float(model_data['rotation'])
                model_data['rotation'] = [value, 0.0, 0.0]
            except (ValueError, TypeError):
                model_data['rotation'] = [0.0, 0.0, 0.0]
        
        # 确保旋转列表有三个元素
        if len(model_data['rotation']) < 3:
            model_data['rotation'] = model_data['rotation'] + [0.0] * (3 - len(model_data['rotation']))
        
        # 标准化缩放
        if 'scale' not in model_data:
            model_data['scale'] = [1.0, 1.0, 1.0]
        elif not isinstance(model_data['scale'], list):
            try:
                value = float(model_data['scale'])
                model_data['scale'] = [value, value, value]
            except (ValueError, TypeError):
                model_data['scale'] = [1.0, 1.0, 1.0]
        
        # 确保缩放列表有三个元素
        if len(model_data['scale']) < 3:
            # 如果只有一个值，应用于所有维度
            if len(model_data['scale']) == 1:
                value = model_data['scale'][0]
                model_data['scale'] = [value, value, value]
            else:
                model_data['scale'] = model_data['scale'] + [1.0] * (3 - len(model_data['scale']))
        
    def on_close(self):
        """关闭编辑器窗口"""
        # 询问是否保存更改
        if self.models:
            result = messagebox.askyesnocancel("保存更改", "是否将模型保存到地图？")
            
            if result is None:  # 取消
                return
            
            if result:  # 保存
                if self.map_data:
                    try:
                        # 如果是字典类型
                        if isinstance(self.map_data, dict):
                            self.map_data['models'] = self.models
                        # 如果是 MapData 对象类型
                        elif hasattr(self.map_data, 'models'):
                            self.map_data.models = self.models
                        self.status_label.config(text="已保存模型到地图")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"保存模型失败: {str(e)}")
                        messagebox.showerror("保存失败", f"保存模型到地图失败: {str(e)}")
        
        # 无需停止渲染线程或清理pygame资源
        
        # 关闭窗口
        self.dialog.destroy()
        
    def apply_material_to_selected(self):
        """将当前材质应用到选中的模型"""
        if self.selected_model_index < 0 or self.selected_model_index >= len(self.models):
            messagebox.showinfo("提示", "请先选择一个模型")
            return
        
        # 收集当前材质属性
        material = {
            'name': self.material_name_var.get(),
            'diffuse_color': self.hex_to_rgb(self.diffuse_color_button.cget('background')),
            'specular': self.specular_var.get(),
            'roughness': self.roughness_var.get(),
            'metallic': self.metallic_var.get(),
            'transparency': self.transparency_var.get(),
            'diffuse': self.diffuse_map_var.get(),
            'normal': self.normal_map_var.get(),
            'roughness_map': self.roughness_map_var.get()
        }
        
        # 应用到选中的模型
        self.models[self.selected_model_index]['material'] = material
        
        # 更新状态栏
        self.status_label.config(text=f"已应用材质到模型: {self.models[self.selected_model_index].get('name', '未命名')}")
        
        # 同步到GL视图
        self.update_gl_models()