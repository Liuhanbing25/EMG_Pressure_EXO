#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import queue
import csv
import time
import datetime
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from data_analysis_ui import DataAnalysisTab

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

# Minimal display card for single/dual channel EMG
class EmgCard(QtWidgets.QFrame):
    def __init__(self, slot_index: int):
        super().__init__()
        self.slot_index = slot_index
        self.dual_mode = (self.slot_index >= 12)  
        
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("""
            EmgCard { background-color: #2b2b2b; border: 1px solid #444; border-radius: 8px; margin-bottom: 8px; }
            QLabel { color: #E0E0E0; font-weight: bold; font-size: 13px; border: none; }
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        self.lbl_title = QtWidgets.QLabel(f"⚡ EMG {self.slot_index:02d} | Waiting...")
        layout.addWidget(self.lbl_title)

        # CH1 Plot Area
        layout.addWidget(QtWidgets.QLabel("Channel 1"))
        self.plot_ch1 = pg.PlotWidget()
        self.plot_ch1.setFixedHeight(100)
        self.plot_ch1.showGrid(x=True, y=True, alpha=0.3)
        self.curve_raw1 = self.plot_ch1.plot(pen=pg.mkPen(color=(120, 120, 120), width=1))
        self.curve_env1 = self.plot_ch1.plot(pen=pg.mkPen(color=(0, 255, 0), width=2))
        layout.addWidget(self.plot_ch1)

        # CH2 Plot Area
        if self.dual_mode:
            layout.addWidget(QtWidgets.QLabel("Channel 2"))
            self.plot_ch2 = pg.PlotWidget()
            self.plot_ch2.setFixedHeight(100)
            self.plot_ch2.showGrid(x=True, y=True, alpha=0.3)
            self.curve_raw2 = self.plot_ch2.plot(pen=pg.mkPen(color=(120, 120, 120), width=1))
            self.curve_env2 = self.plot_ch2.plot(pen=pg.mkPen(color=(0, 180, 255), width=2))
            layout.addWidget(self.plot_ch2)

        self.max_pts = 1500
        self.buf_raw1 = []
        self.buf_env1 = []
        self.buf_raw2 = []
        self.buf_env2 = []
        
        self.hide() 

    def clear_buffers(self):
        self.buf_raw1.clear(); self.buf_env1.clear()
        self.buf_raw2.clear(); self.buf_env2.clear()
        self.curve_raw1.setData([]); self.curve_env1.setData([])
        if self.dual_mode:
            self.curve_raw2.setData([]); self.curve_env2.setData([])

    def update_data(self, bat_pct, raw1, env1, raw2, env2, is_playback=False):
        self.show()
        mode_str = "Playback" if is_playback else "Connected"
        self.lbl_title.setText(f"⚡ EMG {self.slot_index:02d} - {mode_str} | Bat: {int(bat_pct)}%")
        self.lbl_title.setStyleSheet("color: #00FF00;" if not is_playback else "color: #FFAA00;")

        self.buf_raw1.extend(raw1)
        self.buf_env1.extend(env1)
        self.buf_raw1 = self.buf_raw1[-self.max_pts:]
        self.buf_env1 = self.buf_env1[-self.max_pts:]
        self.curve_raw1.setData(self.buf_raw1)
        self.curve_env1.setData(self.buf_env1)

        if self.dual_mode:
            self.buf_raw2.extend(raw2)
            self.buf_env2.extend(env2)
            self.buf_raw2 = self.buf_raw2[-self.max_pts:]
            self.buf_env2 = self.buf_env2[-self.max_pts:]
            self.curve_raw2.setData(self.buf_raw2)
            self.curve_env2.setData(self.buf_env2)


# =============================================================================
# 2. 鞋垫展示区 (优化空间缩放、边缘高保真居中渲染 + 动态异质性Scale补偿)
# =============================================================================
class InsolePanel(QtWidgets.QWidget):
    def __init__(self, side: str):
        super().__init__()
        self.side = side
        layout = QtWidgets.QVBoxLayout(self)
        
        self.lbl_title = QtWidgets.QLabel(f"👣 {self.side} Insole | Waiting...")
        self.lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #888888;")
        layout.addWidget(self.lbl_title)

        self.pg_win = pg.GraphicsLayoutWidget()
        self.view = self.pg_win.addViewBox(invertY=True)
        self.view.setAspectLocked(True)

        self._load_spatial_layout()

        self.img_item = pg.ImageItem(image=np.zeros((self.grid_w, self.grid_h)))
        self.img_item.setRect(QtCore.QRectF(self.phys_min_x, self.phys_min_y, self.phys_max_x - self.phys_min_x, self.phys_max_y - self.phys_min_y))
        
        pos = np.array([0, 0.3, 0.7, 1])
        color = np.array([[0,0,0,0], [0,100,255,180], [0,255,255,230], [255,255,255,255]], dtype=np.ubyte)
        self.img_item.setLookupTable(pg.ColorMap(pos, color).getLookupTable(0.0, 1.0, 256))
        self.view.addItem(self.img_item)

        self.scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(100, 100, 100, 100))
        self.scatter.setData(pos=self.coords_map)
        self.view.addItem(self.scatter)

        self.cop = pg.ScatterPlotItem(size=30, symbol='+', brush='r')
        self.view.addItem(self.cop)
        
        layout.addWidget(self.pg_win)

        self.pointwise_baseline = None
        self.playback_scale_factor = 1.0  # 独属单腿站立灵敏度缩放系数，默认不缩放
        self._try_load_test_zero_folder()

    def _load_spatial_layout(self):
        self.grid_w, self.grid_h = 60, 180
        loaded = False
        try:
            layout_file = "insole_layout.json"
            if not os.path.exists(layout_file): layout_file = "insole_layout_38.json"
            
            with open(layout_file, "r", encoding='utf-8') as f:
                data = json.load(f)
            
            valid = [i for i in data if isinstance(i, dict) and all(k in i for k in ('r', 'c', 'x', 'y'))]
            valid.sort(key=lambda x: (x['r'], x['c']))
            
            xs = [float(p['x']) for p in valid]
            ys = [float(p['y']) for p in valid]
            phys_min_x, phys_max_x = min(xs), max(xs)
            max_y_src = max(ys)

            coords = []
            for p in valid:
                rx = phys_max_x - (float(p['x']) - phys_min_x) if self.side == "Right" else float(p['x'])
                ry = max_y_src - float(p['y'])
                coords.append([rx, ry])
            
            self.coords_map = np.array(coords, dtype=np.float32)
            loaded = True
        except Exception:
            loaded = False

        if not loaded:
            n = 233
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            coords = []
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= n: break
                    coords.append([float(c * 10.0), float(r * 10.0)])
                    idx += 1
            self.coords_map = np.array(coords, dtype=np.float32)

        self.phys_min_x, self.phys_max_x = float(np.min(self.coords_map[:, 0])), float(np.max(self.coords_map[:, 0]))
        self.phys_min_y, self.phys_max_y = float(np.min(self.coords_map[:, 1])), float(np.max(self.coords_map[:, 1]))
        
        dx = self.phys_max_x - self.phys_min_x
        dy = self.phys_max_y - self.phys_min_y
        
        padding_x = dx * 0.10
        padding_y = dy * 0.10
        self.view.setRange(
            xRange=(self.phys_min_x - padding_x, self.phys_max_x + padding_x), 
            yRange=(self.phys_min_y - padding_y, self.phys_max_y + padding_y)
        )
        
        sx = (self.grid_w - 1) / (dx + 0.001)
        sy = (self.grid_h - 1) / (dy + 0.001)
        self.idx_r = np.array([max(0, min(self.grid_h - 1, int((y - self.phys_min_y) * sy))) for x, y in self.coords_map])
        self.idx_c = np.array([max(0, min(self.grid_w - 1, int((x - self.phys_min_x) * sx))) for x, y in self.coords_map])

    def _try_load_test_zero_folder(self):
        import pandas as pd
        zero_dir = "test/zero"
        target_file = f"Insole_{self.side}.csv" if os.path.exists(os.path.join(zero_dir, f"Insole_{self.side}.csv")) else f"Record_{self.side}.csv"
        full_path = os.path.join(zero_dir, target_file)
        
        if os.path.exists(full_path) and os.path.getsize(full_path) > 100:
            try:
                df_zero = pd.read_csv(full_path)
                p_cols = [c for c in df_zero.columns if c.startswith('P_')]
                if p_cols:
                    self.pointwise_baseline = df_zero[p_cols].mean().values.astype(np.float32)
                    print(f"✅ Baseline calibrated for {self.side} insole.")
                    return
            except Exception as e:
                print(f"⚠️ Failed to load zero calibration data: {e}")
        print(f"ℹ️ Continuous learning baseline applied for {self.side} insole.")

    def reset_ui(self):
        self.lbl_title.setText(f"👣 {self.side} Insole | Waiting...")
        self.lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #888888;")
        self.img_item.setImage(np.zeros((self.grid_h, self.grid_w)).T)
        self.scatter.setBrush(pg.mkBrush(100, 100, 100, 100))
        self.cop.setData([], [])

    def update_data(self, data_dict, is_playback=False):
        raw_bat = data_dict['bat']
        v_bat = (raw_bat / 4095.0) * 10
        pct = 100 if v_bat > 4.25 else max(0, min(100, (v_bat - 2.7) / (4.2 - 2.7) * 100))

        color_hex = "#FFAA00" if is_playback else "#00AACC"
        self.lbl_title.setText(f"👣 {self.side} ({data_dict['size']}) | Bat: {int(pct)}% | F1:{int(data_dict['flex'][0])} F2:{int(data_dict['flex'][1])}")
        self.lbl_title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {color_hex};")
        
        press_vals = np.array(data_dict['press'], dtype=np.float32)
        
        if self.pointwise_baseline is None:
            if np.median(press_vals) < 1350.0:
                self.pointwise_baseline = press_vals.copy()
            else:
                self.pointwise_baseline = np.full_like(press_vals, 1025.0)
        
        # 1. 绝对点对点空载扣除
        vals = press_vals - self.pointwise_baseline
        
        # 2. 核心联动：如果是历史对齐回放模式，全自动乘上双脚独属的单腿站立灵敏度比例补偿因子
        if is_playback:
            vals = vals * self.playback_scale_factor
            
        # 3. 完美对应参考平台最左端硬级标定：Threshold = 0 零死区敏锐捕捉
        vals[vals < 0] = 0.0  
        
        if len(vals) != len(self.coords_map): 
            return

        # 散点点阵优化（未踩踏传感器隐形灰色，踩踏区域展现通透饱满高亮红）
        if self.scatter.isVisible():
            # 配合降噪和色彩范围，同步调小受压点散点亮度映射
            brushes = [pg.mkBrush(0, 0, 0, 0) if v <= 0.5 else pg.mkBrush(255, 30, 30, int(min(255, 120 + v * 1.5))) for v in vals] 
            self.scatter.setData(pos=self.coords_map, brush=brushes)

        # 4. 高斯空间连续晕染云图渲染链路
        if self.img_item.isVisible():
            heatmap_grid = np.zeros((self.grid_h, self.grid_w))
            mask = vals > 0
            if np.any(mask):
                # 🚀 降低过载优化点A：将注入强度由 12.0 下调至 5.5，防大面积过曝
                heatmap_grid[self.idx_r[mask], self.idx_c[mask]] = vals[mask] * 5.5
                
            if gaussian_filter is not None:
                # 引入各向异性平滑（sigma=2.7 完美模糊半径）
                disp = gaussian_filter(heatmap_grid, sigma=(4.5, 3.0))
            else:
                disp = heatmap_grid
                
            # 🚀 扩大范围优化点B：将色谱归一化映射分母由 6.0 扩大至 7.5
            # 这相当于拉大了全局最大值范围（Y值区间），有效阻止画面变死白，恢复丰富的颜色递进层次
            disp = disp / 7.5  
            self.img_item.setImage(disp.T, autoLevels=False, levels=(0, 1.0))

        # Realtime Center of Pressure mapping
        total = float(np.sum(vals))
        if total > 100: 
            cx = float(np.sum(vals * self.coords_map[:, 0]) / total)
            cy = float(np.sum(vals * self.coords_map[:, 1]) / total)
            self.cop.setData([cx], [cy])
        else:
            self.cop.setData([], [])


# =============================================================================
# 3. 一体化展示主窗口 Frontend
# =============================================================================
class MasterFrontend(QtWidgets.QMainWindow):
    def __init__(self, shared_state, emg_queue, insole_queue, exo_command_queue=None, exo_data_queue=None):
        super().__init__()
        self.shared_state = shared_state
        self.emg_queue = emg_queue
        self.insole_queue = insole_queue
        self.exo_q = exo_command_queue  
        self.exo_data_q = exo_data_queue 

        self.playback_frames = []      
        self.playback_index = 0       
        self.is_playing = False        
        self.loaded_folders = []       
        self.playback_min_unix = 0.0   

        self._max_live_pts = 300
        self._buf_live_exo_tgt = []
        self._buf_live_exo_meas = []

        self.setWindowTitle("Unified Biomechanics Monitoring System (Real-time Acquisition & Synchronized Playback)")
        self.resize(1720, 980)
        self.setStyleSheet("QMainWindow { background-color: #1e1e1e; } Label { color: white; }")

        self._reported_burst_times = set()

        self._init_ui()

        self.global_timer = QtCore.QTimer()
        self.global_timer.timeout.connect(self._global_timer_tick)
        self.global_timer.start(33)

    def _init_ui(self):
        self.main_tabs = QtWidgets.QTabWidget()
        self.main_tabs.setStyleSheet("QTabBar::tab { font-size: 16px; font-weight: bold; padding: 10px; }")
        self.setCentralWidget(self.main_tabs)
        
        tab_live = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(tab_live)

        top_control_box = QtWidgets.QGroupBox("Control Station Hub")
        top_control_box.setStyleSheet("""
            QGroupBox { color: #00FFCC; font-weight: bold; font-size: 14px; border: 1px solid #444; border-radius: 6px; margin-top: 10px; padding: 10px;}
        """)
        top_layout = QtWidgets.QVBoxLayout(top_control_box)

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(QtWidgets.QLabel("🔄 Target Operating Mode:"))
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["🟢 Real-time Acquisition", "📂 Historical Playback Alignment"])
        self.combo_mode.setStyleSheet("background-color: #333; color: white; padding: 5px; font-size: 14px; font-weight: bold;")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.combo_mode)
        mode_layout.addStretch()
        top_layout.addLayout(mode_layout)

        self.control_stack = QtWidgets.QStackedWidget()
        
        # Realtime pane setup
        page_realtime = QtWidgets.QWidget()
        layout_rt = QtWidgets.QVBoxLayout(page_realtime)
        layout_rt.setContentsMargins(0, 5, 0, 0)
        self.btn_rec = QtWidgets.QPushButton("🔴 Start Global Synchronized Recording (1000Hz Data Logging)")
        self.btn_rec.setMinimumHeight(45)
        self.btn_rec.setStyleSheet("background-color: #AA0000; color: white; font-weight: bold; font-size: 15px; border-radius: 5px;")
        self.btn_rec.clicked.connect(self._toggle_record)
        layout_rt.addWidget(self.btn_rec)

        self.btn_exo_burst = QtWidgets.QPushButton("🔥 Trigger Exoskeleton Transient Perturbation")
        self.btn_exo_burst.setMinimumHeight(45)
        self.btn_exo_burst.setStyleSheet("""
            QPushButton { background-color: #FFAA00; color: black; font-weight: bold; font-size: 15px; border-radius: 5px; }
            QPushButton:hover { background-color: #FFCC33; }
            QPushButton:pressed { background-color: #CC8800; }
        """)
        self.btn_exo_burst.clicked.connect(self._trigger_exo_perturbation)
        layout_rt.addWidget(self.btn_exo_burst)

        self.control_stack.addWidget(page_realtime)

        # Playback pane setup
        page_playback = QtWidgets.QWidget()
        layout_pb = QtWidgets.QVBoxLayout(page_playback)
        layout_pb.setContentsMargins(0, 5, 0, 0)
        
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_load_folders = QtWidgets.QPushButton("➕ Add Playback Directories")
        self.btn_load_folders.setStyleSheet("background-color: #00557F; color: white; font-weight: bold; min-height:30px;")
        self.btn_load_folders.clicked.connect(self._load_folders_dialog)
        btn_row.addWidget(self.btn_load_folders)

        self.btn_play_pause = QtWidgets.QPushButton("▶ Play")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.setStyleSheet("background-color: #333; color: #888; font-weight: bold; min-height:30px; width: 100px;")
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)
        btn_row.addWidget(self.btn_play_pause)

        self.lbl_folder_status = QtWidgets.QLabel("No data directories mapped.")
        self.lbl_folder_status.setStyleSheet("color: #AAA; font-size: 12px;")
        btn_row.addWidget(self.lbl_folder_status, 1)
        layout_pb.addLayout(btn_row)

        progress_row = QtWidgets.QHBoxLayout()
        self.slider_progress = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_progress.setEnabled(False)
        self.slider_progress.sliderMoved.connect(self._on_slider_scrubbed)
        progress_row.addWidget(self.slider_progress)
        
        self.lbl_time_track = QtWidgets.QLabel("00:00.0 / 00:00.0")
        self.lbl_time_track.setStyleSheet("font-family: Consolas; font-size: 13px; color: #00FFCC;")
        progress_row.addWidget(self.lbl_time_track)
        layout_pb.addLayout(progress_row)

        self.control_stack.addWidget(page_playback)
        top_layout.addWidget(self.control_stack)
        main_layout.addWidget(top_control_box)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        visual_left_widget = QtWidgets.QWidget()
        visual_left_layout = QtWidgets.QVBoxLayout(visual_left_widget)
        visual_left_layout.setContentsMargins(0, 0, 0, 0)

        insole_container = QtWidgets.QWidget()
        insole_layout = QtWidgets.QHBoxLayout(insole_container)
        insole_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_l = InsolePanel("Left")
        self.panel_r = InsolePanel("Right")
        insole_layout.addWidget(self.panel_l)
        insole_layout.addWidget(self.panel_r)
        visual_left_layout.addWidget(insole_container, stretch=3)

        self.plot_exo_live = pg.PlotWidget(title="🦾 Exoskeleton Joint Realtime Alignments")
        self.plot_exo_live.setFixedHeight(220)
        self.plot_exo_live.showGrid(x=True, y=True, alpha=0.3)
        self.plot_exo_live.addLegend()
        self.plot_exo_live.setLabel('left', 'Position', units='Deg')
        self.curve_exo_live_tgt = self.plot_exo_live.plot(pen=pg.mkPen(color=(255, 85, 85), width=2), name="Target Command")
        self.curve_exo_live_meas = self.plot_exo_live.plot(pen=pg.mkPen(color=(85, 255, 255), width=2), name="Measured Feedback")
        visual_left_layout.addWidget(self.plot_exo_live, stretch=2)

        splitter.addWidget(visual_left_widget)

        emg_scroll = QtWidgets.QScrollArea()
        emg_scroll.setWidgetResizable(True)
        emg_scroll.setMinimumWidth(400)
        emg_scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")
        
        emg_container = QtWidgets.QWidget()
        self.emg_layout = QtWidgets.QVBoxLayout(emg_container)
        self.emg_layout.setAlignment(QtCore.Qt.AlignTop)
        self.emg_layout.setContentsMargins(0, 0, 5, 0)
        
        self.emg_cards = {}
        for i in range(1, 18): 
            card = EmgCard(i)
            self.emg_cards[i] = card
            self.emg_layout.addWidget(card)
            
        emg_scroll.setWidget(emg_container)
        splitter.addWidget(emg_scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, stretch=1)
        
        self.main_tabs.addTab(tab_live, "📊 Live Monitoring Hub")

        self.tab_analysis = DataAnalysisTab()
        self.main_tabs.addTab(self.tab_analysis, "🔬 Offline Analysis Station")

    def _try_extract_playback_bilateral_scales(self):
        """核心新增：检测 test/left 和 test/right 文件夹，全自动精算 99% 站立幅值反算异质性Scale补偿因子"""
        import pandas as pd
        
        # 重置因子默认值
        self.panel_l.playback_scale_factor = 1.0
        self.panel_r.playback_scale_factor = 1.0
        
        # 定义搜索路径矩阵
        dir_l = "test/left" if os.path.exists("test/left") else "test/Left"
        dir_r = "test/right" if os.path.exists("test/right") else "test/Right"
        
        if os.path.exists(dir_l) and os.path.exists(dir_r):
            print("🚀 [Scale对齐引擎] 侦测到 test/left 和 test/right 标定数据集！正在解算跨通道灵敏度不均因子...")
            try:
                # 处理左脚标定
                file_l = os.path.join(dir_l, "Insole_Left.csv") if os.path.exists(os.path.join(dir_l, "Insole_Left.csv")) else os.path.join(dir_l, "Record_Left.csv")
                if os.path.exists(file_l):
                    df_l = pd.read_csv(file_l)
                    p_cols_l = [c for c in df_l.columns if c.startswith('P_')]
                    z_l = self.panel_l.pointwise_baseline if self.panel_l.pointwise_baseline is not None else np.full(len(p_cols_l), 1025.0)
                    mat_l = np.clip(df_l[p_cols_l].values - z_l, 0, None)
                    max_99_l = np.quantile(np.sum(mat_l, axis=1), 0.99)
                    if max_99_l > 0:
                        self.panel_l.playback_scale_factor = 10000.0 / max_99_l
                
                # 处理右脚标定
                file_r = os.path.join(dir_r, "Insole_Right.csv") if os.path.exists(os.path.join(dir_r, "Insole_Right.csv")) else os.path.join(dir_r, "Record_Right.csv")
                if os.path.exists(file_r):
                    df_r = pd.read_csv(file_r)
                    p_cols_r = [c for c in df_r.columns if c.startswith('P_')]
                    z_r = self.panel_r.pointwise_baseline if self.panel_r.pointwise_baseline is not None else np.full(len(p_cols_r), 1025.0)
                    mat_r = np.clip(df_r[p_cols_r].values - z_r, 0, None)
                    max_99_r = np.quantile(np.sum(mat_r, axis=1), 0.99)
                    if max_99_r > 0:
                        self.panel_r.playback_scale_factor = 10000.0 / max_99_r
                        
                print(f"✅ [Scale对齐引擎] 解算完毕。左脚补偿系数: {self.panel_l.playback_scale_factor:.4f} | 右脚补偿系数: {self.panel_r.playback_scale_factor:.4f}")
            except Exception as e:
                print(f"⚠️ [Scale对齐引擎] 提取单腿站立标定失败: {e}，系统已降级回默认幅值因子。")
        else:
            print("ℹ️ [Scale对齐引擎] 未在根目录下发现有效单腿站立标定集，自动应用常规 1.0 等幅回放。")

    def _trigger_exo_perturbation(self):
        if not self.shared_state.get('is_recording', False):
            QtWidgets.QMessageBox.warning(
                self, 
                "Acquisition Blocked", 
                "Please enable global data logging recording state first."
            )
            return

        if self.exo_q is not None:
            try:
                self.exo_q.put_nowait("TRIGGER_PERTURBATION")
                print("[GUI Main Thread] ⚡ Perturbation pulse dispatched into IPC queue.")
            except queue.Full:
                print("[GUI Main Thread] ⚠️ Command pipe full. Do not flood.")
        else:
            QtWidgets.QMessageBox.critical(self, "Error", "Exoskeleton engine link process unmapped.")

    def _on_mode_changed(self, index):
        self.control_stack.setCurrentIndex(index)
        if self.shared_state.get('is_recording', False):
            self._toggle_record()
        self.is_playing = False
        self.btn_play_pause.setText("▶ Play")

        self.panel_l.reset_ui()
        self.panel_r.reset_ui()
        
        self.curve_exo_live_tgt.setData([])
        self.curve_exo_live_meas.setData([])
        self._buf_live_exo_tgt.clear()
        self._buf_live_exo_meas.clear()
        
        for item in self.plot_exo_live.items()[:]:
            if isinstance(item, pg.InfiniteLine):
                self.plot_exo_live.removeItem(item)
        self._reported_burst_times.clear()

        for card in self.emg_cards.values():
            card.clear_buffers()
            card.hide()

    def _toggle_record(self):
        current_state = self.shared_state.get('is_recording', False)
        if not current_state:
            session_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.shared_state['session_id'] = session_id
            self.shared_state['record_start_unix'] = time.time()
            self.shared_state['is_recording'] = True
            
            self.btn_rec.setText(f"Stop Logging (Writing tracking payload under subpath: test/{session_id}/)")
            self.btn_rec.setStyleSheet("background-color: #00AA00; color: white; font-weight: bold; font-size: 15px; border-radius: 5px;")
        else:
            self.shared_state['is_recording'] = False
            self.btn_rec.setText("🔴 Start Global Synchronized Recording (1000Hz Data Logging)")
            self.btn_rec.setStyleSheet("background-color: #AA0000; color: white; font-weight: bold; font-size: 15px; border-radius: 5px;")

    def _load_folders_dialog(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)
        file_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        file_dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        
        tree_view = file_dialog.findChild(QtWidgets.QTreeView)
        if tree_view: tree_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        list_view = file_dialog.findChild(QtWidgets.QListView)
        if list_view: list_view.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        
        if file_dialog.exec_():
            selected = file_dialog.selectedFiles()
            if selected:
                # 触发跨通道双脚灵敏度异质性标定自动分析提取
                self._try_extract_playback_bilateral_scales()
                self._process_and_align_playback_data(selected)

    def _process_and_align_playback_data(self, folders):
        self.is_playing = False
        self.btn_play_pause.setText("▶ Play")
        
        progress_dialog = QtWidgets.QProgressDialog("⚡ Aligning absolute Unix tracking indexes...", "Cancel", 0, 100, self)
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.show()
        progress_dialog.setValue(5)

        all_csv_rows = [] 
        total_files = 0
        for f in folders:
            if os.path.exists(f): total_files += len([n for n in os.listdir(f) if n.endswith('.csv')])
        
        if total_files == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "No matching tabular log arrays discovered under selection.")
            return

        file_count = 0
        for folder in folders:
            if not os.path.isdir(folder): continue
            for file_name in os.listdir(folder):
                if not file_name.endswith('.csv'): continue
                file_path = os.path.join(folder, file_name)
                
                is_emg = file_name.startswith("emg_dev")
                is_insole = file_name.startswith("Insole_")
                is_box = file_name.startswith("Exoskeleton_Data") 
                if not (is_emg or is_insole or is_box): continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if 'Unix_Time_s' not in row: break 
                            u_time = float(row['Unix_Time_s'])
                            
                            if is_emg:
                                dev_id = int(file_name.replace("emg_dev", "").replace(".csv", ""))
                                all_csv_rows.append((u_time, 'emg', dev_id, row))
                            elif is_insole:
                                side = file_name.replace("Insole_", "").replace(".csv", "")
                                all_csv_rows.append((u_time, 'insole', side, row))
                            elif is_box:
                                all_csv_rows.append((u_time, 'exoskeleton', 'exo_node', row))
                except Exception as e:
                    print(f"Playback parser fault: {file_path} -> {e}")
                
                file_count += 1
                progress_dialog.setValue(int(5 + (file_count / total_files) * 60))
                QtCore.QCoreApplication.processEvents()

        if not all_csv_rows:
            progress_dialog.close()
            QtWidgets.QMessageBox.warning(self, "Error", "Absolute timestamp metrics extraction yielded empty dataset.")
            return

        all_csv_rows.sort(key=lambda x: x[0])
        
        min_unix = all_csv_rows[0][0]
        max_unix = all_csv_rows[-1][0]
        total_duration = max_unix - min_unix
        
        total_frames = int(total_duration / 0.033) + 1
        self.playback_frames = [{"emg": {}, "insole": [], "exo": {"target": None, "measured": None, "is_bursting": 0}} for _ in range(total_frames)]
        self.playback_min_unix = min_unix
        
        self._reported_burst_times.clear()
        for item in self.plot_exo_live.items()[:]:
            if isinstance(item, pg.InfiniteLine):
                self.plot_exo_live.removeItem(item)

        for u_time, f_type, f_id, row in all_csv_rows:
            frame_idx = int((u_time - min_unix) / 0.033)
            if frame_idx >= total_frames: frame_idx = total_frames - 1
            
            if f_type == 'emg':
                if f_id not in self.playback_frames[frame_idx]['emg']:
                    self.playback_frames[frame_idx]['emg'][f_id] = {"raw1":[], "env1":[], "raw2":[], "env2":[]}
                
                self.playback_frames[frame_idx]['emg'][f_id]['raw1'].append(float(row.get('ch1_raw_uV', 0)))
                self.playback_frames[frame_idx]['emg'][f_id]['env1'].append(float(row.get('ch1_env_uV', 0)))
                self.playback_frames[frame_idx]['emg'][f_id]['raw2'].append(float(row.get('ch2_raw_uV', 0)))
                self.playback_frames[frame_idx]['emg'][f_id]['env2'].append(float(row.get('ch2_env_uV', 0)))
                
            elif f_type == 'insole':
                press_list = []
                p_idx = 1
                while f'P_{p_idx}' in row:
                    press_list.append(float(row[f'P_{p_idx}']))
                    p_idx += 1
                
                size_lbl = "Size 38" if len(press_list) < 200 else "Size 42"
                
                snapshot = {
                    'side': f_id,
                    'size': size_lbl,
                    'press': press_list,
                    'bat': float(row.get('Bat', 4095)),
                    'flex': [float(row.get('Flex1', 0)), float(row.get('Flex2', 0))]
                }
                self.playback_frames[frame_idx]['insole'].append(snapshot)

            elif f_type == 'exoskeleton': 
                self.playback_frames[frame_idx]['exo']['target'] = float(row.get('Target_Position_Deg', 0.0))
                self.playback_frames[frame_idx]['exo']['measured'] = float(row.get('Measured_Position_Deg', 0.0))
                self.playback_frames[frame_idx]['exo']['is_bursting'] = int(row.get('Is_Bursting', 0))

        progress_dialog.setValue(100)
        progress_dialog.close()

        self.playback_index = 0
        self.slider_progress.setEnabled(True)
        self.slider_progress.setRange(0, total_frames - 1)
        self.slider_progress.setValue(0)
        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.setStyleSheet("background-color: #00AA00; color: white; font-weight: bold; min-height:30px; width: 100px;")
        
        self.loaded_folders = [os.path.basename(f) for f in folders]
        self.lbl_folder_status.setText(f"Synchronized {len(folders)} folders ({total_frames} frames, {total_duration:.1f}s)")
        
        self._render_single_playback_frame(0)

    def _toggle_play_pause(self):
        if not self.playback_frames: return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play_pause.setText("⏸ Pause")
            self.btn_play_pause.setStyleSheet("background-color: #FFAA00; color: black; font-weight: bold; min-height:30px;")
        else:
            self.btn_play_pause.setText("▶ Play")
            self.btn_play_pause.setStyleSheet("background-color: #00AA00; color: white; font-weight: bold; min-height:30px;")

    def _on_slider_scrubbed(self, value):
        if not self.playback_frames: return
        self.playback_index = value
        self._render_single_playback_frame(self.playback_index)

    def _global_timer_tick(self):
        mode_idx = self.combo_mode.currentIndex()
        
        if mode_idx == 0:
            while True:
                try:
                    emg_payloads = self.emg_queue.get_nowait() 
                    for dev_data in emg_payloads:
                        slot = dev_data['slot_index']
                        if slot in self.emg_cards:
                            self.emg_cards[slot].update_data(
                                dev_data['bat_pct'], 
                                dev_data['raw_ch1'], dev_data['env_ch1'],
                                dev_data.get('raw_ch2', []), dev_data.get('env_ch2', []),
                                is_playback=False
                            )
                except queue.Empty:
                    break 

            latest_insole = None
            while True:
                try: latest_insole = self.insole_queue.get_nowait()
                except queue.Empty: break
                    
            if latest_insole:
                for side_data in latest_insole:
                    if side_data['side'] == 'Left': self.panel_l.update_data(side_data, is_playback=False)
                    elif side_data['side'] == 'Right': self.panel_r.update_data(side_data, is_playback=False)

            while self.exo_data_q is not None and not self.exo_data_q.empty():
                try:
                    exo_live_packet = self.exo_data_q.get_nowait()
                    self._buf_live_exo_tgt.append(exo_live_packet["target"])
                    self._buf_live_exo_meas.append(exo_live_packet["measured"])
                    
                    self._buf_live_exo_tgt = self._buf_live_exo_tgt[-self._max_live_pts:]
                    self._buf_live_exo_meas = self._buf_live_exo_meas[-self._max_live_pts:]
                    
                    self.curve_exo_live_tgt.setData(self._buf_live_exo_tgt)
                    self.curve_exo_live_meas.setData(self._buf_live_exo_meas)
                except queue.Empty:
                    break
                    
        elif mode_idx == 1:
            if self.is_playing and self.playback_frames:
                if self.playback_index < len(self.playback_frames):
                    self.slider_progress.blockSignals(True)
                    self.slider_progress.setValue(self.playback_index)
                    self.slider_progress.blockSignals(False)
                    
                    self._render_single_playback_frame(self.playback_index)
                    self.playback_index += 1
                else:
                    self.is_playing = False
                    self.btn_play_pause.setText("▶ Play")
                    self.btn_play_pause.setStyleSheet("background-color: #00AA00; color: white; font-weight: bold; min-height:30px;")
                    self.playback_index = 0

    def _render_single_playback_frame(self, index):
        if index >= len(self.playback_frames): return
        frame = self.playback_frames[index]

        for dev_id, card in self.emg_cards.items():
            if dev_id in frame['emg']:
                d = frame['emg'][dev_id]
                card.update_data(100, d['raw1'], d['env1'], d['raw2'], d['env2'], is_playback=True)

        for insole_data in frame['insole']:
            if insole_data['side'] == 'Left':
                self.panel_l.update_data(insole_data, is_playback=True)
            elif insole_data['side'] == 'Right':
                self.panel_r.update_data(insole_data, is_playback=True)

        cur_sec = index * 0.033
        tot_sec = len(self.playback_frames) * 0.033
        self.lbl_time_track.setText(f"{cur_sec // 60:02.0f}:{cur_sec % 60:04.1f} / {tot_sec // 60:02.0f}:{tot_sec % 60:04.1f}")

        time_axis = []
        target_series = []
        measured_series = []
        
        for i in range(index + 1):
            f_exo = self.playback_frames[i]['exo']
            if f_exo['target'] is not None and f_exo['measured'] is not None:
                t_val = i * 0.033
                time_axis.append(t_val)
                target_series.append(f_exo['target'])
                measured_series.append(f_exo['measured'])
                
                if f_exo['is_bursting'] == 1 and i > 0 and self.playback_frames[i-1]['exo']['is_bursting'] == 0:
                    if t_val not in self._reported_burst_times:
                        self._reported_burst_times.add(t_val)
                        
                        v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', style=QtCore.Qt.DashLine, width=2))
                        v_line.setValue(t_val)
                        self.plot_exo_live.addItem(v_line)

        if time_axis:
            self.curve_exo_live_tgt.setData(time_axis, target_series)
            self.curve_exo_live_meas.setData(time_axis, measured_series)

    def closeEvent(self, event):
        self.shared_state['cmd_stop_all'] = True
        event.accept()


def run_gui_process(shared_state, emg_queue, insole_queue, exo_command_queue=None, exo_data_queue=None):
    app = QtWidgets.QApplication(sys.argv)
    window = MasterFrontend(shared_state, emg_queue, insole_queue, exo_command_queue, exo_data_queue)
    window.show()
    sys.exit(app.exec_())
