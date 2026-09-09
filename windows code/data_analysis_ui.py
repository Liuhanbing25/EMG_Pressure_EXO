#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from scipy.signal import savgol_filter

try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None

class DataAnalysisTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        # 🌟 新增：持久化缓存当前选择的分析文件夹路径，防止调参时重复弹窗
        self.target_folder = None  
        
        # Calibration parameters memory
        self.zero_offset_l = None  
        self.zero_offset_r = None  
        self.scale_l = 1.0
        self.scale_r = 1.0
        self.mvc_values = {}       
        self.mvc_edits = {}        
        
        self.coords_map_l = self._load_layout("Left")
        self.coords_map_r = self._load_layout("Right")
        
        # Plot element references cache
        self.exo_burst_lines = []
        self.cop_v_lines = []
        self.emg_curves = {}
        self.emg_onset_lines = {}
        
        self._init_ui()

    def _load_layout(self, side):
        try:
            layout_file = "insole_layout.json"
            if not os.path.exists(layout_file): layout_file = "insole_layout_38.json"
            with open(layout_file, "r", encoding='utf-8') as f:
                data = json.load(f)
            valid = [i for i in data if isinstance(i, dict) and all(k in i for k in ('r', 'c', 'x', 'y'))]
            valid.sort(key=lambda x: (x['r'], x['c']))
            
            xs, ys = [float(p['x']) for p in valid], [float(p['y']) for p in valid]
            min_x, max_x, min_y = min(xs), max(xs), min(ys)

            coords_x, coords_y = [], []
            for p in valid:
                cx = float(p['x']) - min_x
                cy = float(p['y']) - min_y
                if side == "Right": 
                    cx = (max_x - min_x) - cx 
                coords_x.append(cx)
                coords_y.append(cy)
            return np.column_stack((coords_x, coords_y))
        except Exception:
            return np.array([[float(c*10), float(r*10)] for r in range(16) for c in range(15)][:233], dtype=np.float32)

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # ==================== 1. Left Control Panel ====================
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setFixedWidth(375)
        left_scroll.setWidgetResizable(True)
        left_widget = QtWidgets.QWidget()
        v_layout = QtWidgets.QVBoxLayout(left_widget)
        v_layout.setContentsMargins(5, 5, 5, 5)
        
        # --- Insole Calibration ---
        gb_insole = QtWidgets.QGroupBox("👣 Insole Parameters & Calibration")
        gb_insole.setStyleSheet("QGroupBox { color: #00AACC; font-weight: bold; }")
        l_insole = QtWidgets.QVBoxLayout(gb_insole)
        
        self.btn_load_zero = QtWidgets.QPushButton("1. Import Tare/Zero Data")
        self.btn_load_zero.clicked.connect(self._calc_pointwise_zero)
        self.lbl_zero_status = QtWidgets.QLabel("Zero Offset: Not Ready")
        self.lbl_zero_status.setWordWrap(True)
        
        self.btn_load_l_scale = QtWidgets.QPushButton("2. Import Left Single-Stance")
        self.btn_load_l_scale.clicked.connect(lambda: self._calc_scale_99("Left"))
        self.btn_load_r_scale = QtWidgets.QPushButton("3. Import Right Single-Stance")
        self.btn_load_r_scale.clicked.connect(lambda: self._calc_scale_99("Right"))
        
        f_scale = QtWidgets.QFormLayout()
        f_scale.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        self.ed_scale_l = QtWidgets.QLineEdit("1.000"); self.ed_scale_l.textChanged.connect(self._sync_params)
        self.ed_scale_r = QtWidgets.QLineEdit("1.000"); self.ed_scale_r.textChanged.connect(self._sync_params)
        f_scale.addRow("Left Scale Factor:", self.ed_scale_l)
        f_scale.addRow("Right Scale Factor:", self.ed_scale_r)
        
        l_insole.addWidget(self.btn_load_zero); l_insole.addWidget(self.lbl_zero_status)
        l_insole.addWidget(self.btn_load_l_scale); l_insole.addWidget(self.btn_load_r_scale); l_insole.addLayout(f_scale)
        v_layout.addWidget(gb_insole)

        # --- EMG Calibration ---
        self.gb_emg = QtWidgets.QGroupBox("⚡ EMG Reference Calibration")
        self.gb_emg.setStyleSheet("QGroupBox { color: #00FF00; font-weight: bold; }")
        self.l_emg = QtWidgets.QVBoxLayout(self.gb_emg)
        self.btn_load_mvc = QtWidgets.QPushButton("Import MVC Data (Extract 99% Max)")
        self.btn_load_mvc.clicked.connect(self._calc_dynamic_mvc)
        self.l_emg.addWidget(self.btn_load_mvc)
        self.emg_form_widget = QtWidgets.QWidget()
        self.emg_form_layout = QtWidgets.QFormLayout(self.emg_form_widget)
        self.emg_form_layout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        self.l_emg.addWidget(self.emg_form_widget)
        v_layout.addWidget(self.gb_emg)

        # --- Advanced EMG Settings ---
        self.gb_emg_set = QtWidgets.QGroupBox("⚙️ Advanced EMG Onset Settings")
        self.gb_emg_set.setStyleSheet("QGroupBox { color: #AA55FF; font-weight: bold; }")
        l_emg_set = QtWidgets.QFormLayout(self.gb_emg_set)
        l_emg_set.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        
        self.ed_right_map = QtWidgets.QLineEdit("DEV12_CH1")
        self.ed_left_map = QtWidgets.QLineEdit("DEV15_CH1")
        self.ed_time_diff = QtWidgets.QLineEdit("-50")
        self.ed_amp_th = QtWidgets.QLineEdit("3.5")
        self.ed_persist_th = QtWidgets.QLineEdit("0.85")
        
        # 🌟 核心新增：在 UI 控件区无缝扩容高阶双阈值及高斯熨平参数输入框
        self.ed_smooth_sigma = QtWidgets.QLineEdit("15")       # 二次高斯平滑核大小 (单位: ms)
        self.ed_low_th_factor = QtWidgets.QLineEdit("1.2")     # 逆向追溯跌落的低阈值 SD 倍数
        self.ed_back_track_max = QtWidgets.QLineEdit("60")     # 允许最大向前倒退寻找边界的跨度 (单位: ms)
        
        l_emg_set.addRow("Right Leg Map (e.g. DEV12_CH1):", self.ed_right_map)
        l_emg_set.addRow("Left Leg Map (e.g. DEV15_CH1):", self.ed_left_map)
        l_emg_set.addRow("Left Leg Earliest Allowed Diff (ms):", self.ed_time_diff)
        l_emg_set.addRow("Amplitude SD Multiplier (High Th):", self.ed_amp_th)
        l_emg_set.addRow("EMG Persistence Threshold (0.0~1.0):", self.ed_persist_th)
        
        # 将新增参数渲染挂载到左侧滚动区布局中
        l_emg_set.addRow("Gaussian Smooth Kernel Width (ms):", self.ed_smooth_sigma)
        l_emg_set.addRow("Low Threshold SD Multiplier:", self.ed_low_th_factor)
        l_emg_set.addRow("Max Backward Track Time (ms):", self.ed_back_track_max)
        v_layout.addWidget(self.gb_emg_set)

        # --- Analysis Trigger ---
        gb_run = QtWidgets.QGroupBox("🚀 Biomechanical Comprehensive Analysis")
        gb_run.setStyleSheet("QGroupBox { color: #FFAA00; font-weight: bold; }")
        l_run = QtWidgets.QVBoxLayout(gb_run)
        self.btn_sel_target = QtWidgets.QPushButton("📂 Select Target Data Directory")
        self.btn_sel_target.setStyleSheet("background-color: #00557F; color: white; min-height: 35px; font-weight: bold;")
        self.btn_sel_target.clicked.connect(lambda: self._run_biomechanic_analysis(choose_new_folder=True))
        self.lbl_target = QtWidgets.QLabel("Awaiting folder selection...")
        self.lbl_target.setWordWrap(True)
        l_run.addWidget(self.btn_sel_target); l_run.addWidget(self.lbl_target)
        
        # 🌟 新增：手动应用当前参数并一键重新计算的按键（免除重复选择文件夹的困扰）
        self.btn_refresh = QtWidgets.QPushButton("🔄 Apply Settings & Recalculate")
        self.btn_refresh.setStyleSheet("background-color: #2E7D32; color: white; min-height: 30px; font-weight: bold;")
        self.btn_refresh.clicked.connect(lambda: self._run_biomechanic_analysis(choose_new_folder=False))
        self.l_run = l_run # 挂载句柄
        l_run.addWidget(self.btn_refresh)
        
        v_layout.addWidget(gb_run)
        v_layout.addStretch()

        left_scroll.setWidget(left_widget)
        main_layout.addWidget(left_scroll)

        # ==================== 2. Dual Display Screen Layout ====================
        splitter_horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        pg.setConfigOptions(antialias=True)
        
        left_time_widget = QtWidgets.QWidget()
        left_time_layout = QtWidgets.QVBoxLayout(left_time_widget)
        left_time_layout.setContentsMargins(0,0,0,0)
        splitter_vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        self.p_press = pg.PlotWidget(title="1. Plantar Force Trend Curves (Total/Left/Right)")
        self.p_press.showGrid(x=True, y=True); self.p_press.addLegend()
        self.c_press_tot = self.p_press.plot(pen=pg.mkPen('w', width=2), name="Total")
        self.c_press_l = self.p_press.plot(pen=pg.mkPen('#00AACC', width=1.5), name="Left")
        self.c_press_r = self.p_press.plot(pen=pg.mkPen('#FFAA00', width=1.5), name="Right")
        splitter_vertical.addWidget(self.p_press)
        
        self.p_ratio = pg.PlotWidget(title="2. Plantar Force Weight Distribution Percentage (%)")
        self.p_ratio.showGrid(x=True, y=True); self.p_ratio.addLegend(); self.p_ratio.setYRange(-5, 105)
        self.p_ratio.setXLink(self.p_press) 
        self.c_ratio_l = self.p_ratio.plot(pen=pg.mkPen('#00AACC', width=1.5), name="Left %")
        self.c_ratio_r = self.p_ratio.plot(pen=pg.mkPen('#FFAA00', width=1.5), name="Right %")
        self.line_mid = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#AA00FF', style=QtCore.Qt.DashLine))
        self.p_ratio.addItem(self.line_mid)
        splitter_vertical.addWidget(self.p_ratio)

        self.p_exo_deg = pg.PlotWidget(title="3. Exoskeleton Ankle Target vs. Measured Angle Alignment")
        self.p_exo_deg.showGrid(x=True, y=True); self.p_exo_deg.addLegend()
        self.p_exo_deg.setXLink(self.p_press) 
        self.p_exo_deg.setLabel('left', 'Angle Position', units='Deg')
        self.c_exo_tgt = self.p_exo_deg.plot(pen=pg.mkPen('#FF5555', width=2), name="TargetCmd")
        self.c_exo_meas = self.p_exo_deg.plot(pen=pg.mkPen('#55FFFF', width=2), name="MeasuredFb")
        splitter_vertical.addWidget(self.p_exo_deg)
        
        self.p_cop_v = pg.PlotWidget(title="4. Center of Pressure (CoP) Velocity & Adaptation Onset (mm/s)")
        self.p_cop_v.showGrid(x=True, y=True); self.p_cop_v.addLegend()
        self.p_cop_v.setXLink(self.p_press) 
        self.c_v_l = self.p_cop_v.plot(pen=pg.mkPen('#00AACC', width=1.5), name="Left V")
        self.c_v_r = self.p_cop_v.plot(pen=pg.mkPen('#FFAA00', width=1.5), name="Right V")
        self.c_v_g = self.p_cop_v.plot(pen=pg.mkPen('#AA00FF', width=2), name="Global V")
        splitter_vertical.addWidget(self.p_cop_v)
        
        self.p_emg = pg.PlotWidget(title="5. Multi-channel Electromyography Envelope Activation (% MVC)")
        self.p_emg.showGrid(x=True, y=True); self.p_emg.addLegend()
        self.p_emg.setXLink(self.p_press) 
        splitter_vertical.addWidget(self.p_emg)
        
        splitter_vertical.setSizes([160, 160, 160, 160, 210])
        left_time_layout.addWidget(splitter_vertical)
        splitter_horizontal.addWidget(left_time_widget)
        
        # Spatial View Layout
        right_space_widget = QtWidgets.QWidget()
        right_space_layout = QtWidgets.QVBoxLayout(right_space_widget)
        right_space_layout.setContentsMargins(0,0,0,0)
        
        self.p_cop_xy = pg.PlotWidget(title="Spatial Center of Pressure (CoP) Convex Hull Mapping")
        self.p_cop_xy.showGrid(x=True, y=True)
        self.p_cop_xy.addLegend()
        
        self.bg_spots_l = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(150, 150, 150, 120), pen=None)
        self.bg_spots_r = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(150, 150, 150, 120), pen=None)
        self.p_cop_xy.addItem(self.bg_spots_l)
        self.p_cop_xy.addItem(self.bg_spots_r)
        
        self.c_cop_l_line = self.p_cop_xy.plot(pen=pg.mkPen('#00D4FF', width=2), name="Left Insole CoP")
        self.c_cop_r_line = self.p_cop_xy.plot(pen=pg.mkPen('#FF7800', width=2), name="Right Insole CoP")
        self.c_cop_g_line = self.p_cop_xy.plot(pen=pg.mkPen('#D400FF', width=3), name="Global CoP")
        
        self.curve_hull_l = self.p_cop_xy.plot(pen=pg.mkPen('#00D4FF', width=1.5, style=QtCore.Qt.DashLine))
        self.curve_hull_r = self.p_cop_xy.plot(pen=pg.mkPen('#FF7800', width=1.5, style=QtCore.Qt.DashLine))
        self.curve_hull_g = self.p_cop_xy.plot(pen=pg.mkPen('#D400FF', width=2.5))
        
        self.p_cop_xy.setAspectLocked(True, ratio=1.0)
        self.p_cop_xy.setRange(xRange=(-50, 450), yRange=(-50, 250), padding=0.0)
        self.p_cop_xy.enableAutoRange(axis='xy', enable=False)
        
        right_space_layout.addWidget(self.p_cop_xy, stretch=3)

        gb_metrics = QtWidgets.QGroupBox("📊 Kinetics & Electromyography Comprehensive Report")
        gb_metrics.setStyleSheet("QGroupBox { color: #00FFCC; font-weight: bold; }")
        l_metrics = QtWidgets.QVBoxLayout(gb_metrics)
        self.txt_metrics = QtWidgets.QTextEdit()
        self.txt_metrics.setReadOnly(True)
        self.txt_metrics.setStyleSheet("background-color: #1a1a1a; color: #00FFCC; font-family: Consolas; font-size: 13px;")
        l_metrics.addWidget(self.txt_metrics)
        right_space_layout.addWidget(gb_metrics, stretch=2)

        splitter_horizontal.addWidget(right_space_widget)
        
        splitter_horizontal.setSizes([1150, 550])
        main_layout.addWidget(splitter_horizontal)

    def _sync_params(self):
        try: self.scale_l = float(self.ed_scale_l.text())
        except: pass
        try: self.scale_r = float(self.ed_scale_r.text())
        except: pass
        for key, edit in self.mvc_edits.items():
            try: self.mvc_values[key] = float(edit.text())
            except: pass

    def _calc_pointwise_zero(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Load Unloaded Tare Baseline Folder")
        if not folder: return
        try:
            l_path = os.path.join(folder, "Insole_Left.csv") if os.path.exists(os.path.join(folder, "Insole_Left.csv")) else os.path.join(folder, "Record_Left.csv")
            r_path = os.path.join(folder, "Insole_Right.csv") if os.path.exists(os.path.join(folder, "Insole_Right.csv")) else os.path.join(folder, "Record_Right.csv")
            status_str = "Zero Status: "
            if os.path.exists(l_path) and os.path.getsize(l_path) > 100:
                self.zero_offset_l = pd.read_csv(l_path)[lambda df: [c for c in df.columns if c.startswith('P_')]].mean().values  
                status_str += "Left "
            if os.path.exists(r_path) and os.path.getsize(r_path) > 100:
                self.zero_offset_r = pd.read_csv(r_path)[lambda df: [c for c in df.columns if c.startswith('P_')]].mean().values  
                status_str += "Right "
            self.lbl_zero_status.setText(status_str + " ✅")
            QtWidgets.QMessageBox.information(self, "Success", "Pointwise decoupling offsets calculated successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Extraction failed:\n{e}")

    def _calc_scale_99(self, side):
        if (side == "Left" and self.zero_offset_l is None) or (side == "Right" and self.zero_offset_r is None):
            QtWidgets.QMessageBox.warning(self, "Warning", "Please extract the unloaded baseline tare matrix first.")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, f"Load Single-Stance Data for {side} Foot")
        if not folder: return
        try:
            path = os.path.join(folder, f"Insole_{side}.csv")
            if not os.path.exists(path): path = os.path.join(folder, f"Record_{side}.csv")
            if os.path.exists(path) and os.path.getsize(path) > 100:
                df = pd.read_csv(path)
                p_cols = [c for c in df.columns if c.startswith('P_')]
                
                zero_arr = self.zero_offset_l if side == "Left" else self.zero_offset_r
                press_mat = np.clip(df[p_cols].values - zero_arr, 0, None)
                total_press = np.sum(press_mat, axis=1)
                
                max_99 = np.quantile(total_press, 0.99)
                target_standard = 10000.0 
                scale = target_standard / max_99 if max_99 > 0 else 1.0
                
                if side == "Left": self.ed_scale_l.setText(f"{scale:.4f}")
                else: self.ed_scale_r.setText(f"{scale:.4f}")
                QtWidgets.QMessageBox.information(self, "Success", f"{side} Stance 99% baseline evaluated: {max_99:.1f}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Calculation failed:\n{e}")

    def _calc_dynamic_mvc(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Load Maximal Voluntary Contraction Calibration Folder")
        if not folder: return
        try:
            emg_files = [f for f in os.listdir(folder) if f.startswith('emg_dev') and f.endswith('.csv') and 'imu' not in f]
            for i in range(self.emg_form_layout.count()):
                w = self.emg_form_layout.itemAt(i).widget()
                if w: w.deleteLater()
            self.mvc_values.clear()  
            self.mvc_edits = {}
            
            print("\n============ 📥 Extracting MVC Reference Baseline ============")
            for filename in emg_files:
                dev_name = filename.replace('.csv', '').upper().replace('EMG_', '').strip()
                df = pd.read_csv(os.path.join(folder, filename), comment='#')
                if not df.empty:
                    for col in [c for c in df.columns if '_env_' in c.lower()]:
                        ch_tag = "CH1" if 'ch1' in col.lower() else "CH2"
                        m_key = f"{dev_name}_{ch_tag}".strip()
                        
                        mvc_val = float(df[col].quantile(0.99)) 
                        self.mvc_values[m_key] = mvc_val
                        print(f" Cached validation key -> [{m_key}]: {mvc_val:.2f} uV")
                        
                        edit = QtWidgets.QLineEdit(f"{mvc_val:.2f}")
                        self.mvc_edits[m_key] = edit
                        self.emg_form_layout.addRow(f"{m_key} MVC (uV):", edit)
            print("========================================================\n")
            QtWidgets.QMessageBox.information(self, "Success", f"Identified and structured {len(self.mvc_edits)} EMG channels.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Extraction failed:\n{e}")

    # =========================================================================
    # 🌟 核心升级：高精双阈值联动型流式 Onset 检测核心算法 (零改动对接)
    # =========================================================================
    def _detect_emg_onset_algo(self, time_axis, signal, t_trigger=None, window_sec=2.0, is_emg=True, min_valid_time=None, amp_factor=3.5, persistence_th=0.85, smooth_sigma_ms=15.0, low_th_ratio=1.2, back_track_ms=60.0):
        sig = np.asarray(signal, dtype=float)
        t_arr = np.asarray(time_axis, dtype=float)

        if len(sig) < 100 or t_trigger is None:
            return None, None

        dt = np.median(np.diff(t_arr))
        if dt <= 0: return None, None
        fs = 1.0 / dt

        # 1. 引入 1D 高斯平滑对包络线进行二次无损波形整流
        from scipy.ndimage import gaussian_filter1d
        sigma_pts = max(1, int((smooth_sigma_ms / 1000.0) * fs))
        sig_smoothed = gaussian_filter1d(sig, sigma=sigma_pts)

        # 2. 锁定扰动触发前纯净的 300ms 窗口提取局部基线分布
        baseline_mask = (t_arr < t_trigger) & (t_arr >= t_trigger - 0.4)
        if np.sum(baseline_mask) > 10:
            baseline = sig_smoothed[baseline_mask]
        else:
            baseline = sig_smoothed[:max(int(fs * 0.5), 50)]
            
        base_mean = np.mean(baseline)
        base_std = np.std(baseline)
        if base_std < 1e-4: base_std = 1e-4

        # 3. 双防线参数装配
        if is_emg:
            amp_threshold = base_mean + amp_factor * base_std        # 高门限（决定动作真的发生了）
            low_threshold = base_mean + low_th_ratio * base_std       # 低门限（用于向后追溯原点）
            stable_w = max(int(0.035 * fs), 10)                      # 维持算窗
            persistence_threshold = persistence_th 
        else:
            amp_threshold = base_mean + 4.0 * base_std
            low_threshold = base_mean + 1.5 * base_std
            stable_w = max(int(0.035 * fs), 10)
            persistence_threshold = 0.80

        search_mask = (t_arr >= t_trigger) & (t_arr <= t_trigger + window_sec)
        if not np.any(search_mask): return None, None

        indices = np.where(search_mask)[0]
        back_track_pts = max(1, int((back_track_ms / 1000.0) * fs))

        for idx in indices:
            if min_valid_time is not None and t_arr[idx] < min_valid_time:
                continue

            if idx + stable_w >= len(sig_smoothed):
                continue
            
            # 条件一：平滑信号流跨过高阈值门槛
            if sig_smoothed[idx] > amp_threshold:
                # 条件二：滑窗内维持概率满足 Persistence 限制
                persistence = np.mean(sig_smoothed[idx : idx + stable_w] > (base_mean + 1.2 * base_std))
                
                if persistence > persistence_threshold:
                    # 🚀 Onset 核心逆向回溯机制：沿着当前突破点向前查找，直到曲线跌破低门限
                    onset_idx = idx
                    for back_idx in range(idx, max(0, idx - back_track_pts), -1):
                        if sig_smoothed[back_idx] <= low_threshold:
                            onset_idx = back_idx
                            break
                        # 辅助斜率判断：往回走的过程中曲线已经磨平，即为起始转折点
                        if back_idx < idx and (sig_smoothed[back_idx] - sig_smoothed[back_idx-1]) < (base_std * 0.05):
                            onset_idx = back_idx
                            break
                            
                    return onset_idx, t_arr[onset_idx]
                    
        return None, None

    def _remove_outliers(self, cx, cy, keep_ratio=0.99):
        if len(cx) < 3: return cx, cy
        center_x = np.mean(cx)
        center_y = np.mean(cy)
        dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        threshold = np.percentile(dist, keep_ratio * 100.0)
        mask = dist <= threshold
        return cx[mask], cy[mask]

    # 🌟 修改：扩展函数签名，使其支持由“刷新”按钮调用或由“选择目录”按钮调用
    def _run_biomechanic_analysis(self, choose_new_folder=True):
        if choose_new_folder or not self.target_folder:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Comprehensive Target Evaluation Directory")
            if not folder: return
            self.target_folder = folder
            self.lbl_target.setText(self.target_folder)
        
        folder = self.target_folder
        
        # UI 闭环大核心修复点：触发分析运行的瞬间，暴力抽离并刷新当前界面输入框中的实时自定义数值
        right_map = self.ed_right_map.text().strip().upper()
        left_map = self.ed_left_map.text().strip().upper()
        try: time_diff_ms = float(self.ed_time_diff.text().strip())
        except: time_diff_ms = -50.0
        try: amp_factor = float(self.ed_amp_th.text().strip())
        except: amp_factor = 3.5
        try: persistence_th = float(self.ed_persist_th.text().strip()) 
        except: persistence_th = 0.85
        
        # 🌟 实时捕获并转换 UI 左侧新增的可调 Onset 高级参数
        try: smooth_sigma_ms = float(self.ed_smooth_sigma.text().strip())
        except: smooth_sigma_ms = 15.0
        try: low_th_ratio = float(self.ed_low_th_factor.text().strip())
        except: low_th_ratio = 1.2
        try: back_track_ms = float(self.ed_back_track_max.text().strip())
        except: back_track_ms = 60.0
        
        for m_key, edit in self.mvc_edits.items():
            try:
                self.mvc_values[m_key] = float(edit.text().strip())
                print(f" [UI Sync] Synchronized {m_key} MVC value: {self.mvc_values[m_key]:.2f} uV")
            except:
                pass
                
        self._sync_params()
        
        SEARCH_WINDOW_SEC = 2.0
        
        # Reset display line markers
        for line in self.cop_v_lines: self.p_cop_v.removeItem(line)
        self.cop_v_lines.clear()
        for line in self.emg_onset_lines.values(): self.p_emg.removeItem(line)
        self.emg_onset_lines.clear()
        for line in self.exo_burst_lines:
            self.p_press.removeItem(line)
            self.p_ratio.removeItem(line)
            self.p_exo_deg.removeItem(line)
            self.p_cop_v.removeItem(line)
            self.p_emg.removeItem(line)
        self.exo_burst_lines.clear()
        
        self.c_exo_tgt.setData([])
        self.c_exo_meas.setData([])

        try:
            l_path = os.path.join(folder, "Insole_Left.csv") if os.path.exists(os.path.join(folder, "Insole_Left.csv")) else os.path.join(folder, "Record_Left.csv")
            r_path = os.path.join(folder, "Insole_Right.csv") if os.path.exists(os.path.join(folder, "Insole_Right.csv")) else os.path.join(folder, "Record_Right.csv")
            exo_path = os.path.join(folder, "Exoskeleton_Data.csv")
            emg_files = [f for f in os.listdir(folder) if f.startswith('emg_dev') and f.endswith('.csv') and 'imu' not in f]
            
            start_times = []
            try:
                if os.path.exists(l_path) and os.path.getsize(l_path) > 100:
                    df_tmp = pd.read_csv(l_path, nrows=1)
                    if not df_tmp.empty and 'Unix_Time_s' in df_tmp.columns: start_times.append(df_tmp['Unix_Time_s'].iloc[0])
                if os.path.exists(r_path) and os.path.getsize(r_path) > 100:
                    df_tmp = pd.read_csv(r_path, nrows=1)
                    if not df_tmp.empty and 'Unix_Time_s' in df_tmp.columns: start_times.append(df_tmp['Unix_Time_s'].iloc[0])
                if os.path.exists(exo_path) and os.path.getsize(exo_path) > 100:
                    df_tmp = pd.read_csv(exo_path, nrows=1)
                    if not df_tmp.empty and 'Unix_Time_s' in df_tmp.columns: start_times.append(df_tmp['Unix_Time_s'].iloc[0])
                for filename in emg_files:
                    epath = os.path.join(folder, filename)
                    if os.path.getsize(epath) > 100:
                        df_tmp = pd.read_csv(epath, nrows=1, comment='#')
                        if not df_tmp.empty and 'Unix_Time_s' in df_tmp.columns: start_times.append(df_tmp['Unix_Time_s'].iloc[0])
            except Exception:
                pass

            global_start_time = min(start_times) if start_times else 0
            out_report = ["=== BIOMECHANICAL ANALYTICAL COMPREHENSIVE REPORT ===\n"]

            # Exoskeleton tracking trace loops
            detected_burst_times = []
            if os.path.exists(exo_path) and os.path.getsize(exo_path) > 100:
                try:
                    df_exo = pd.read_csv(exo_path)
                    if not df_exo.empty and len(df_exo) > 1:
                        t_exo_arr = df_exo['Unix_Time_s'].values - global_start_time
                        for i in range(1, len(t_exo_arr)):
                            if t_exo_arr[i] <= t_exo_arr[i-1]: t_exo_arr[i] = t_exo_arr[i-1] + 1e-4
                        
                        y_tgt = df_exo['Target_Position_Deg'].values
                        y_meas = df_exo['Measured_Position_Deg'].values
                        
                        self.c_exo_tgt.setData(t_exo_arr, y_tgt)
                        self.c_exo_meas.setData(t_exo_arr, y_meas)
                        
                        if 'Is_Bursting' in df_exo.columns:
                            is_burst_arr = df_exo['Is_Bursting'].values
                            burst_idxes = np.where((is_burst_arr[:-1] == 0) & (is_burst_arr[1:] == 1))[0] + 1
                            for b_idx in burst_idxes:
                                t_burst_moment = t_exo_arr[b_idx]
                                detected_burst_times.append(t_burst_moment)
                                
                                out_report.append(f"🔥 [Exoskeleton Perturbation Captured] (Time: {t_burst_moment:.3f} s):")
                                out_report.append(f"   -> Target Position Command: {y_tgt[b_idx]:.2f} Deg")
                                out_report.append(f"   -> Measured Feedback Position: {y_meas[b_idx]:.2f} Deg\n")
                except Exception as ex:
                    out_report.append(f"⚠️ Failed to parse exoskeleton data link: {ex}\n")

            t_anchor = detected_burst_times[0] if (detected_burst_times and len(detected_burst_times) > 0) else None
            if t_anchor is not None:
                out_report.append(f"🎯 Exoskeleton event reference locked: {t_anchor:.3f}s. Active segment bounded to +{SEARCH_WINDOW_SEC}s.")
            else:
                out_report.append("ℹ️ No active exoskeleton perturbation detected; switching to default macro blind scanning.")

            sei_calc_box = {"areas": {}, "iemg_sum": 0.0}

            # Ground reaction forces tracking 
            if os.path.exists(l_path) and os.path.exists(r_path) and os.path.getsize(l_path) > 100:
                try:
                    df_l = pd.read_csv(l_path)
                    df_r = pd.read_csv(r_path)
                    if not df_l.empty and not df_r.empty and len(df_l) > 2:
                        p_cols_l = [c for c in df_l.columns if c.startswith('P_')]
                        p_cols_r = [c for c in df_r.columns if c.startswith('P_')]
                        min_len = min(len(df_l), len(df_r))
                        
                        t_arr = df_l['Unix_Time_s'].values[:min_len] - global_start_time
                        for i in range(1, len(t_arr)):
                            if t_arr[i] <= t_arr[i-1]: t_arr[i] = t_arr[i-1] + 1e-4
                        
                        z_l = self.zero_offset_l if self.zero_offset_l is not None else np.full(len(p_cols_l), 1025.0)
                        z_r = self.zero_offset_r if self.zero_offset_r is not None else np.full(len(p_cols_r), 1025.0)
                        
                        mat_l = np.clip(df_l[p_cols_l].values[:min_len] - z_l, 0, None) * self.scale_l
                        mat_r = np.clip(df_r[p_cols_r].values[:min_len] - z_r, 0, None) * self.scale_r
                        sum_l, sum_r = np.sum(mat_l, axis=1), np.sum(mat_r, axis=1)
                        sum_tot = sum_l + sum_r
                        
                        self.c_press_tot.setData(t_arr, sum_tot)
                        self.c_press_l.setData(t_arr, sum_l)
                        self.c_press_r.setData(t_arr, sum_r)
                        
                        safe_tot = np.where(sum_tot < 1, 1, sum_tot)
                        ratio_l = (sum_l / safe_tot) * 100.0
                        ratio_r = (sum_r / safe_tot) * 100.0
                        self.c_ratio_l.setData(t_arr, ratio_l)
                        self.c_ratio_r.setData(t_arr, ratio_r)

                        if t_anchor is not None:
                            post_3s_mask = (t_arr >= t_anchor) & (t_arr <= (t_anchor + 3.0))
                            if np.any(post_3s_mask):
                                t_post_3s = t_arr[post_3s_mask]
                                ratio_l_post = ratio_l[post_3s_mask]
                                
                                max_ratio_idx = np.argmax(ratio_l_post)
                                t_peak_ratio_l = t_post_3s[max_ratio_idx]
                                peak_val_ratio_l = ratio_l_post[max_ratio_idx]
                                ttp_ms = (t_peak_ratio_l - t_anchor) * 1000.0
                                
                                # 🌟 核心改进：截取扰动前一段时间（t_anchor 前 0.5s）的平均占比作为健康基准线
                                pre_mask = (t_arr < t_anchor) & (t_arr >= t_anchor - 0.5)
                                if np.sum(pre_mask) < 5:
                                    pre_mask = (t_arr < t_anchor)
                                    
                                if np.any(pre_mask):
                                    mean_ratio_l_pre = np.mean(ratio_l[pre_mask])
                                    delta_ratio_l = peak_val_ratio_l - mean_ratio_l_pre
                                    pre_baseline_str = f"| Pre-Perturb Avg: {mean_ratio_l_pre:.1f} % | Delta: {delta_ratio_l:+.1f} %"
                                else:
                                    pre_baseline_str = "| Pre-Perturb Avg: N/A | Delta: N/A"
                                
                                out_report.append(f"⚖️ [Post-Perturbation Bilateral Balance Parameters - 3.0s Window]:")
                                out_report.append(f"   -> Left Peak Ratio Time: {t_peak_ratio_l:.3f} s | Value: {peak_val_ratio_l:.1f} % {pre_baseline_str} | Time-to-Peak (TTP): {ttp_ms:.1f} ms\n")

                        # 动力学测算保持底层签名兼容
                        _, t_l_force_onset = self._detect_emg_onset_algo(t_arr, sum_l, t_trigger=t_anchor, window_sec=2.0, is_emg=False, smooth_sigma_ms=smooth_sigma_ms, low_th_ratio=low_th_ratio, back_track_ms=back_track_ms)
                        _, t_r_force_onset = self._detect_emg_onset_algo(t_arr, sum_r, t_trigger=t_anchor, window_sec=2.0, is_emg=False, smooth_sigma_ms=smooth_sigma_ms, low_th_ratio=low_th_ratio, back_track_ms=back_track_ms)
                        if t_l_force_onset is not None or t_r_force_onset is not None:
                            out_report.append(f"👣 [Plantar Force Mechanical Impulse Response Latencies]:")
                            if t_l_force_onset is not None:
                                out_report.append(f"   -> Left Vertical Pressure Activation Delay: {(t_l_force_onset - t_anchor)*1000:.1f} ms")
                            if t_r_force_onset is not None:
                                out_report.append(f"   -> Right Vertical Pressure Activation Delay: {(t_r_force_onset - t_anchor)*1000:.1f} ms\n")
                        
                        global_coords_l = self.coords_map_l.copy()
                        global_coords_r = self.coords_map_r.copy()
                        STANCE_WIDTH = 150.0
                        global_coords_r[:, 0] += np.max(global_coords_l[:, 0]) + STANCE_WIDTH
                        
                        self.bg_spots_l.setData(global_coords_l[:, 0], global_coords_l[:, 1])
                        self.bg_spots_r.setData(global_coords_r[:, 0], global_coords_r[:, 1])
                        
                        def get_clean_cop(matrix, coords, weights):
                            cx = np.dot(matrix, coords[:, 0]) / (weights + 1e-6)
                            cy = np.dot(matrix, coords[:, 1]) / (weights + 1e-6)
                            cx_f = pd.Series(cx).interpolate().bfill().ffill().values
                            cy_f = pd.Series(cy).interpolate().bfill().ffill().values
                            return cx_f, cy_f

                        cop_x_l, cop_y_l = get_clean_cop(mat_l, global_coords_l, sum_l)
                        cop_x_r, cop_y_r = get_clean_cop(mat_r, global_coords_r, sum_r)
                        
                        cop_x_g_real = (np.dot(mat_l, global_coords_l[:, 0]) + np.dot(mat_r, global_coords_r[:, 0])) / (sum_tot + 1e-6)
                        cop_y_g_real = (np.dot(mat_l, global_coords_l[:, 1]) + np.dot(mat_r, global_coords_r[:, 1])) / (sum_tot + 1e-6)
                        cop_x_g, cop_y_g = pd.Series(cop_x_g_real).interpolate().bfill().ffill().values, pd.Series(cop_y_g_real).interpolate().bfill().ffill().values
                        
                        v_channels = {
                            "Left": (cop_x_l, cop_y_l, self.c_v_l, self.c_cop_l_line, self.curve_hull_l, "Left Insole CoP", '#00D4FF'),
                            "Right": (cop_x_r, cop_y_r, self.c_v_r, self.c_cop_r_line, self.curve_hull_r, "Right Insole CoP", '#FF7800'),
                            "Global": (cop_x_g, cop_y_g, self.c_v_g, self.c_cop_g_line, self.curve_hull_g, "Global Sway Trajectory", '#D400FF')
                        }
                        
                        for ch_lbl, (cx, cy, curve_v, plot_line, curve_hull, report_lbl, v_color) in v_channels.items():
                            dt_median = np.median(np.diff(t_arr)) if len(t_arr) > 1 else 0.01
                            fs_est = 1.0 / dt_median if dt_median > 0 else 100.0
                            sg_win = int(round(fs_est * 210 / 1000))
                            if sg_win % 2 == 0: sg_win += 1
                            sg_win = max(sg_win, 5)

                            if len(cx) >= sg_win:
                                cx_smoothed = savgol_filter(cx, sg_win, 3)
                                cy_smoothed = savgol_filter(cy, sg_win, 3)
                            else:
                                cx_smoothed, cy_smoothed = cx, cy
                                
                            dx = np.gradient(cx_smoothed)
                            dy = np.gradient(cy_smoothed)
                            dt = np.gradient(t_arr)
                            dt[dt < 0.005] = 0.005
                            
                            raw_speed = np.sqrt(dx**2 + dy**2) / dt
                            speed = pd.Series(raw_speed).rolling(window=10, min_periods=1, center=True).mean().values
                            curve_v.setData(t_arr, speed)
                            
                            _, onset_t = self._detect_emg_onset_algo(t_arr, speed, t_trigger=t_anchor, window_sec=2.0, is_emg=False, smooth_sigma_ms=smooth_sigma_ms, low_th_ratio=low_th_ratio, back_track_ms=back_track_ms)
                            if onset_t is None:
                                onset_t = t_anchor if t_anchor is not None else t_arr[0]
                            
                            v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(v_color, style=QtCore.Qt.DashLine, width=1.8))
                            v_line.setValue(onset_t)
                            self.p_cop_v.addItem(v_line)
                            self.cop_v_lines.append(v_line)
                            
                            hull_anchor = t_anchor if t_anchor is not None else t_arr[0]
                            slice_mask = (t_arr >= hull_anchor) & (t_arr <= (hull_anchor + 3.0))
                            
                            plot_line.setData(cx_smoothed[slice_mask], cy_smoothed[slice_mask])
                            cx_slice, cy_slice = cx_smoothed[slice_mask], cy_smoothed[slice_mask]
                            
                            area, path_length, mean_v, peak_v, max_disp = 0.0, 0.0, 0.0, 0.0, 0.0
                            
                            if len(cx_slice) >= 3:
                                cx_area, cy_area = self._remove_outliers(cx_slice, cy_slice, keep_ratio=0.99)
                                
                                if len(cx_area) >= 3 and ConvexHull is not None:
                                    try:
                                        pts_slice = np.column_stack((cx_area, cy_area))
                                        hull = ConvexHull(pts_slice)
                                        area = hull.volume  
                                        h_pts = hull.points[hull.vertices]
                                        
                                        if ch_lbl == "Global":
                                            curve_hull.setData(h_pts[:, 0].tolist() + [h_pts[0, 0]], h_pts[:, 1].tolist() + [h_pts[0, 1]])
                                        else:
                                            curve_hull.setData([], [])
                                    except: 
                                        curve_hull.setData([], [])
                                
                                dx_slice, dy_slice = np.diff(cx_slice), np.diff(cy_slice)
                                path_length = np.sum(np.sqrt(dx_slice**2 + dy_slice**2))
                                peak_v = np.max(speed[slice_mask])
                                mean_v = np.mean(speed[slice_mask])
                                
                                base_indices = np.where(t_arr >= hull_anchor)[0]
                                base_idx = base_indices[0] if len(base_indices) > 0 else 0
                                max_disp = np.max(np.sqrt((cx_slice - cx_smoothed[base_idx])**2 + (cy_slice - cy_smoothed[base_idx])**2))
                            else:
                                curve_hull.setData([], [])
                            
                            sei_calc_box["areas"][ch_lbl] = area

                            out_report.append(f"[{report_lbl}] (CoP Anchor Frame: {hull_anchor:.3f} s):")
                            if t_anchor is not None:
                                out_report.append(f"   -> CoP Velocity Inflexion Activation Latency (Δt): {(onset_t - t_anchor)*1000:.1f} ms")
                            out_report.append(f"   -> Posture Sway Convex Area (Area) : {area:.2f} mm²")
                            out_report.append(f"   -> Trajectory Total Path Length (Path Len) : {path_length:.2f} mm")
                            out_report.append(f"   -> Posture Control Average Velocity (Mean V): {mean_v:.2f} mm/s")
                            out_report.append(f"   -> Instability Transient Peak Velocity (Peak V): {peak_v:.2f} mm/s")
                            out_report.append(f"   -> Maximal Displacement Amplitude (Max Disp) : {max_disp:.2f} mm\n")
                except Exception as p_ex:
                    out_report.append(f"⚠️ Kinetic processing error layout trace: {p_ex}")
            else:
                out_report.append("ℹ️ Insole matrix data unavailable. Kinetics tracking ignored.")

            # Surface EMG multi-channel tracking payload
            emg_active = False
            if emg_files:
                try:
                    if hasattr(self, 'p_emg'):
                        self.p_emg.clear()
                    color_pool = ['#00FF00', '#00AAFF', '#FF55FF', '#FFFF00', '#FF5555']
                    c_idx = 0
                    out_report.append("⚡ [EMG Activation Latencies & Workload Indexes]:")
                    
                    emg_data_dict = {}
                    for filename in emg_files:
                        dev_name = filename.replace('.csv', '').upper().replace('EMG_', '').strip()
                        epath = os.path.join(folder, filename)
                        if os.path.getsize(epath) < 100: continue
                        
                        df_e = pd.read_csv(epath, comment='#')
                        if not df_e.empty and len(df_e) > 2:
                            t_emg = df_e['Unix_Time_s'].values - global_start_time
                            emg_active = True
                            
                            for col in [c for c in df_e.columns if '_env_' in c.lower()]:
                                ch_tag = "CH1" if 'ch1' in col.lower() else "CH2"
                                m_key = f"{dev_name}_{ch_tag}".strip()
                                
                                if m_key not in self.mvc_values:
                                    mvc_base = 1000.0
                                else:
                                    mvc_base = self.mvc_values[m_key]
                                
                                norm_pct = (df_e[col].values / mvc_base) * 100.0
                                norm_pct = np.clip(norm_pct, 0.0, 150.0)
                                
                                emg_data_dict[m_key] = {
                                    "t_emg": t_emg,
                                    "norm_pct": norm_pct,
                                    "raw_signal": df_e[col].values
                                }

                    # Sequence array prioritized execution filters
                    eval_keys = []
                    if right_map in emg_data_dict: eval_keys.append(right_map)
                    if left_map in emg_data_dict: eval_keys.append(left_map)
                    for k in emg_data_dict.keys():
                        if k not in eval_keys: eval_keys.append(k)

                    t_onset_r = None
                    t_onset_l = None

                    # 💯 保留你原汁原味的多通道遍历映射逻辑，绝不吞掉任何一路信号
                    for m_key in eval_keys:
                        d = emg_data_dict[m_key]
                        
                        color = color_pool[c_idx % len(color_pool)]
                        curve = self.p_emg.plot(pen=pg.mkPen(color, width=1.8), name=f"{m_key} %MVC")
                        curve.setData(d["t_emg"], d["norm_pct"])
                        c_idx += 1
                        
                        curr_min_valid = None
                        if m_key == left_map and t_onset_r is not None:
                            curr_min_valid = t_onset_r + (time_diff_ms / 1000.0)
                        
                        # 注入升级后的自适应双阈值流式重构核心，将 UI 全局控件里的系数值完美动态下沉闭环
                        o_idx, o_time_emg = self._detect_emg_onset_algo(
                            d["t_emg"], 
                            d["raw_signal"], 
                            t_trigger=t_anchor, 
                            window_sec=SEARCH_WINDOW_SEC,
                            is_emg=True, 
                            min_valid_time=curr_min_valid,
                            amp_factor=amp_factor,
                            persistence_th=persistence_th,
                            smooth_sigma_ms=smooth_sigma_ms,
                            low_th_ratio=low_th_ratio,
                            back_track_ms=back_track_ms
                        )
                        
                        if m_key == right_map: t_onset_r = o_time_emg
                        if m_key == left_map: t_onset_l = o_time_emg
                        
                        if t_anchor is not None:
                            emg_win_mask = (d["t_emg"] >= t_anchor) & (d["t_emg"] <= (t_anchor + SEARCH_WINDOW_SEC))
                            if np.any(emg_win_mask):
                                single_iemg = float(np.trapz(d["norm_pct"][emg_win_mask], d["t_emg"][emg_win_mask]))
                                sei_calc_box["iemg_sum"] += single_iemg

                        if o_time_emg is not None:
                            out_report.append(f"   -> Muscle [{m_key}] Onset Burst Time: {o_time_emg:.3f} s")
                            if t_anchor is not None:
                                out_report.append(f"      ⏱ Perturbation to Muscle Activation Delay (Δt): {(o_time_emg - t_anchor)*1000:.1f} ms")
                            o_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color, style=QtCore.Qt.DashLine, width=1.5))
                            o_line.setValue(o_time_emg)
                            self.p_emg.addItem(o_line)
                        else:
                            out_report.append(f"   -> Muscle [{m_key}] No significant activation found.")
                            
                    if t_onset_r is not None and t_onset_l is not None:
                        diff_ms = (t_onset_l - t_onset_r) * 1000.0
                        out_report.append(f"\n   🌟 [Bilateral Synchronization]: Left ({left_map}) vs. Right ({right_map}) Activation Delta (L-R): {diff_ms:.1f} ms")

                except Exception as emg_ex:
                    out_report.append(f"⚠️ Electromyography file parser execution fault: {emg_ex}")
            
            if not emg_active:
                out_report.append("ℹ️ EMG metrics files skipped or missing.")

            # Performance efficiency cross indicators (SEI Engine)
            if t_anchor is not None and sei_calc_box.get("iemg_sum", 0.0) > 0 and "Global" in sei_calc_box["areas"]:
                out_report.append("🏅 [System Postural Efficiency Indexes - SEI Summary]:")
                g_area = sei_calc_box["areas"].get("Global", 0.0)
                l_area = sei_calc_box["areas"].get("Left", 0.0)
                r_area = sei_calc_box["areas"].get("Right", 0.0)
                total_iemg = sei_calc_box["iemg_sum"]
                
                sei_global = 1.0 / (g_area * total_iemg + 1e-6) if g_area > 0 else 0.0
                sei_left = 1.0 / (l_area * total_iemg + 1e-6) if l_area > 0 else 0.0
                sei_right = 1.0 / (r_area * total_iemg + 1e-6) if r_area > 0 else 0.0
                
                out_report.append(f"   -> Integrated Electromyography Burden (iEMG): {total_iemg:.2f} %MVC·s")
                out_report.append(f"   -> ⚖️ Postural Stability Efficiency Index (SEI Left) : {sei_left * 1000:.4f} (×10³)")
                out_report.append(f"   -> ⚖️ Postural Stability Efficiency Index (SEI Right) : {sei_right * 1000:.4f} (×10³)")
                out_report.append(f"   -> 🏆 Global Postural Control Index Balance (SEI Global): {sei_global * 1000:.4f} (×10³)")
                out_report.append("   (Note: Higher SEI signifies superior neuromuscular cost efficiency vs. stabilization output.)\n")

            # Draw perturbation dynamic event indicator vertical lines
            for t_burst in detected_burst_times:
                for target_plot in [self.p_press, self.p_ratio, self.p_exo_deg, self.p_cop_v, self.p_emg]:
                    cross_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FF55FF', style=QtCore.Qt.SolidLine, width=2.0))
                    cross_line.setValue(t_burst)
                    target_plot.addItem(cross_line)
                    self.exo_burst_lines.append(cross_line)

            self.txt_metrics.setPlainText("\n".join(out_report))
            self.p_press.autoRange()
            self.p_exo_deg.autoRange()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Analysis Faulted", f"Data analytics environment exception:\n{e}")
