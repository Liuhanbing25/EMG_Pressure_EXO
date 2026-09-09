#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import socket
import struct
import csv
import math
import json
import datetime
from collections import deque
import numpy as np

# Configuration constants
UDP_PORT = 44444

SIZE_CONFIGS = {
    "42": {"num_points": 233},
    "38": {"num_points": 192},
}

def calc_packet_size(num_points: int) -> int:
    return 2 + (num_points * 2 + 6 + 52 + 52) + 2

PACKET_SIZE_TO_SIZE = {calc_packet_size(cfg["num_points"]): size for size, cfg in SIZE_CONFIGS.items()}
NUM_POINTS_TO_SIZE = {cfg["num_points"]: size for size, cfg in SIZE_CONFIGS.items()}

MAG_AXIS_MODE = "FLIP_Y" 
_MAG_SIGN_MAP = {
    "NONE": (1, 1, 1), "FLIP_X": (-1, 1, 1), "FLIP_Y": (1, 1, 1), "FLIP_XY": (-1, -1, 1),
}
MAG_SIGN = _MAG_SIGN_MAP.get(MAG_AXIS_MODE, (1, 1, 1))


# Math helper functions and filter classes
class SignalFilter:
    def __init__(self, median_window: int = 7, ema_alpha: float = 0.08):
        self.median_window = int(median_window)
        self.buffer = deque(maxlen=self.median_window)
        self.ema_alpha = float(ema_alpha)
        self.ema_value = None

    def update(self, val: float) -> float:
        self.buffer.append(val)
        if len(self.buffer) < self.median_window:
            return float(sum(self.buffer) / len(self.buffer))
        sorted_vals = sorted(self.buffer)
        median_val = float(sorted_vals[len(sorted_vals) // 2])
        if self.ema_value is None: self.ema_value = median_val
        else: self.ema_value = (self.ema_alpha * median_val) + ((1 - self.ema_alpha) * self.ema_value)
        return float(self.ema_value)

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def quat_normalize(q):
    n = math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    if n < 1e-12: return [1.0, 0.0, 0.0, 0.0]
    return [q[0]/n, q[1]/n, q[2]/n, q[3]/n]

def quat_to_euler_deg(q):
    w, x, y, z = q
    sinr_cosp = 2.0*(w*x + y*z)
    cosr_cosp = 1.0 - 2.0*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0*(w*y - z*x)
    pitch = math.asin(_clamp(sinp, -1.0, 1.0))
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

class MadgwickAHRS:
    def __init__(self, sample_freq=100.0, beta=0.08):
        self.sample_freq = float(sample_freq)
        self.beta = float(beta)
        self.q = [1.0, 0.0, 0.0, 0.0]

    def reset(self, q_init=None):
        if q_init is None: self.q = [1.0, 0.0, 0.0, 0.0]
        else: self.q = quat_normalize([float(q_init[0]), float(q_init[1]), float(q_init[2]), float(q_init[3])])

    def update_imu(self, gx, gy, gz, ax, ay, az):
        q1, q2, q3, q4 = self.q
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm < 1e-12: return
        ax, ay, az = ax/norm, ay/norm, az/norm
        _2q1, _2q2, _2q3, _2q4 = 2.0*q1, 2.0*q2, 2.0*q3, 2.0*q4
        _4q1, _4q2, _4q3 = 4.0*q1, 4.0*q2, 4.0*q3
        _8q2, _8q3 = 8.0*q2, 8.0*q3
        q1q1, q2q2, q3q3, q4q4 = q1*q1, q2*q2, q3*q3, q4*q4
        s1 = _4q1*q3q3 + _2q3*ax + _4q1*q2q2 - _2q2*ay
        s2 = _4q2*q4q4 - _2q4*ax + 4.0*q1q1*q2 - _2q1*ay - _4q2 + _8q2*q2q2 + _8q2*q3q3 + _4q2*az
        s3 = 4.0*q1q1*q3 + _2q1*ax + _4q3*q4q4 - _2q4*ay - _4q3 + _8q3*q2q2 + _8q3*q3q3 + _4q3*az
        s4 = 4.0*q2q2*q4 - _2q2*ax + 4.0*q3q3*q4 - _2q3*ay
        norm_s = math.sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4)
        if norm_s < 1e-12: return
        s1, s2, s3, s4 = s1/norm_s, s2/norm_s, s3/norm_s, s4/norm_s
        qDot1 = 0.5 * (-q2*gx - q3*gy - q4*gz) - self.beta*s1
        qDot2 = 0.5 * ( q1*gx + q3*gz - q4*gy) - self.beta*s2
        qDot3 = 0.5 * ( q1*gy - q2*gz + q4*gx) - self.beta*s3
        qDot4 = 0.5 * ( q1*gz + q2*gy - q3*gx) - self.beta*s4
        dt = 1.0 / self.sample_freq
        self.q = quat_normalize([q1 + qDot1*dt, q2 + qDot2*dt, q3 + qDot3*dt, q4 + qDot4*dt])

    def update(self, gx, gy, gz, ax, ay, az, mx, my, mz):
        self.update_imu(gx, gy, gz, ax, ay, az)


# Single insole headless device manager
class HeadlessInsole:
    def __init__(self, side: str):
        self.side = side 
        self.size_label = "AUTO"
        self.num_points = 0
        
        self.filter_f1 = SignalFilter()
        self.filter_f2 = SignalFilter()
        
        self.fusion = {
            'imu1': MadgwickAHRS(sample_freq=100.0, beta=0.08),
            'imu2': MadgwickAHRS(sample_freq=100.0, beta=0.08),
        }
        self.fusion_initialized = {'imu1': False, 'imu2': False}
        self.mag_calib = {'imu1': {'offset': [0,0,0], 'scale': [1,1,1]}, 'imu2': {'offset': [0,0,0], 'scale': [1,1,1]}}
        self._load_mag_calibration()

        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.record_unix_start = 0.0
        self.gui_snapshot = None

    def _load_mag_calibration(self):
        calib_file = f"mag_calib_{self.side}.json"
        if os.path.exists(calib_file):
            try:
                with open(calib_file, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                for k in ['imu1', 'imu2']:
                    if k in d:
                        self.mag_calib[k]['offset'] = [float(x) for x in d[k].get('offset', [0,0,0])]
                        self.mag_calib[k]['scale'] = [float(x) for x in d[k].get('scale', [1,1,1])]
            except Exception: pass

    def start_recording(self, session_id: str, record_start_unix: float):
        if self.is_recording: return
        
        save_dir = f"test/{session_id}"
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{save_dir}/Insole_{self.side}.csv"
        
        try:
            self.csv_file = open(fname, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.record_unix_start = record_start_unix
            
            h = ['Unix_Time_s', 'RelativeTime_s'] + [f'P_{i + 1}' for i in range(self.num_points)] + ['Flex1', 'Flex2', 'Bat']
            cols = ['RefTime', 'Ax', 'Ay', 'Az', 'Wx', 'Wy', 'Wz', 'Roll', 'Pitch', 'Yaw', 'Hx', 'Hy', 'Hz', 'Q0', 'Q1', 'Q2', 'Q3']
            for s in ['IMU1', 'IMU2']: h += [f'{s}_{c}' for c in cols]
            
            cols9 = ['FusedRoll', 'FusedPitch', 'FusedYaw', 'FusedQ0', 'FusedQ1', 'FusedQ2', 'FusedQ3', 'MagValid']
            for s in ['IMU1', 'IMU2']: h += [f'{s}_{c}' for c in cols9]
                
            self.csv_writer.writerow(h)
            self.is_recording = True
        except Exception as e:
            print(f"[Insole Backend] Failed to create CSV for {self.side}: {e}")

    def stop_recording(self):
        self.is_recording = False
        if self.csv_file:
            try: self.csv_file.flush(); self.csv_file.close()
            except Exception: pass
            self.csv_file = None

    def process_packet(self, press_vals, flex1, flex2, bat, imu1_data, imu2_data):
        smooth_f1 = int(self.filter_f1.update(flex1))
        smooth_f2 = int(self.filter_f2.update(flex2))
        
        imus = {'imu1': imu1_data, 'imu2': imu2_data}
        for k in ['imu1', 'imu2']:
            imu = imus[k]
            if not self.fusion_initialized[k] and len(imu['quat']) == 4:
                self.fusion[k].reset(imu['quat'])
                self.fusion_initialized[k] = True
            
            ax, ay, az = imu['acc']
            gx, gy, gz = [x * math.pi / 180.0 for x in imu['gyro']]
            
            mx, my, mz = imu['mag']
            off = self.mag_calib[k]['offset']
            scl = self.mag_calib[k]['scale']
            mx, my, mz = (mx-off[0])*scl[0], (my-off[1])*scl[1], (mz-off[2])*scl[2]
            
            self.fusion[k].update(gx, gy, gz, ax, ay, az, mx, my, mz)
            
            q9 = self.fusion[k].q
            imu['quat9'] = q9
            imu['angle9'] = quat_to_euler_deg(q9)

        current_unix = time.time()
        
        if self.is_recording and self.csv_writer:
            t_rel = current_unix - self.record_unix_start
            row = [f"{current_unix:.6f}", f"{t_rel:.3f}"]
            row.extend(press_vals.tolist())
            row.extend([smooth_f1, smooth_f2, bat])
            
            for imu in [imu1_data, imu2_data]:
                row.append(imu['time_str'])
                row.extend(imu['acc']); row.extend(imu['gyro']); row.extend(imu['angle'])
                row.extend(imu['mag']); row.extend(imu['quat'])
                row.extend(imu['angle9']); row.extend(imu['quat9']); row.append(1)
                
            self.csv_writer.writerow(row)
            
        self.gui_snapshot = {
            'side': self.side,
            'size': self.size_label,
            'press': press_vals.tolist(), 
            'bat': bat,
            'flex': [smooth_f1, smooth_f2],
            'imu_front': imus['imu1']['angle9'],
            'imu_heel': imus['imu2']['angle9']
        }


# Main worker process loop
def insole_worker_process(shared_state, insole_data_queue):
    print("[Insole Backend] Process started. Listening on UDP port 44444.")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    try:
        sock.bind(('0.0.0.0', UDP_PORT))
        sock.settimeout(0.5)
    except Exception as e:
        print(f"[Insole Backend] Port bind failed: {e}")
        return

    HEADER_LEFT = b'\xFA\xAA'
    HEADER_RIGHT = b'\xFA\xBB'
    TAIL = b'\xED\x0F'

    panel_l = HeadlessInsole("Left")
    panel_r = HeadlessInsole("Right")
    
    last_gui_push_time = time.time()
    is_global_recording = False

    while not shared_state.get('cmd_stop_all', False):
        target_record_state = shared_state.get('is_recording', False)
        if target_record_state != is_global_recording:
            is_global_recording = target_record_state
            if is_global_recording:
                sid = shared_state.get('session_id', 'default_session')
                sunix = shared_state.get('record_start_unix', time.time())
                panel_l.start_recording(sid, sunix)
                panel_r.start_recording(sid, sunix)
            else:
                panel_l.stop_recording()
                panel_r.stop_recording()

        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except Exception:
            break

        packet_len = len(data)
        size_label = PACKET_SIZE_TO_SIZE.get(packet_len)
        
        if size_label is None or data[-2:] != TAIL:
            continue
            
        is_left = data.startswith(HEADER_LEFT)
        is_right = data.startswith(HEADER_RIGHT)
        if not (is_left or is_right): continue

        num_points = int(SIZE_CONFIGS[size_label]["num_points"])
        payload = data[2:-2]
        
        try:
            press_bytes = payload[0: num_points * 2]
            press_vals = np.frombuffer(press_bytes, dtype='>u2').astype(np.float32)

            offset = num_points * 2
            flex1 = struct.unpack(">H", payload[offset:offset + 2])[0]
            flex2 = struct.unpack(">H", payload[offset + 2:offset + 4])[0]
            bat = struct.unpack(">H", payload[offset + 4:offset + 6])[0]
            offset += 6

            def parse_imu(db: bytes):
                try:
                    hh, mi, ss, ms = db[3], db[4], db[5], (db[7] << 8) | db[6]
                    time_str = f"\t{hh:02d}:{mi:02d}:{ss:02d}.{ms:03d}"
                except Exception: time_str = "\t00:00:00.000"

                acc = struct.unpack("<hhh", db[8:14])
                gyro = struct.unpack("<hhh", db[14:20])
                angle = struct.unpack("<hhh", db[20:26])
                mag = struct.unpack("<hhh", db[26:32])
                mag = (int(mag[0] * MAG_SIGN[0]), int(mag[1] * MAG_SIGN[1]), int(mag[2] * MAG_SIGN[2]))
                quat = struct.unpack("<hhhh", db[32:40])

                f_a, f_w, f_ang, f_q = 16.0 / 32768, 2000.0 / 32768, 180.0 / 32768, 1.0 / 32768
                return {
                    'time_str': time_str,
                    'acc': [x * f_a for x in acc],
                    'gyro': [x * f_w for x in gyro],
                    'angle': [x * f_ang for x in angle],
                    'mag': [float(mag[0]), float(mag[1]), float(mag[2])],
                    'quat': [x * f_q for x in quat]
                }

            imu1_data = parse_imu(payload[offset: offset + 52])
            imu2_data = parse_imu(payload[offset + 52: offset + 104])
            
            target_panel = panel_l if is_left else panel_r
            target_panel.size_label = size_label
            target_panel.num_points = num_points
            
            target_panel.process_packet(press_vals, flex1, flex2, bat, imu1_data, imu2_data)
            
        except Exception:
            continue

        now = time.time()
        if now - last_gui_push_time >= 0.033:
            payloads = []
            if panel_l.gui_snapshot: payloads.append(panel_l.gui_snapshot)
            if panel_r.gui_snapshot: payloads.append(panel_r.gui_snapshot)
            
            if payloads:
                import queue
                try: insole_data_queue.put_nowait(payloads)
                except queue.Full: pass
            
            last_gui_push_time = now

    sock.close()
    panel_l.stop_recording()
    panel_r.stop_recording()
