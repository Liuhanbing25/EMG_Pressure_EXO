#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import socket
import struct
import threading
import csv
import datetime
from collections import deque
import numpy as np

# Constants and signal processing configurations
DISCOVERY_PORT = 3334
DISCOVERY_MAGIC = b"EMG_DISCOVER_V1"
DISCOVERY_REPLY_PREFIX = "EMG_HERE_V1"
VREF_V = 2.418  
ADC_FULL_SCALE = float(1 << 23)

def ADS1292R_uV_per_count(gain: int) -> float:
    g = int(gain) if gain > 0 else 1
    return (VREF_V * 1e6) / (float(g) * ADC_FULL_SCALE)

class Biquad:
    __slots__ = ("b0", "b1", "b2", "a1", "a2", "z1", "z2")
    def __init__(self, b0, b1, b2, a1, a2):
        self.b0 = float(b0); self.b1 = float(b1); self.b2 = float(b2)
        self.a1 = float(a1); self.a2 = float(a2)
        self.z1 = 0.0; self.z2 = 0.0

    def reset(self):
        self.z1 = 0.0; self.z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float64)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        z1, z2 = self.z1, self.z2
        for i in range(x.size):
            xi = float(x[i])
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi
        self.z1, self.z2 = z1, z2
        return y

def _norm_f0(fs, f0): return max(1e-6, min(0.499, f0 / fs)) if fs > 0 else 0.0

def design_hpf(fs, fc, q=0.707):
    w0 = 2 * math.pi * _norm_f0(fs, fc); alpha = math.sin(w0) / (2 * q); cosw = math.cos(w0)
    a0 = 1 + alpha
    return Biquad((1 + cosw) / 2 / a0, -(1 + cosw) / a0, (1 + cosw) / 2 / a0, -2 * cosw / a0, (1 - alpha) / a0)

def design_lpf(fs, fc, q=0.707):
    w0 = 2 * math.pi * _norm_f0(fs, fc); alpha = math.sin(w0) / (2 * q); cosw = math.cos(w0)
    a0 = 1 + alpha
    return Biquad((1 - cosw) / 2 / a0, (1 - cosw) / a0, (1 - cosw) / 2 / a0, -2 * cosw / a0, (1 - alpha) / a0)

def design_notch(fs, f0, q=30.0):
    w0 = 2 * math.pi * _norm_f0(fs, f0); alpha = math.sin(w0) / (2 * q); cosw = math.cos(w0)
    a0 = 1 + alpha
    return Biquad(1 / a0, -2 * cosw / a0, 1 / a0, -2 * cosw / a0, (1 - alpha) / a0)


# Network TCP receiver
class Receiver:
    PKT_EMG = ord('E'); PKT_IMU = ord('I'); PKT_BAT = ord('B')

    def __init__(self):
        self.sock = None
        self.buf = bytearray()
        self.running = False
        self.q_emg = deque()
        self.q_imu = deque()
        self.q_bat = deque()
        self.connected = False
        self._rx_thread = None
        self._stop_evt = threading.Event()
        self._tx_lock = threading.Lock()

    def connect_to(self, host: str, port: int) -> bool:
        self.disconnect()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((host, int(port)))
            s.settimeout(0.25)
            self.sock = s
            self.connected = True
            self.running = True
            self.buf.clear()
            self._stop_evt.clear()
            th = threading.Thread(target=self._rx_loop, args=(s,), daemon=True)
            self._rx_thread = th
            th.start()
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        self.running = False; self.connected = False
        self._stop_evt.set()
        if self.sock:
            try: self.sock.close()
            except Exception: pass
        self.sock = None

    def send_line(self, line: str) -> None:
        if not line or not self.sock: return
        data = (line + "\n").encode("utf-8", errors="ignore")
        try:
            with self._tx_lock: self.sock.sendall(data)
        except Exception: pass

    def _rx_loop(self, s: socket.socket) -> None:
        try:
            while not self._stop_evt.is_set() and self.running and self.sock is s:
                try: chunk = s.recv(4096)
                except socket.timeout: continue
                except Exception: break
                if not chunk: break
                self.buf.extend(chunk)
                self._parse_buf()
        except Exception: pass
        self.running = False; self.connected = False

    def _parse_buf(self) -> None:
        while True:
            if len(self.buf) < 7: return
            if self.buf[0] != 0xA5 or self.buf[1] != 0x5A:
                try: i = self.buf.index(0xA5)
                except ValueError: self.buf.clear(); return
                self.buf = self.buf[i:]
                if len(self.buf) < 7: return
            
            plen = self.buf[3] | (self.buf[4] << 8)
            frame_len = 7 + plen
            if len(self.buf) < frame_len: return
            
            ptype = self.buf[2]
            payload = self.buf[5:5 + plen]
            self.buf = self.buf[frame_len:]

            if ptype == self.PKT_EMG and plen >= 8 + 40 * 4:
                start_idx = struct.unpack_from("<I", payload, 0)[0]
                gain = payload[4]; flags = payload[5]; fs = struct.unpack_from("<H", payload, 6)[0]
                off = 8
                ch1 = np.frombuffer(payload, dtype=np.int32, count=40, offset=off).copy()
                ch2 = np.frombuffer(payload, dtype=np.int32, count=40, offset=off + 160).copy() if plen >= 8 + 320 else np.zeros(40, dtype=np.int32)
                self.q_emg.append((start_idx, gain, flags, fs, ch1, ch2))
            
            elif ptype == self.PKT_IMU:
                if plen == 49:
                    seq, ms = struct.unpack_from("<II", payload, 0)
                    v = struct.unpack_from("<10f", payload, 8)
                    self.q_imu.append({"seq": seq, "ms": ms, "ax": v[0], "ay": v[1], "az": v[2], "gx": v[3], "gy": v[4], "gz": v[5], "mx": v[6], "my": v[7], "mz": v[8], "qw": 1, "qx": 0, "qy": 0, "qz": 0})
            
            elif ptype == self.PKT_BAT and plen >= 7:
                pct = payload[6]
                self.q_bat.append(pct)


# CSV File logger
class CsvLogger:
    def __init__(self):
        self.emg_f = None; self.imu_f = None
        self.enabled = False
        self.record_start_unix = 0.0

    def start(self, dev_id: int, session_id: str, record_start_unix: float) -> None:
        self.stop()
        save_dir = f"test/{session_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        self.emg_f = open(f"{save_dir}/emg_dev{dev_id:02d}.csv", "w", encoding="utf-8", newline="")
        self.imu_f = open(f"{save_dir}/emg_dev{dev_id:02d}_imu.csv", "w", encoding="utf-8", newline="")
        self.enabled = True
        self.record_start_unix = record_start_unix

        self.emg_f.write("Unix_Time_s,EMG_t_s,sample_idx,fs,gain,flags,emg_valid,ch1_raw_uV,ch1_filt_uV,ch1_env_uV,ch2_raw_uV,ch2_filt_uV,ch2_env_uV\n")
        self.imu_f.write("Unix_Time_s,IMU1_RefTime,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,mx_uT,my_uT,mz_uT,temp_C,qw,qx,qy,qz\n")

    def stop(self) -> None:
        for f in (self.emg_f, self.imu_f):
            if f:
                try: f.flush(); f.close()
                except Exception: pass
        self.enabled = False

    def write_emg(self, block_lines: list):
        if self.enabled and self.emg_f:
            self.emg_f.write("\n".join(block_lines) + "\n")
            
    def write_imu(self, imu: dict, abs_unix: float):
        if self.enabled and self.imu_f:
            self.imu_f.write(f"{abs_unix:.6f},{imu['ms']},{imu['ax']:.6f},{imu['ay']:.6f},{imu['az']:.6f},{imu['gx']:.6f},{imu['gy']:.6f},{imu['gz']:.6f},{imu['mx']:.6f},{imu['my']:.6f},{imu['mz']:.6f},0.0,{imu['qw']:.6f},{imu['qx']:.6f},{imu['qy']:.6f},{imu['qz']:.6f}\n")


# Headless single device driver manager
class HeadlessDevice:
    def __init__(self, slot_index):
        self.slot_index = slot_index
        self.dual_mode = (self.slot_index >= 12) 
        
        self.rx = Receiver()
        self.csv = CsvLogger()
        self.bat_pct = 0
        
        self.hpf1 = design_hpf(1000, 20.0); self.lpf1 = design_lpf(1000, 450.0); self.notch1 = design_notch(1000, 50.0)
        self.hpf2 = design_hpf(1000, 20.0); self.lpf2 = design_lpf(1000, 450.0); self.notch2 = design_notch(1000, 50.0)
        
        self.env_n = 50 
        self._env_buf_ch1 = np.zeros(self.env_n - 1, dtype=np.float64)
        self._env_buf_ch2 = np.zeros(self.env_n - 1, dtype=np.float64)
        
        self.gui_buffer_raw1 = []; self.gui_buffer_env1 = []
        self.gui_buffer_raw2 = []; self.gui_buffer_env2 = []
        
        self.last_good_uv_ch1 = 0.0
        self.last_good_uv_ch2 = 0.0

        # Global synchronized clock variables
        self._base_start_idx = None
        self._base_unix_time = 0.0
        self._base_imu_ms = None 
        self._base_imu_unix = 0.0 

    def process_queues(self):
        while self.rx.q_bat: self.bat_pct = self.rx.q_bat.popleft()
            
        while self.rx.q_imu:
            imu = self.rx.q_imu.popleft()
            if self.csv.enabled: 
                if self._base_imu_ms is None:
                    self._base_imu_ms = imu['ms']
                    self._base_imu_unix = self.csv.record_start_unix if self.csv.enabled else time.time()
                    
                abs_unix = self._base_imu_unix + ((imu['ms'] - self._base_imu_ms) / 1000.0)
                self.csv.write_imu(imu, abs_unix)

        updated_gui = False
        while self.rx.q_emg:
            start_idx, gain, flags, fs, ch1, ch2 = self.rx.q_emg.popleft()
            
            if self._base_start_idx is None: 
                self._base_start_idx = start_idx
                self._base_unix_time = self.csv.record_start_unix if self.csv.enabled else time.time()

            uv_scale = ADS1292R_uV_per_count(gain)
            ch1_uv = ch1.astype(np.float64) * uv_scale
            ch2_uv = ch2.astype(np.float64) * uv_scale
            
            # Anti-noise Shield Level 1: Out-of-bounds voltage check
            valid1 = np.abs(ch1_uv) <= 20000.0
            if not np.all(valid1):
                last1 = self.last_good_uv_ch1
                for i in range(len(ch1_uv)):
                    if abs(ch1_uv[i]) > 20000.0: ch1_uv[i] = last1
                    else: last1 = ch1_uv[i]
            self.last_good_uv_ch1 = ch1_uv[-1]

            valid2 = np.abs(ch2_uv) <= 20000.0
            if not np.all(valid2):
                last2 = self.last_good_uv_ch2
                for i in range(len(ch2_uv)):
                    if abs(ch2_uv[i]) > 20000.0: ch2_uv[i] = last2
                    else: last2 = ch2_uv[i]
            self.last_good_uv_ch2 = ch2_uv[-1]

            # Anti-noise Shield Level 2: Median filter for spike removal
            from scipy.signal import medfilt
            ch1_uv = medfilt(ch1_uv, kernel_size=3)
            ch2_uv = medfilt(ch2_uv, kernel_size=3)

            # Cascade filter link (HPF -> Notch -> LPF)
            y1 = self.lpf1.process(self.notch1.process(self.hpf1.process(ch1_uv.copy())))
            sq1 = y1 * y1
            concat1 = np.concatenate((self._env_buf_ch1, sq1))
            mean_sq1 = np.convolve(concat1, np.ones(self.env_n)/self.env_n, mode='valid')
            self._env_buf_ch1 = concat1[-(self.env_n - 1):]
            env1 = np.sqrt(np.maximum(mean_sq1, 0.0))

            if self.dual_mode:
                y2 = self.lpf2.process(self.notch2.process(self.hpf2.process(ch2_uv.copy())))
                sq2 = y2 * y2
                concat2 = np.concatenate((self._env_buf_ch2, sq2))
                mean_sq2 = np.convolve(concat2, np.ones(self.env_n)/self.env_n, mode='valid')
                self._env_buf_ch2 = concat2[-(self.env_n - 1):]
                env2 = np.sqrt(np.maximum(mean_sq2, 0.0))
            else:
                y2 = ch2_uv.copy(); env2 = np.zeros_like(y2)

            # Anti-noise Shield Level 3: Physiological limit clipping
            MAX_PHYSIOLOGICAL_ENV_UV = 3500.0  
            env1 = np.clip(env1, 0.0, MAX_PHYSIOLOGICAL_ENV_UV)
            env2 = np.clip(env2, 0.0, MAX_PHYSIOLOGICAL_ENV_UV)

            # Update UI plot cache
            self.gui_buffer_raw1.extend(ch1_uv.tolist()); self.gui_buffer_env1.extend(env1.tolist())
            if self.dual_mode:
                self.gui_buffer_raw2.extend(ch2_uv.tolist()); self.gui_buffer_env2.extend(env2.tolist())
                
            updated_gui = True

            # Write high-frequency data to CSV files
            if self.csv.enabled:
                csv_lines = []
                inv_fs = 1.0 / fs
                t_arr = (start_idx - self._base_start_idx + np.arange(40)) * inv_fs
                for i in range(40):
                    s_idx = start_idx + i
                    abs_unix_ts = self._base_unix_time + t_arr[i]
                    t_s = abs_unix_ts - self.csv.record_start_unix
                    
                    csv_lines.append(f"{abs_unix_ts:.6f},{t_s:.6f},{s_idx},{fs},{gain},{flags},1,{ch1_uv[i]:.3f},{y1[i]:.3f},{env1[i]:.3f},{ch2_uv[i]:.3f},{y2[i]:.3f},{env2[i]:.3f}")
                self.csv.write_emg(csv_lines)

        return updated_gui
    

# Main worker process loop
def discover_devices():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(0.2)
    try: s.bind(("", 0))
    except Exception: pass
    
    found = []
    try:
        s.sendto(DISCOVERY_MAGIC, ("255.255.255.255", DISCOVERY_PORT))
        t0 = time.time()
        while (time.time() - t0) < 0.5:
            try: data, addr = s.recvfrom(512)
            except socket.timeout: continue
            if data.decode("utf-8", errors="ignore").startswith(DISCOVERY_REPLY_PREFIX):
                parts = data.decode("utf-8").split()
                d = {"ip": addr[0], "tcp": 3333, "id": 0}
                for token in parts:
                    if "=" in token: k, v = token.split("=", 1); d[k] = v
                found.append(d)
    except Exception: pass
    finally: s.close()
    return found

def emg_worker_process(shared_state, emg_data_queue):
    print("[EMG Backend] Process started. Ready for concurrent acquisition.")
    devices = [HeadlessDevice(i+1) for i in range(17)]
    last_discover_time = 0
    last_gui_push_time = time.time()
    is_global_recording = False

    while not shared_state.get('cmd_stop_all', False):
        now = time.time()
        if now - last_discover_time > 3.0:
            devs = discover_devices()
            for d in devs:
                did = int(d.get("id", 0))
                if 1 <= did <= 17:
                    dev = devices[did-1]
                    if not dev.rx.connected:
                        if dev.rx.connect_to(d.get("ip"), int(d.get("tcp", 3333))):
                            print(f"[EMG Backend] Device {did} connected successfully.")
                            dev.rx.send_line("STREAM 1")
            last_discover_time = now

        target_record_state = shared_state.get('is_recording', False)
        if target_record_state != is_global_recording:
            is_global_recording = target_record_state
            for dev in devices:
                dev._base_start_idx = None
                dev._base_imu_ms = None
                if dev.rx.connected:
                    if is_global_recording: 
                        sid = shared_state.get('session_id', 'default_session')
                        sunix = shared_state.get('record_start_unix', time.time())
                        dev.csv.start(dev.slot_index, sid, sunix)
                    else: 
                        dev.csv.stop()

        any_updated = False
        for dev in devices:
            if dev.rx.connected:
                if dev.process_queues():
                    any_updated = True

        if any_updated and (now - last_gui_push_time >= 0.033):
            gui_payload = []
            for dev in devices:
                if dev.rx.connected and len(dev.gui_buffer_raw1) > 0:
                    gui_payload.append({
                        "slot_index": dev.slot_index,
                        "bat_pct": dev.bat_pct,
                        "raw_ch1": dev.gui_buffer_raw1,
                        "env_ch1": dev.gui_buffer_env1,
                        "raw_ch2": dev.gui_buffer_raw2,
                        "env_ch2": dev.gui_buffer_env2
                    })
                    dev.gui_buffer_raw1 = []; dev.gui_buffer_env1 = []
                    dev.gui_buffer_raw2 = []; dev.gui_buffer_env2 = []
            
            import queue
            if gui_payload:
                try: emg_data_queue.put_nowait(gui_payload)
                except queue.Full: pass
            
            last_gui_push_time = now
        time.sleep(0.005) 

    for dev in devices:
        if dev.rx.connected:
            dev.rx.send_line("STREAM 0")
            dev.rx.disconnect()
        if dev.csv.enabled: dev.csv.stop()
