#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import socket
import json
import queue

# Network parameters configuration
LINUX_ROS_IP = "10.50.191.1"  
LINUX_UDP_PORT = 50051           
WINDOWS_DATA_PORT = 50052        

class ContinuousExoskeletonLogger:
    def __init__(self):
        self.is_recording = False
        self.csv_file = None
        self.csv_writer = None
        self.record_unix_start = 0.0
        
    def start_recording(self, session_id, record_start_unix):
        import csv
        save_dir = f"test/{session_id}"
        os.makedirs(save_dir, exist_ok=True)
        self.csv_file = open(f"{save_dir}/Exoskeleton_Data.csv", "w", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write CSV headers
        self.csv_writer.writerow(["Unix_Time_s", "RelativeTime_s", "Target_Position_Deg", "Measured_Position_Deg", "Is_Bursting"])
        self.record_unix_start = record_start_unix
        self.is_recording = True
        print(f"[Exo Logger] 📝 Data recording started synchronized.")

    def stop_recording(self):
        self.is_recording = False
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            print(f"[Exo Logger] ⏹ Data recording stopped and saved safely.")

    def write_live_frame(self, target_angle, measured_angle, is_bursting):
        if self.is_recording and self.csv_writer:
            cur_unix = time.time()  
            rel_time = cur_unix - self.record_unix_start
            self.csv_writer.writerow([
                f"{cur_unix:.6f}", 
                f"{rel_time:.3f}", 
                f"{target_angle:.3f}", 
                f"{measured_angle:.3f}", 
                int(is_bursting)
            ])


def exo_worker_process(shared_state, exo_command_queue, exo_data_queue=None):
    print("[Exo Backend] Exoskeleton wireless sync engine started.")
    logger = ContinuousExoskeletonLogger()
    
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Increase UDP receive buffer size to 64KB to prevent dropouts
        rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        rx_sock.bind(('0.0.0.0', WINDOWS_DATA_PORT))
        rx_sock.settimeout(0.005) 
        print(f"[Exo Backend] Socket bound to port {WINDOWS_DATA_PORT} successfully.")
    except Exception as e:
        print(f"[Exo Backend] ❌ Socket binding or config failed: {e}")
        tx_sock.close()
        return
        
    is_global_recording = False
    last_valid_measured_angle = 6.0  

    # Main worker loop
    while not shared_state.get('cmd_stop_all', False):
        
        # Link global recording switch state
        target_record_state = shared_state.get('is_recording', False)
        if target_record_state != is_global_recording:
            is_global_recording = target_record_state
            if is_global_recording:
                sid = shared_state.get('session_id', 'default_session')
                sunix = shared_state.get('record_start_unix', time.time())
                logger.start_recording(sid, sunix)
            else:
                logger.stop_recording()

        # Check and handle perturbation trigger from UI
        if exo_command_queue is not None and not exo_command_queue.empty():
            try:
                cmd = exo_command_queue.get_nowait()
                if cmd == "TRIGGER_PERTURBATION":
                    tx_packet = b"TRIGGER_PERTURBATION"
                    tx_sock.sendto(tx_packet, (LINUX_ROS_IP, LINUX_UDP_PORT))
                    print(f"[Exo Backend] 🚀 Perturbation trigger sent to Linux.")
            except queue.Empty:
                pass

        # Parse streaming incoming data packets
        try:
            data, addr = rx_sock.recvfrom(65536)
            payload = json.loads(data.decode('utf-8'))
            
            data_points = payload.get("data_points", [])
            if data_points:
                target_angle = data_points[-1].get("target_pos", 6.0)
                raw_measured_angle = data_points[-1].get("measured_pos", 6.0)
                current_bursting_state = data_points[-1].get("is_bursting", 0)
                
                # Filter out sudden anomaly bus spikes and outliers
                if raw_measured_angle < -1.0 or abs(raw_measured_angle - last_valid_measured_angle) > 4.0:
                    filtered_measured_angle = last_valid_measured_angle
                else:
                    filtered_measured_angle = raw_measured_angle
                    last_valid_measured_angle = raw_measured_angle
                
                logger.write_live_frame(target_angle, filtered_measured_angle, current_bursting_state)

                # Forward clean live frames to front-end UI queue
                if exo_data_queue is not None:
                    try:
                        exo_data_queue.put_nowait({
                            "target": target_angle,
                            "measured": filtered_measured_angle,
                            "is_bursting": current_bursting_state
                        })
                    except queue.Full:
                        pass
        except socket.timeout:
            continue
        except Exception:
            continue

    tx_sock.close()
    rx_sock.close()
    logger.stop_recording()
    print("[Exo Backend] 🛑 Exoskeleton communications process exited safely.")
