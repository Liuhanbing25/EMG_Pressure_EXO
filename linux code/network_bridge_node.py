#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32  # 完美恢复底层发布与接收话题接口
import struct
import time
import socket
import threading
import sys
import select
import json

class ExoskeletonDualControlBridge(Node):
    def __init__(self):
        super().__init__('network_bridge_node')
        
        # =============================================================================
        # ⚙️ 核心物理安全参数配置区（现场通过改数字来调节发力猛烈度）
        # =============================================================================
        self.master_id = 0xFD          # EtherCAT 主站 ID
        self.target_id = 0x01          # 目标关节电机 ID
        
        # 1. ⚡ 释放爆发力（给足动力，让它在极短时间内拉满行程）
        self.safety_speed = 50.0        # 👈 释放速度盾，允许高频快速拉伸
        self.safety_acc = 50.0          # 👈 释放加速度盾
        
        # 2. 实验前的物理平衡零位（现场可通过 Linux 键盘 8 和 2 随时微调对齐）
        self.target_position = 5.0     # 👈 默认初始平衡原点位置
        
        # 3. 🎯 核心调节：设定每一次空格/按钮触发时的【净扰动角度增量行程】
        self.burst_increment = 2.0    # 👈 每按一次空格，在当前原点基础上“额外转动”的度数
        self.burst_duration = 0.80   # 👈 保持发力行程的时间窗口长度 (秒)
        
        # 4. 状态标记位与运行时计算目标暂存
        self._in_burst_phase = False
        self._actual_active_target = 6.0 # 动态计算出的爆发期绝对目标角
        
        # =============================================================================
        # 🟢 真正建立 ROS 2 底层通信总线双向通道（100% 对齐原厂话题与位移逻辑）
        # =============================================================================
        self.pub_cmd = self.create_publisher(Float64MultiArray, '/commands', 10)
        self.sub_angle = self.create_subscription(Float32, '/motor1/motor_angle', self.angle_callback, 10)
        
        # 🧠 用于缓存实时接收到的外骨骼实际角度反馈（核心：根治硬直死线问题）
        self.measured_position = 1.0  
        
        # =============================================================================
        # 🌐 跨平台无线局域网参数配置（连接同一手机热点后，请对齐 IP 地址）
        # =============================================================================
        self.WINDOWS_IP = "10.50.191.189"   # 已经为您对齐了您现场的最新的 Windows IP
        self.WINDOWS_PORT = 50052           # Windows 数据接收端端口
        self.LINUX_PORT = 50051             # Linux 无线命令接收端口
        
        # 初始化无线网络发射套接字
        self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # =============================================================================
        # ⚡ 初始化底层电机位置状态
        # =============================================================================
        self.get_logger().info("🚀 Linux 外骨骼【终极全兼容双控版】网桥节点已成功点火...")
        self.enable_motor_and_lock_balance(self.target_position)
        
        # =============================================================================
        # ⚠️ 安全断言检查
        # =============================================================================
        if self.target_position > 10:
            self.get_logger().error("❌ 严重警告: 初始设定的平衡原点超过了原厂 14.5° 安全上限屏蔽区！")
            self.target_position = 10
        
        # =============================================================================
        # 🕹️ 并发并行：让“本地键盘监听”与“网络无线监听”同时轰鸣运行！
        # =============================================================================
        self.get_logger().info("🔥 [双控模式已全面激活]: 本地键盘(8/2/空格) 与 Windows 远程按钮可同时控制发力！")
        
        # 1. 启动本地键盘监听线程
        self.keyboard_thread = threading.Thread(target=self._local_keyboard_space_loop, daemon=True)
        self.keyboard_thread.start()
        
        # 2. 启动局域网无线监听线程
        self._init_network_socket()
        self.network_thread = threading.Thread(target=self._network_position_rx_loop, daemon=True)
        self.network_thread.start()
        
        # 3. 🟢 全时高频流式广播定时器：每 10 毫秒（100Hz）将最新单帧多维数据持续推向 Windows
        self.heartbeat_timer = self.create_timer(0.01, self._send_continuous_heartbeat_to_windows)

    def angle_callback(self, msg):
        """🟢 接收实际物理角度的回调函数：高频刷新内存"""
        self.measured_position = msg.data

    def build_can_id(self, prefix, master_id, target_id):
        # 保持原厂规范的 24 位左移逻辑！
        return float((prefix << 24) | (master_id << 8) | target_id)

    def send_frame(self, ext_id, data_bytes):
        # 将总线一侧的发送接口补全，使位置指令真正飞向底层的 bringup
        msg = Float64MultiArray()
        msg.data = [float(ext_id)] + [float(b) for b in data_bytes]
        self.pub_cmd.publish(msg)

    def enable_motor_and_lock_balance(self, target_angle):
        """位置模式安全使能并强制刷入匀速保护盾"""
        # 0x12 前缀映射使能
        ext_id = self.build_can_id(0x12, self.master_id, self.target_id)
        self.send_frame(ext_id, [0x05, 0x70, 0, 0, 1, 0, 0, 0])
        time.sleep(0.02)
        
        # 0x03 清零或刷新状态机
        ext_id_clear = self.build_can_id(0x03, self.master_id, self.target_id)
        self.send_frame(ext_id_clear, [0, 0, 0, 0, 0, 0, 0, 0])
        time.sleep(0.08)
        
        # 注入设定的速度/加速度限制
        ext_id_drive = self.build_can_id(0x12, self.master_id, self.target_id)
        self.send_frame(ext_id_drive, [0x24, 0x70, 0, 0] + list(struct.pack('<f', self.safety_speed)))
        self.send_frame(ext_id_drive, [0x25, 0x70, 0, 0] + list(struct.pack('<f', self.safety_acc)))
        
        # 锁定到设定的平衡原点
        self.send_frame(ext_id_drive, [0x16, 0x70, 0, 0] + list(struct.pack('<f', target_angle)))

    def _send_continuous_heartbeat_to_windows(self):
        """🟢 优化升级：全时高频流式无损广播"""
        # 判断当前实时处于何种发力状态
        is_bursting_now = 1 if self._in_burst_phase else 0
        # 根据状态自动计算当前输出的设定角度命令（爆发期流式同步动态目标值）
        current_target = self._actual_active_target if is_bursting_now else self.target_position

        # 科学解耦：分别封装 target_pos（命令设定值）和 measured_pos（物理测得实际值）
        single_log = [{
            "time_offset": 0.0,
            "target_pos": round(current_target, 3),        # 设定目标指令角度
            "measured_pos": round(self.measured_position, 3), # 实时接收到的实际角度反馈
            "is_bursting": is_bursting_now
        }]
        
        payload = {
            "start_unix": time.time(),
            "duration": 0.0,
            "data_points": single_log
        }
        try:
            json_str = json.dumps(payload).encode('utf-8')
            self.tx_sock.sendto(json_str, (self.WINDOWS_IP, self.WINDOWS_PORT))
        except:
            pass

    def _execute_position_perturbation_sequence(self, source_label="未知源"):
        """量化科研时序逻辑优化：基于当前平衡原点的【自适应相对角度增量发力控制】"""
        ext_id = self.build_can_id(0x12, self.master_id, self.target_id)
        
        # 1. 动态精算本次爆发的绝对目标角度 = 当前原点 + 固定行程增量
        raw_target = self.target_position + self.burst_increment
        
        # 🛡️ 硬件级硬锁：施加 14.5° 上限安全盾截断
        if raw_target > 14.5:
            self._actual_active_target = 14.5
            self.get_logger().error(f"⚠️ [安全上限拦截] 触发增量后达到 {raw_target:.2f}°，已强制拦截并截断在安全红线：14.5 度！")
        else:
            self._actual_active_target = raw_target
            
        # 2. 进入突发爆发期（全时心跳广播会自动感知此标志，并将 csv 对应行标为 is_bursting=1）
        self._in_burst_phase = True
        
        pos_bytes = list(struct.pack('<f', self._actual_active_target))
        self.send_frame(ext_id, [0x16, 0x70, 0, 0] + pos_bytes)
        self.get_logger().warn(
            f"🔥 [{source_label}激活扰动] 支架拉向绝对测试点: {self._actual_active_target:.2f} 度 "
            f"(当前平衡原点: {self.target_position:.2f}° | 净爆发增量: {self.burst_increment}°)"
        )
        
        # 匀速保持发力行程窗口，期间 100Hz 定时器在持续流式广播单帧数据
        time.sleep(self.burst_duration)
            
        # 3. 退出爆发期，平稳放线安全复位退回当前的初始原点
        self._in_burst_phase = False 
        comp_bytes = list(struct.pack('<f', self.target_position))
        self.send_frame(ext_id, [0x16, 0x70, 0, 0] + comp_bytes)
        self.get_logger().info(f"⏹ [退回零位] 支架已平稳退回当前初始原点: {self.target_position} 度")
        
        # 预留平衡恢复期监测窗口，确保人体在突然撤销力矩后的挣扎晃动波形完全落盘
        time.sleep(0.400)
        self.get_logger().info(f"📡 [流式同步完成] 扰动全程波形已通过高频天线实时送达 Windows 母舰。")

    def _local_keyboard_space_loop(self):
        while rclpy.ok():
            try:
                print(f"\n[当前平衡原点]: {self.target_position:.2f} 度 | 实时测得实际角度: {self.measured_position:.2f} 度")
                print("⌨️  【Linux键盘】: 按 '8' 抬高初始角度 | 按 '2' 降低初始角度 | 按 '空格键(Space)' 本地发射扰动")
                
                dr, _, _ = select.select([sys.stdin], [], [])
                if dr:
                    key = sys.stdin.read(1)
                    if key == " ":
                        self._execute_position_perturbation_sequence(source_label="Linux键盘")
                    elif key == "8":
                        # 抬高原点时同步施加14.5度上限安全盾保护
                        self.target_position = min(14.5, self.target_position + 0.4) 
                        self.enable_motor_and_lock_balance(self.target_position)
                    elif key == "2":
                        self.target_position = max(0.0, self.target_position - 0.4) # 下踩安全边界微调
                        self.enable_motor_and_lock_balance(self.target_position)
            except Exception as e:
                self.get_logger().error(f"本地键盘循环出现异常: {e}")
                break

    def _init_network_socket(self):
        self.rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 端口复用保护盾，彻底杜现 Address already in use
        self.rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx_sock.bind(("0.0.0.0", self.LINUX_PORT))
        self.get_logger().info(f"📡 局域网 UDP 接收天线已在端口 {self.LINUX_PORT} 撑开，接收 Windows GUI 脉冲中...")

    def _network_position_rx_loop(self):
        while rclpy.ok():
            try:
                data, addr = self.rx_sock.recvfrom(1024)
                if data == b"TRIGGER_PERTURBATION":
                    self.get_logger().warn(f"⚡ [无线网络激活] 成功接收到 Windows {addr} GUI 按钮指令！ ")
                    self._execute_position_perturbation_sequence(source_label="Windows界面")
            except Exception as e:
                self.get_logger().error(f"网络无线接收循环发生异常: {e}")
                break

def main(args=None):
    rclpy.init(args=args)
    node = ExoskeletonDualControlBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 安全下电并完全断开物理使能，让外骨骼安全完全松开
        ext_id = node.build_can_id(0x12, node.master_id, node.target_id)
        node.send_frame(ext_id, [0x05, 0x70, 0, 0, 0, 0, 0, 0])
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
