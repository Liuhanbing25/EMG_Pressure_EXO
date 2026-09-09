#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing as mp
import sys
import time

from backend_emg import emg_worker_process
from backend_insole import insole_worker_process
from frontend_gui import run_gui_process
from backend_exoskeleton import exo_worker_process  

def main():
    print("==================================================")
    print("🚀 Starting Multiprocess Biomechanics System")
    print("==================================================")

    # Initialize shared memory (IPC)
    manager = mp.Manager()
    shared_state = manager.dict()
    shared_state['is_recording'] = False 
    shared_state['cmd_stop_all'] = False 

    # Initialize data communication queues
    emg_data_queue = mp.Queue(maxsize=100)
    insole_data_queue = mp.Queue(maxsize=100)
    exo_command_queue = mp.Queue(maxsize=10)     
    exo_data_queue = mp.Queue(maxsize=100)

    # Initialize background processes
    emg_process = mp.Process(
        target=emg_worker_process, 
        args=(shared_state, emg_data_queue),
        name="Backend_EMG_Engine"
    )
    
    insole_process = mp.Process(
        target=insole_worker_process, 
        args=(shared_state, insole_data_queue),
        name="Backend_Insole_Engine"
    )

    exo_process = mp.Process(
        target=exo_worker_process,
        args=(shared_state, exo_command_queue, exo_data_queue),
        name="Backend_Exoskeleton_Engine"
    )

    # Start all background processes
    emg_process.start()
    insole_process.start()
    exo_process.start()
    print("[Main] 🔥 Background processes started successfully.")

    # Start the frontend GUI inside the main thread
    try:
        print("[Main] Launching frontend UI...")
        run_gui_process(shared_state, emg_data_queue, insole_data_queue, exo_command_queue, exo_data_queue)
        
    except KeyboardInterrupt:
        print("\n[Main] Hardware interrupt detected (Ctrl+C)...")
    finally:
        # Graceful cleanup and safe resource recovery
        print("[Main] Stopping background engines and saving files...")
        shared_state['cmd_stop_all'] = True 
        
        # Flush and empty queues to prevent multiprocessing deadlock
        for q, q_name in [(emg_data_queue, "EMG Queue"), (insole_data_queue, "Insole Queue"), (exo_command_queue, "Exo Cmd Queue"), (exo_data_queue, "Exo Data Queue")]:
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass

        # Give processes time to flush data to CSV files
        time.sleep(0.5) 
        
        # Terminate any remaining active processes
        for p, name in [(emg_process, "EMG"), (insole_process, "Insole"), (exo_process, "Exoskeleton")]:
            if p.is_alive():
                print(f"[Main] Terminating {name} engine...")
                p.terminate()
                p.join()
            else:
                print(f"[Main] {name} engine stopped safely.")
            
        print("==================================================")
        print("🎉 System shut down completely and safely.")
        print("==================================================")

if __name__ == '__main__':
    # Fix for multiprocessing support on Windows
    mp.freeze_support()
    main()
