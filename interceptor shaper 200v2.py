import numpy as np
import pandas as pd
from numba import njit
import plotly.graph_objects as go

# --- 1. MISSION PARAMETERS ---
# Updated for 200 m/s Target Profile
INTERCEPTOR_SPEED_MS = np.float32(280.0)  # Hammer
SHAPER_SPEED_MS      = np.float32(200.0)  # Shaper
TARGET_SPEED_MS      = np.float32(200.0)  # Target

SIM_DURATION_S       = 30.0   # Increased to account for lower closure rate
DT                   = 0.01   
STEPS                = int(SIM_DURATION_S / DT)

# --- 2. GUIDANCE LOGIC ---
@njit
def compute_swarm_target(t_pos, t_vel, drone_pos, mode="SHAPER"):
    if mode == "SHAPER":
        # Adjusted lead time for 200 m/s maneuvering[cite: 2]
        lead_time    = np.float32(3.5) 
        target_point = t_pos + (t_vel * lead_time)
        target_point[2] -= np.float32(15.0) # Forcing target upward[cite: 2]
        return target_point
    else:
        # Hammer: ProNav with updated Speed-of-Sound/Interceptor Speed[cite: 2]
        rel_pos = t_pos - drone_pos
        dist = np.sqrt(np.sum(rel_pos**2))
        
        # Updated hardcoded TOF constant to 280.0 m/s[cite: 2]
        tof  = dist / np.float32(280.0) 
        return t_pos + t_vel * tof

@njit
def simulate():
    # Initial Positions[cite: 2]
    target_pos = np.array([0.0, 5000.0, 2000.0], dtype=np.float32)
    target_vel = np.array([200.0, 0.0, 0.0], dtype=np.float32) # Target at 200 m/s[cite: 2]
    
    shaper_pos = np.array([0.0, 4800.0, 1800.0], dtype=np.float32)
    hammer_pos = np.array([-500.0, 5200.0, 2500.0], dtype=np.float32)
    
    results = np.zeros((STEPS, 9), dtype=np.float32)
    
    for i in range(STEPS):
        # 1. Update Target (Simulating slight evasion)[cite: 2]
        target_pos += target_vel * DT
        
        # 2. Shaper Guidance[cite: 2]
        s_goal = compute_swarm_target(target_pos, target_vel, shaper_pos, "SHAPER")
        s_dir = (s_goal - shaper_pos)
        s_dir /= np.linalg.norm(s_dir)
        shaper_pos += s_dir * SHAPER_SPEED_MS * DT
        
        # 3. Hammer Guidance[cite: 2]
        h_goal = compute_swarm_target(target_pos, target_vel, hammer_pos, "HAMMER")
        h_dir = (h_goal - hammer_pos)
        h_dir /= np.linalg.norm(h_dir)
        hammer_pos += h_dir * INTERCEPTOR_SPEED_MS * DT
        
        # Store results
        results[i, 0:3] = target_pos
        results[i, 3:6] = shaper_pos
        results[i, 6:9] = hammer_pos
        
        # Impact Check[cite: 2]
        if np.linalg.norm(hammer_pos - target_pos) < 2.0:
            return results[:i]
            
    return results

# --- 3. EXECUTION & VISUALIZATION ---
data = simulate()
df = pd.DataFrame(data, columns=['Tx', 'Ty', 'Tz', 'Sx', 'Sy', 'Sz', 'Hx', 'Hy', 'Hz'])

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=df['Tx'], y=df['Ty'], z=df['Tz'], name='Target (200m/s)', line=dict(color='red')))
fig.add_trace(go.Scatter3d(x=df['Sx'], y=df['Sy'], z=df['Sz'], name='Shaper (200m/s)', line=dict(color='blue')))
fig.add_trace(go.Scatter3d(x=df['Hx'], y=df['Hy'], z=df['Hz'], name='Hammer (280m/s)', line=dict(color='black', width=5)))

fig.update_layout(title="ESPIRIDI Intercept: 200m/s Target Profile", scene=dict(aspectmode='data'))
fig.show()