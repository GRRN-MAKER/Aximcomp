#pragma once
#include "rust/cxx.h"

// ============================================================================
// Axim GlassOS — Edge Rail Thermal Management
// ============================================================================
// Implements the thermodynamic model for the 20-inch Edge Rail Dock.
// Core formula: Q = P_input × (1 − η)
//
// The dock must continuously dissipate:
//   Q_total = [P_display × (1 − η_display)] + [P_compute × (1 − η_compute)]
//
// At 100W display (30% eff) + 30W NPU load → 100W continuous thermal budget.
// Cooling: Vapor chamber + dual piezoelectric micro-blowers + aerogel isolation.
// ============================================================================

struct ThermalState {
    float display_power_watts;     // e.g., 100W for transparent OLED
    float display_efficiency;       // e.g., 0.30 for 30% luminous efficiency
    float compute_power_watts;      // e.g., 30W for NPU under agentic load
    float compute_efficiency;       // e.g., 0.15 for edge AI processor
    float ambient_temp_celsius;     // Room temperature
    float glass_surface_temp;       // Must stay below 43°C for touch safety
    float vapor_chamber_temp;       // Internal cooling element temperature
    bool  micro_blowers_active;     // Active cooling state
    bool  thermal_throttle_active;  // Emergency thermal limiting
};

struct CoolingProfile {
    float max_tdp_watts;            // Maximum thermal design power (100W)
    float vapor_chamber_capacity;   // Heat pipe capacity in watts
    float blower_cfm;               // Airflow in cubic feet per minute
    float aerogel_r_value;          // Thermal resistance of isolation layer
};

// Calculate total waste heat from display + compute subsystems
// Q_total = [P_display × (1 − η_display)] + [P_compute × (1 − η_compute)]
float calculate_thermal_load(float display_power, float display_efficiency,
                             float compute_power, float compute_efficiency);

// Calculate battery runtime in minutes
// T_runtime = (battery_wh / total_power_draw) × 60
float calculate_battery_runtime(float battery_wh, float display_power,
                                float npu_power, float aux_power);

// Check if cooling system can handle the current thermal load
bool is_cooling_sufficient(float thermal_load, float max_tdp);

// Get cooling system status string for diagnostics
rust::String get_cooling_status(float thermal_load, float max_tdp,
                                float glass_temp);
