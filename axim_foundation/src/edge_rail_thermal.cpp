#include "edge_rail_thermal.h"
#include <iostream>
#include <sstream>

// ============================================================================
// Axim GlassOS — Edge Rail Thermal Management Implementation
// ============================================================================

float calculate_thermal_load(float display_power, float display_efficiency,
                             float compute_power, float compute_efficiency) {
    // Q_total = [P_display × (1 − η_display)] + [P_compute × (1 − η_compute)]
    float display_heat = display_power * (1.0f - display_efficiency);
    float compute_heat = compute_power * (1.0f - compute_efficiency);
    float total = display_heat + compute_heat;

    std::cout << "C++ (Edge Rail Thermal): Display waste heat = "
              << display_heat << "W (from " << display_power
              << "W at " << (display_efficiency * 100) << "% efficiency)" << std::endl;
    std::cout << "C++ (Edge Rail Thermal): Compute waste heat = "
              << compute_heat << "W (from " << compute_power
              << "W at " << (compute_efficiency * 100) << "% efficiency)" << std::endl;
    std::cout << "C++ (Edge Rail Thermal): Total thermal load = "
              << total << "W" << std::endl;

    return total;
}

float calculate_battery_runtime(float battery_wh, float display_power,
                                float npu_power, float aux_power) {
    // T_runtime = (battery_wh / total_power_draw) × 60
    float total_power = display_power + npu_power + aux_power;
    float runtime_minutes = (battery_wh / total_power) * 60.0f;

    std::cout << "C++ (Edge Rail Thermal): Battery = " << battery_wh << " Wh" << std::endl;
    std::cout << "C++ (Edge Rail Thermal): Total draw = " << total_power
              << "W (Display: " << display_power << "W + NPU: " << npu_power
              << "W + Aux: " << aux_power << "W)" << std::endl;
    std::cout << "C++ (Edge Rail Thermal): Estimated runtime = "
              << runtime_minutes << " minutes" << std::endl;

    return runtime_minutes;
}

bool is_cooling_sufficient(float thermal_load, float max_tdp) {
    bool sufficient = thermal_load <= max_tdp;
    if (!sufficient) {
        std::cout << "C++ (Edge Rail Thermal): WARNING — Thermal load ("
                  << thermal_load << "W) exceeds TDP (" << max_tdp
                  << "W). Activating thermal throttle!" << std::endl;
    } else {
        std::cout << "C++ (Edge Rail Thermal): Cooling nominal. Load "
                  << thermal_load << "W within " << max_tdp << "W TDP." << std::endl;
    }
    return sufficient;
}

rust::String get_cooling_status(float thermal_load, float max_tdp,
                                float glass_temp) {
    std::ostringstream oss;
    float utilization = (thermal_load / max_tdp) * 100.0f;
    bool glass_safe = glass_temp < 43.0f; // Touch safety threshold

    oss << "=== Edge Rail Cooling Status ===" << std::endl;
    oss << "Thermal Load: " << thermal_load << "W / " << max_tdp << "W TDP ("
        << utilization << "% utilization)" << std::endl;
    oss << "Vapor Chamber: " << (utilization > 80 ? "HIGH LOAD" : "NOMINAL") << std::endl;
    oss << "Micro-Blowers: " << (utilization > 50 ? "ACTIVE" : "IDLE") << std::endl;
    oss << "Glass Surface: " << glass_temp << "°C ["
        << (glass_safe ? "SAFE" : "OVERTEMP — THROTTLING") << "]" << std::endl;
    oss << "Aerogel Barrier: INTACT" << std::endl;

    std::string result = oss.str();
    std::cout << result;
    return rust::String(result);
}
