#include "ghost_touch.h"
#include <iostream>
#include <sstream>
#include <cmath>

// ============================================================================
// Axim GlassOS — Ghost-Touch Protection & Haptic Feedback Implementation
// ============================================================================

// Ghost-touch minimum pressure threshold (Newtons)
static const float GHOST_TOUCH_THRESHOLD = 0.5f;

// FTIR IR intensity threshold to distinguish Side-A from Side-B
// Side-A touches produce higher intensity (direct refractive shadow)
// Side-B touches produce lower intensity (attenuated through electrochromic core)
static const float SIDE_A_IR_THRESHOLD = 0.55f;

int classify_touch_side(float ir_intensity, float pressure) {
    if (pressure < GHOST_TOUCH_THRESHOLD) {
        std::cout << "C++ (Ghost-Touch): Sub-threshold contact ("
                  << pressure << "N < " << GHOST_TOUCH_THRESHOLD
                  << "N). Classified as ghost touch — REJECTED." << std::endl;
        return -1;  // Ghost touch, no side
    }

    bool is_side_a = ir_intensity >= SIDE_A_IR_THRESHOLD;
    std::cout << "C++ (Ghost-Touch): IR intensity = " << ir_intensity
              << " → Classified as Side-" << (is_side_a ? "A" : "B")
              << " (threshold: " << SIDE_A_IR_THRESHOLD << ")" << std::endl;

    return is_side_a ? 0 : 1;  // 0 = Side-A, 1 = Side-B
}

bool is_authorized_touch(bool is_front_side, bool side_b_permitted) {
    if (is_front_side) {
        std::cout << "C++ (Ghost-Touch): Side-A touch — AUTHORIZED (primary user)." << std::endl;
        return true;
    }

    if (side_b_permitted) {
        std::cout << "C++ (Ghost-Touch): Side-B touch — AUTHORIZED (Shared Surface active)." << std::endl;
        return true;
    }

    std::cout << "C++ (Ghost-Touch): Side-B touch — BLOCKED. "
              << "No Shared Surface permission. Triggering ripple barrier." << std::endl;
    return false;
}

bool filter_ghost_touch(float pressure_newtons) {
    bool valid = pressure_newtons >= GHOST_TOUCH_THRESHOLD;
    if (!valid) {
        std::cout << "C++ (Ghost-Touch): Filtering ghost contact at "
                  << pressure_newtons << "N (sleeve/jewelry interference)." << std::endl;
    }
    return valid;
}

void trigger_lateral_wave(float start_x, float end_x, float duration_ms) {
    // Lateral Traveling Wave via TDK PiezoHapt™ at 2.5mm pitch
    // Carrier: 40 kHz ultrasonic (squeeze-film friction reduction)
    // Modulation: 220 Hz (peak Pacinian corpuscle sensitivity)
    // Phase shift: 90° sequential across actuator array

    float distance = std::fabs(end_x - start_x);
    int actuator_count = static_cast<int>(distance * 200.0f); // ~200 actuators across 20"

    std::cout << "C++ (Haptics LTW): Lateral Traveling Wave triggered." << std::endl;
    std::cout << "  Path: X=" << start_x << " → X=" << end_x
              << " (" << actuator_count << " actuators)" << std::endl;
    std::cout << "  Carrier: 40 kHz ultrasonic (squeeze-film active)" << std::endl;
    std::cout << "  Modulation: 220 Hz (Pacinian peak)" << std::endl;
    std::cout << "  Phase shift: φ=90° sequential" << std::endl;
    std::cout << "  Duration: " << duration_ms << "ms" << std::endl;
    std::cout << "  Effect: 'Passing' sensation — object slides from Side-A to Side-B." << std::endl;
}

void trigger_ripple_barrier(float x_coord, float y_coord) {
    // Viscous ripple barrier — creates the "liquid mercury" resistance effect
    // The glass renders an opaque white frosted circle at the contact point
    // UI elements undergo displacement mapping to "push away" from the intruder

    std::cout << "C++ (Haptics): Ripple barrier activated at ("
              << x_coord << ", " << y_coord << ")." << std::endl;
    std::cout << "  Visual: Frosted circle follows unauthorized finger." << std::endl;
    std::cout << "  UI: Displacement mapping — elements pushed away from contact." << std::endl;
    std::cout << "  Haptic: Resistance pulse at 220 Hz via Edge Rail." << std::endl;
    std::cout << "  Alert: Haptic notification → owner's GRRN Ring (HMI-R1)." << std::endl;
    std::cout << "  Edge Rail: Red border glow activated." << std::endl;
}

rust::String get_touch_diagnostics(int active_touches_a, int active_touches_b,
                                   bool ghost_protection_active) {
    std::ostringstream oss;
    oss << "=== FTIR Touch Diagnostics ===" << std::endl;
    oss << "Side-A (Front) Active Touches: " << active_touches_a << "/10" << std::endl;
    oss << "Side-B (Rear)  Active Touches: " << active_touches_b << "/10" << std::endl;
    oss << "Ghost-Touch Protection: "
        << (ghost_protection_active ? "ACTIVE" : "SUSPENDED (Shared Surface mode)")
        << std::endl;
    oss << "Pressure Threshold: " << GHOST_TOUCH_THRESHOLD << "N" << std::endl;
    oss << "IR Classification Threshold: " << SIDE_A_IR_THRESHOLD << std::endl;
    oss << "Capacitance Profiling: ENABLED" << std::endl;
    oss << "Max Simultaneous Multitouch: 10-point per side" << std::endl;

    std::string result = oss.str();
    std::cout << result;
    return rust::String(result);
}
