#pragma once
#include "rust/cxx.h"

// ============================================================================
// Axim GlassOS — Ghost-Touch Protection & Haptic Feedback
// ============================================================================
// FTIR (Frustrated Total Internal Reflection) allows the Edge Rail to
// differentiate between Side-A (authorized) and Side-B (unauthorized) touches
// based on IR intensity curves through the electrochromic core.
//
// Haptic architecture: "Lateral Traveling Wave" (LTW) via TDK PiezoHapt™
// actuators at 2.5mm pitch, with 40kHz ultrasonic carrier + 220Hz modulation.
//
// Ghost-touch threshold: 0.5 Newtons minimum surface pressure for registration.
// ============================================================================

struct TouchProfile {
    int   touch_id;                // Unique identifier for multi-touch tracking
    float x_coord;                 // Normalized X position [0.0 - 1.0]
    float y_coord;                 // Normalized Y position [0.0 - 1.0]
    float pressure_newtons;        // Surface pressure in Newtons
    float ir_intensity;            // IR disruption intensity [0.0 - 1.0]
    bool  is_front_side;           // true = Side-A (front), false = Side-B (rear)
    bool  is_authorized;           // Authorization result for this touch
};

struct HapticPulse {
    float frequency_hz;            // Modulation frequency (220 Hz for Pacinian)
    float carrier_khz;             // Ultrasonic carrier (40 kHz squeeze-film)
    float amplitude;               // Pulse strength [0.0 - 1.0]
    float phase_shift_degrees;     // Phase offset for LTW (90° sequential)
    float duration_ms;             // Pulse duration in milliseconds
};

// Classify which side of the glass a touch originates from
// Uses FTIR refractive shadow analysis through the electrochromic core
int classify_touch_side(float ir_intensity, float pressure);

// Check if a touch is authorized based on side + active permission state
bool is_authorized_touch(bool is_front_side, bool side_b_permitted);

// Filter ghost touches below the 0.5N threshold
// Returns true if the touch should be processed, false if rejected
bool filter_ghost_touch(float pressure_newtons);

// Trigger a Lateral Traveling Wave haptic effect along the Edge Rail
// Used for the "Shared Surface" passing sensation
void trigger_lateral_wave(float start_x, float end_x, float duration_ms);

// Trigger a viscous ripple barrier at the point of unauthorized contact
// Creates the "liquid mercury" visual + haptic barrier effect
void trigger_ripple_barrier(float x_coord, float y_coord);

// Get a combined touch + haptic diagnostic string
rust::String get_touch_diagnostics(int active_touches_a, int active_touches_b,
                                   bool ghost_protection_active);
