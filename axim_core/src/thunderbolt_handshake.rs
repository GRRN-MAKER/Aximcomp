// ============================================================================
// Axim GlassOS — Thunderbolt 5 Handshake & Connectivity Manager (Rust Core)
// ============================================================================
// Manages the "Quantum Handshake" wake-up flow:
//   Phase A: TB5 bandwidth negotiation (120 Gbps allocation)
//   Phase B: "GRRN" Liquid Glass boot animation trigger
//   Phase C: Gaze-Locked Authentication (Optic ID / Ring)
//
// Handles seamless handover between:
//   BYOC Mode (Wired) — external CPU/GPU via Thunderbolt 5
//   Agentic Mode (Wireless) — internal NPU via Wi-Fi 7 MLO
//
// Energy budgeting:
//   T_runtime = (30 Wh / P_total) × 60 min
// ============================================================================

use pyo3::prelude::*;
use std::sync::Mutex;
use std::sync::OnceLock;

// ---- Enums ----

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectionMode {
    /// Wired mode — external CPU/GPU via Thunderbolt 5 (120 Gbps, 240W PD)
    BYOC,
    /// Wireless mode — internal NPU + Wi-Fi 7 MLO (46 Gbps aggregate)
    Agentic,
    /// No connection — display off or in standby
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PowerProfile {
    /// Full performance — all subsystems at max
    HighPerformance,
    /// Balanced — moderate throttling for extended runtime
    Balanced,
    /// Eco mode — "Deep Obsidian" palette, max OLED black pixels, ~90 min
    EcoMode,
}

// ---- Global State ----

struct ConnectionState {
    mode: ConnectionMode,
    power_profile: PowerProfile,
    // Bandwidth allocation (Gbps)
    bandwidth_video: f32,
    bandwidth_data: f32,
    bandwidth_aux: f32,
    // Battery
    battery_wh: f32,
    battery_remaining_pct: f32,
    is_charging: bool,
    // Wi-Fi 7 MLO
    wifi7_mlo_active: bool,
    wifi7_latency_ms: f32,
    // Authentication
    user_authenticated: bool,
    authenticated_user: String,
}

static CONNECTION_STATE: OnceLock<Mutex<ConnectionState>> = OnceLock::new();

fn get_connection_state_inner() -> &'static Mutex<ConnectionState> {
    CONNECTION_STATE.get_or_init(|| {
        Mutex::new(ConnectionState {
            mode: ConnectionMode::Disconnected,
            power_profile: PowerProfile::HighPerformance,
            bandwidth_video: 0.0,
            bandwidth_data: 0.0,
            bandwidth_aux: 0.0,
            battery_wh: 30.0,
            battery_remaining_pct: 100.0,
            is_charging: false,
            wifi7_mlo_active: false,
            wifi7_latency_ms: 0.0,
            user_authenticated: false,
            authenticated_user: String::new(),
        })
    })
}

// ---- PyO3 Functions ----

/// Phase A: Thunderbolt 5 bandwidth negotiation — the "Quantum Handshake"
#[pyfunction]
pub fn thunderbolt_connect(user_id: String) -> PyResult<String> {
    let mut state = get_connection_state_inner().lock().unwrap();

    println!("Rust (TB5 Handshake): Thunderbolt 5 cable detected.");
    println!("Rust (TB5 Handshake): Initiating Bandwidth Boost protocol...");

    // B_total = B_video + B_data + B_aux = 120 Gbps
    state.bandwidth_video = 80.0; // Dual-sided 8K stream
    state.bandwidth_data = 30.0;  // NPU agent synchronization
    state.bandwidth_aux = 10.0;   // Peripheral + sensor data
    let b_total = state.bandwidth_video + state.bandwidth_data + state.bandwidth_aux;

    println!("Rust (TB5 Handshake): Bandwidth allocated:");
    println!("  Video (dual 8K):  {} Gbps", state.bandwidth_video);
    println!("  Data (NPU sync):  {} Gbps", state.bandwidth_data);
    println!("  Aux (sensors):    {} Gbps", state.bandwidth_aux);
    println!("  Total:            {} Gbps / 120 Gbps", b_total);

    // Power delivery — 140W rapid charge
    state.is_charging = true;
    state.mode = ConnectionMode::BYOC;
    println!("Rust (TB5 Handshake): Power Delivery → 140W rapid charge initiated.");
    println!("Rust (TB5 Handshake): Mode → BYOC (external CPU/GPU active).");

    // Phase B: Trigger the "GRRN" boot animation
    println!("Rust (TB5 Handshake): Phase B — Liquid Glass ripple animation triggered.");
    println!("  Electrochromic refractive index shift: PRIMING pixels...");
    println!("  Visual: Glass appears to 'liquefy' before UI settles.");

    // Phase C: Gaze-Locked Authentication
    println!("Rust (TB5 Handshake): Phase C — Scanning for Optic ID / Ring signature...");
    println!("  Display in Ghost Mode (10% opacity) until gaze lock...");
    state.user_authenticated = true;
    state.authenticated_user = user_id.clone();
    println!("  Optic ID MATCHED: '{}'. UI crystallizing at gaze point.", user_id);

    // Haptic confirmation
    println!("Rust (TB5 Handshake): Edge Rail haptic hum → low-frequency confirmation.");

    Ok(format!("Thunderbolt 5 connected. User '{}' authenticated. BYOC mode active.", user_id))
}

/// Disconnect TB5 — seamless handover to Agentic mode via Wi-Fi 7 MLO
#[pyfunction]
pub fn thunderbolt_disconnect() -> PyResult<String> {
    let mut state = get_connection_state_inner().lock().unwrap();

    println!("Rust (TB5 Handshake): Thunderbolt 5 cable DISCONNECTED.");
    println!("Rust (TB5 Handshake): Initiating seamless state transition...");

    // Networking handover — Wi-Fi 7 MLO pre-negotiated
    state.wifi7_mlo_active = true;
    state.wifi7_latency_ms = 8.5; // Sub-10ms target
    state.mode = ConnectionMode::Agentic;
    state.is_charging = false;

    println!("Rust (Wi-Fi 7 MLO): Multi-Link Operation activated.");
    println!("  2.4 GHz band: ✓ Connected");
    println!("  5 GHz band:   ✓ Connected");
    println!("  6 GHz band:   ✓ Connected");
    println!("  Aggregate bandwidth: ~46 Gbps");
    println!("  Latency jump: < {}ms (pre-negotiated during wired phase)", state.wifi7_latency_ms);

    // Battery runtime calculation
    // P_total ≈ 28W (display 18W + NPU 7W + aux 3W)
    let p_display = 18.0_f32;
    let p_npu = 7.0_f32;
    let p_aux = 3.0_f32;
    let p_total = p_display + p_npu + p_aux;
    let runtime_min = (state.battery_wh / p_total) * 60.0;

    println!("Rust (Energy Budget): Battery = {} Wh", state.battery_wh);
    println!("  Display: {}W + NPU: {}W + Aux: {}W = {}W", p_display, p_npu, p_aux, p_total);
    println!("  Estimated runtime: {:.0} minutes (standard brightness)", runtime_min);

    // Reset bandwidth for wireless mode
    state.bandwidth_video = 30.0;
    state.bandwidth_data = 12.0;
    state.bandwidth_aux = 4.0;

    println!("Rust (TB5 Handshake): Mode → AGENTIC (internal NPU active).");
    println!("  Active documents cached on Edge Rail storage.");
    println!("  GRRN Agent: 'Switching to battery. High-performance for 18 min; Eco-Mode available.'");

    Ok(format!("Switched to Agentic mode. Wi-Fi 7 MLO active. Runtime: {:.0} min.", runtime_min))
}

/// Set battery power profile
#[pyfunction]
pub fn set_power_profile(profile: String) -> PyResult<String> {
    let parsed = match profile.to_lowercase().as_str() {
        "high" | "high_performance" | "performance" => PowerProfile::HighPerformance,
        "balanced" | "normal" => PowerProfile::Balanced,
        "eco" | "eco_mode" | "deep_obsidian" => PowerProfile::EcoMode,
        _ => {
            return Ok(format!("Error: Unknown profile '{}'. Use: high, balanced, eco", profile));
        }
    };

    let mut state = get_connection_state_inner().lock().unwrap();
    state.power_profile = parsed;

    match parsed {
        PowerProfile::HighPerformance => {
            println!("Rust (Power): Profile → HIGH PERFORMANCE. All subsystems unthrottled.");
        }
        PowerProfile::Balanced => {
            println!("Rust (Power): Profile → BALANCED. Moderate throttling enabled.");
        }
        PowerProfile::EcoMode => {
            println!("Rust (Power): Profile → ECO MODE ('Deep Obsidian').");
            println!("  Color palette: Maximizing OLED black pixels (0W per pixel).");
            println!("  Estimated runtime extension: ~90 minutes.");
        }
    }

    Ok(format!("Power profile set to: {:?}", parsed))
}

/// Get the full connection state as a diagnostic string
#[pyfunction]
pub fn get_connection_status() -> PyResult<String> {
    let state = get_connection_state_inner().lock().unwrap();
    let status = format!(
        "=== Connection & Power Status ===\n\
         Mode: {:?}\n\
         Power Profile: {:?}\n\
         Bandwidth: Video {:.0} Gbps / Data {:.0} Gbps / Aux {:.0} Gbps\n\
         Battery: {:.0} Wh ({:.0}%) {}\n\
         Wi-Fi 7 MLO: {} (Latency: {:.1}ms)\n\
         User: {} (Auth: {})\n",
        state.mode,
        state.power_profile,
        state.bandwidth_video, state.bandwidth_data, state.bandwidth_aux,
        state.battery_wh, state.battery_remaining_pct,
        if state.is_charging { "[CHARGING 140W]" } else { "" },
        if state.wifi7_mlo_active { "ACTIVE" } else { "STANDBY" },
        state.wifi7_latency_ms,
        state.authenticated_user,
        state.user_authenticated,
    );
    println!("{}", status);
    Ok(status)
}
