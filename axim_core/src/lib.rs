use pyo3::prelude::*;
use std::sync::OnceLock;

mod display_manager;
mod thunderbolt_handshake;

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("vision.h");
        include!("kv_cache.h");
        include!("edge_rail_thermal.h");
        include!("ghost_touch.h");

        // --- Vision & KV Cache (original) ---
        fn execute_vision_check() -> bool;
        fn apply_kv_cache_quantization(precision: &str);

        // --- Edge Rail Thermal Management ---
        fn calculate_thermal_load(display_power: f32, display_efficiency: f32,
                                  compute_power: f32, compute_efficiency: f32) -> f32;
        fn calculate_battery_runtime(battery_wh: f32, display_power: f32,
                                     npu_power: f32, aux_power: f32) -> f32;
        fn is_cooling_sufficient(thermal_load: f32, max_tdp: f32) -> bool;
        fn get_cooling_status(thermal_load: f32, max_tdp: f32, glass_temp: f32) -> String;

        // --- Ghost-Touch Protection & Haptics ---
        fn classify_touch_side(ir_intensity: f32, pressure: f32) -> i32;
        fn is_authorized_touch(is_front_side: bool, side_b_permitted: bool) -> bool;
        fn filter_ghost_touch(pressure_newtons: f32) -> bool;
        fn trigger_lateral_wave(start_x: f32, end_x: f32, duration_ms: f32);
        fn trigger_ripple_barrier(x_coord: f32, y_coord: f32);
        fn get_touch_diagnostics(active_touches_a: i32, active_touches_b: i32,
                                 ghost_protection_active: bool) -> String;
    }
}

// Global state to ensure ROS 2 context and publisher are kept alive.
// This prevents RCL_RET_ALREADY_INIT crashes and solves the DDS discovery drop issue.
struct AximRosNode {
    _context: rclrs::Context,
    _node: rclrs::Node,
    publisher: rclrs::Publisher<std_msgs::msg::String>,
}

// In rclrs, Nodes and Contexts are generally safe to wrap in OnceLock for global access
// Assuming we only publish from the main Python thread.
static ROS_NODE: OnceLock<AximRosNode> = OnceLock::new();

fn init_ros_node() -> &'static AximRosNode {
    ROS_NODE.get_or_init(|| {
        let context = rclrs::Context::new(std::env::args()).unwrap();
        let node = rclrs::create_node(&context, "axim_core_node").unwrap();
        let publisher = node.create_publisher::<std_msgs::msg::String>("axim_commands", rclrs::QOS_PROFILE_DEFAULT).unwrap();
        
        // Brief sleep to allow DDS discovery to complete before publishing happens
        std::thread::sleep(std::time::Duration::from_millis(500));
        
        AximRosNode {
            _context: context,
            _node: node,
            publisher,
        }
    })
}

/// The main logic called from Python, backed by Rust's memory safety.
#[pyfunction]
fn lock_station() -> PyResult<()> {
    println!("Rust (Axim Skeleton): Received lock command. Verifying user authorization...");
    
    // Perform memory-safe operations here...
    let authorized = true; // Assume success for this skeleton

    if authorized {
        println!("Rust (Axim Skeleton): User authorized. Calling AXIM compute kernel (Vulkan/Metal/SIMD)...");

        // Call the C++ AXIM engine room safely using cxx.
        // Vendor-neutral: Vulkan/Metal on GPU, AVX-512/NEON on CPU. Zero CUDA.
        let success = ffi::execute_vision_check();
        
        if success {
            println!("Rust (Axim Skeleton): Vision check passed (Stella UWB locked).");
            println!("Rust (Axim Skeleton): Sending 'Close Latches' to Arduino Portenta H7 M4 core via micro-ROS...");
            
            // Micro-ROS rclrs publishing to Portenta H7
            let ros_setup = init_ros_node();
            let msg = std_msgs::msg::String {
                data: "Close Latches".to_string(),
            };
            ros_setup.publisher.publish(&msg).unwrap();
            println!("Rust (Axim Skeleton): Message successfully published to Arduino!");
        } else {
            println!("Rust (Axim Skeleton): Vision check failed! Aborting lock.");
        }
    } else {
        println!("Rust (Axim Skeleton): Authorization failed! DefenseClaw will log this attempt.");
    }
    
    Ok(())
}

/// Dynamically adjust the KV cache quantization precision for all Axim long-context models.
#[pyfunction]
fn set_memory_quantization(precision: String) -> PyResult<()> {
    println!("Rust (Axim Skeleton): Establishing secure KV cache pool with {} precision...", precision);
    ffi::apply_kv_cache_quantization(&precision);
    Ok(())
}

// ---- Edge Rail Thermal (Python wrappers for C++ FFI) ----

#[pyfunction]
fn get_thermal_load(display_power: f32, display_efficiency: f32,
                    compute_power: f32, compute_efficiency: f32) -> PyResult<f32> {
    Ok(ffi::calculate_thermal_load(display_power, display_efficiency,
                                   compute_power, compute_efficiency))
}

#[pyfunction]
fn get_battery_runtime(battery_wh: f32, display_power: f32,
                       npu_power: f32, aux_power: f32) -> PyResult<f32> {
    Ok(ffi::calculate_battery_runtime(battery_wh, display_power, npu_power, aux_power))
}

#[pyfunction]
fn check_cooling(thermal_load: f32, max_tdp: f32) -> PyResult<bool> {
    Ok(ffi::is_cooling_sufficient(thermal_load, max_tdp))
}

#[pyfunction]
fn cooling_status(thermal_load: f32, max_tdp: f32, glass_temp: f32) -> PyResult<String> {
    Ok(ffi::get_cooling_status(thermal_load, max_tdp, glass_temp))
}

// ---- Ghost-Touch (Python wrappers for C++ FFI) ----

#[pyfunction]
fn touch_classify(ir_intensity: f32, pressure: f32) -> PyResult<i32> {
    Ok(ffi::classify_touch_side(ir_intensity, pressure))
}

#[pyfunction]
fn touch_authorize(is_front_side: bool, side_b_permitted: bool) -> PyResult<bool> {
    Ok(ffi::is_authorized_touch(is_front_side, side_b_permitted))
}

#[pyfunction]
fn touch_filter_ghost(pressure_newtons: f32) -> PyResult<bool> {
    Ok(ffi::filter_ghost_touch(pressure_newtons))
}

#[pyfunction]
fn haptic_lateral_wave(start_x: f32, end_x: f32, duration_ms: f32) -> PyResult<()> {
    ffi::trigger_lateral_wave(start_x, end_x, duration_ms);
    Ok(())
}

#[pyfunction]
fn haptic_ripple_barrier(x_coord: f32, y_coord: f32) -> PyResult<()> {
    ffi::trigger_ripple_barrier(x_coord, y_coord);
    Ok(())
}

#[pyfunction]
fn touch_diagnostics(active_a: i32, active_b: i32, ghost_active: bool) -> PyResult<String> {
    Ok(ffi::get_touch_diagnostics(active_a, active_b, ghost_active))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn axim_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Original functions
    m.add_function(wrap_pyfunction!(lock_station, m)?)?;
    m.add_function(wrap_pyfunction!(set_memory_quantization, m)?)?;

    // Display Manager (Rust-native)
    m.add_function(wrap_pyfunction!(display_manager::set_rendering_mode, m)?)?;
    m.add_function(wrap_pyfunction!(display_manager::set_electrochromic_state, m)?)?;
    m.add_function(wrap_pyfunction!(display_manager::set_brightness, m)?)?;
    m.add_function(wrap_pyfunction!(display_manager::activate_shared_surface, m)?)?;
    m.add_function(wrap_pyfunction!(display_manager::get_display_status, m)?)?;

    // Thunderbolt Handshake (Rust-native)
    m.add_function(wrap_pyfunction!(thunderbolt_handshake::thunderbolt_connect, m)?)?;
    m.add_function(wrap_pyfunction!(thunderbolt_handshake::thunderbolt_disconnect, m)?)?;
    m.add_function(wrap_pyfunction!(thunderbolt_handshake::set_power_profile, m)?)?;
    m.add_function(wrap_pyfunction!(thunderbolt_handshake::get_connection_status, m)?)?;

    // Edge Rail Thermal (C++ FFI wrappers)
    m.add_function(wrap_pyfunction!(get_thermal_load, m)?)?;
    m.add_function(wrap_pyfunction!(get_battery_runtime, m)?)?;
    m.add_function(wrap_pyfunction!(check_cooling, m)?)?;
    m.add_function(wrap_pyfunction!(cooling_status, m)?)?;

    // Ghost-Touch & Haptics (C++ FFI wrappers)
    m.add_function(wrap_pyfunction!(touch_classify, m)?)?;
    m.add_function(wrap_pyfunction!(touch_authorize, m)?)?;
    m.add_function(wrap_pyfunction!(touch_filter_ghost, m)?)?;
    m.add_function(wrap_pyfunction!(haptic_lateral_wave, m)?)?;
    m.add_function(wrap_pyfunction!(haptic_ripple_barrier, m)?)?;
    m.add_function(wrap_pyfunction!(touch_diagnostics, m)?)?;

    Ok(())
}
