// ============================================================================
// Axim GlassOS — Display Manager (Rust Core)
// ============================================================================
// Manages the dual-sided transparent display system:
//   - Three-layer optical stack (Front Active Matrix + Electrochromic Core + Rear Active Matrix)
//   - Three rendering modes (Holographic, SplitPrivacy, Presentation)
//   - Electrochromic state control (transparency, tint, opacity)
//   - Split-work orchestration between Side-A and Side-B users
// ============================================================================

use pyo3::prelude::*;
use std::sync::Mutex;
use std::sync::OnceLock;

// ---- Enums ----

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderingMode {
    /// "Holographic" Shared Mode — electrochromic core transparent.
    /// Both users see 3D content; text rendered independently per side.
    Holographic,
    /// Split Privacy Mode — electrochromic core opaque.
    /// Each side runs a completely independent desktop environment.
    SplitPrivacy,
    /// Presentation Mode — directional rendering.
    /// Side-A sees controls + private notes; Side-B sees only presentation content.
    Presentation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElectrochromicState {
    /// 100% transparent — glass appears as clear slab
    Transparent,
    /// Partially tinted — reduces ambient light bleed
    Tinted,
    /// Fully opaque — black, no light passes through
    Opaque,
    /// Bottom half opaque, top half transparent — hides keyboard, maintains eye contact
    HalfMast,
    /// Opaque backing only behind the active window — max brightness (4,500 nits)
    ShadowBox,
}

// ---- Global State ----

struct DisplayState {
    rendering_mode: RenderingMode,
    electrochromic_state: ElectrochromicState,
    side_a_active: bool,
    side_b_active: bool,
    brightness_nits: u32,
    shared_surface_zones: Vec<(f32, f32, f32, f32)>, // (x, y, width, height) bounding boxes
}

static DISPLAY_STATE: OnceLock<Mutex<DisplayState>> = OnceLock::new();

fn get_display_state() -> &'static Mutex<DisplayState> {
    DISPLAY_STATE.get_or_init(|| {
        Mutex::new(DisplayState {
            rendering_mode: RenderingMode::Holographic,
            electrochromic_state: ElectrochromicState::Transparent,
            side_a_active: true,
            side_b_active: false,
            brightness_nits: 600, // Professional indoor standard
            shared_surface_zones: Vec::new(),
        })
    })
}

// ---- PyO3 Functions ----

#[pyfunction]
pub fn set_rendering_mode(mode: String) -> PyResult<String> {
    let parsed_mode = match mode.to_lowercase().as_str() {
        "holographic" => RenderingMode::Holographic,
        "split_privacy" | "split" | "privacy" => RenderingMode::SplitPrivacy,
        "presentation" | "present" => RenderingMode::Presentation,
        _ => {
            return Ok(format!("Error: Unknown rendering mode '{}'. Use: holographic, split_privacy, presentation", mode));
        }
    };

    let mut state = get_display_state().lock().unwrap();
    state.rendering_mode = parsed_mode;

    // Auto-adjust electrochromic core based on mode
    match parsed_mode {
        RenderingMode::Holographic => {
            state.electrochromic_state = ElectrochromicState::Transparent;
            println!("Rust (Display Manager): Mode → HOLOGRAPHIC. Electrochromic core → TRANSPARENT.");
            println!("  Front Active Matrix: Rendering 3D content (User A text orientation).");
            println!("  Rear Active Matrix: Rendering 3D content (User B text orientation).");
        }
        RenderingMode::SplitPrivacy => {
            state.electrochromic_state = ElectrochromicState::Opaque;
            println!("Rust (Display Manager): Mode → SPLIT PRIVACY. Electrochromic core → OPAQUE.");
            println!("  Front Active Matrix: Independent Desktop A.");
            println!("  Rear Active Matrix: Independent Desktop B.");
        }
        RenderingMode::Presentation => {
            state.electrochromic_state = ElectrochromicState::Tinted;
            println!("Rust (Display Manager): Mode → PRESENTATION. Electrochromic core → TINTED.");
            println!("  Front Active Matrix: Presenter controls + private notes (User A).");
            println!("  Rear Active Matrix: Mirrored presentation content (User B).");
        }
    }

    Ok(format!("Rendering mode set to: {:?}", parsed_mode))
}

#[pyfunction]
pub fn set_electrochromic_state(state_name: String) -> PyResult<String> {
    let parsed_state = match state_name.to_lowercase().as_str() {
        "transparent" | "clear" => ElectrochromicState::Transparent,
        "tinted" => ElectrochromicState::Tinted,
        "opaque" | "black" => ElectrochromicState::Opaque,
        "half_mast" | "halfmast" => ElectrochromicState::HalfMast,
        "shadow_box" | "shadowbox" => ElectrochromicState::ShadowBox,
        _ => {
            return Ok(format!("Error: Unknown state '{}'. Use: transparent, tinted, opaque, half_mast, shadow_box", state_name));
        }
    };

    let mut state = get_display_state().lock().unwrap();
    state.electrochromic_state = parsed_state;

    match parsed_state {
        ElectrochromicState::Transparent => {
            println!("Rust (Display Manager): Electrochromic core → 100% TRANSPARENT.");
        }
        ElectrochromicState::Tinted => {
            println!("Rust (Display Manager): Electrochromic core → TINTED (ambient reduction).");
        }
        ElectrochromicState::Opaque => {
            println!("Rust (Display Manager): Electrochromic core → 100% OPAQUE (Jet Black).");
        }
        ElectrochromicState::HalfMast => {
            println!("Rust (Display Manager): Electrochromic core → HALF-MAST.");
            println!("  Bottom 50%: 95% opaque (hides keyboard/hands).");
            println!("  Top 50%: Transparent (face-to-face eye contact maintained).");
        }
        ElectrochromicState::ShadowBox => {
            println!("Rust (Display Manager): Electrochromic core → SHADOW-BOX.");
            println!("  Active window backing: OPAQUE (4,500 nits peak brightness).");
            println!("  Remaining area: TRANSPARENT.");
            println!("  Side-B sees: sleek black slate behind window region only.");
        }
    }

    Ok(format!("Electrochromic state set to: {:?}", parsed_state))
}

#[pyfunction]
pub fn set_brightness(nits: u32) -> PyResult<String> {
    let mut state = get_display_state().lock().unwrap();
    state.brightness_nits = nits;
    println!("Rust (Display Manager): Brightness set to {} nits.", nits);
    Ok(format!("Brightness: {} nits", nits))
}

#[pyfunction]
pub fn activate_shared_surface(x: f32, y: f32, width: f32, height: f32) -> PyResult<String> {
    let mut state = get_display_state().lock().unwrap();
    state.shared_surface_zones.push((x, y, width, height));
    state.side_b_active = true;

    println!("Rust (Display Manager): SHARED SURFACE activated.");
    println!("  Zone: ({}, {}) {}×{}", x, y, width, height);
    println!("  Side-B touch: ENABLED for this bounding box only.");
    println!("  Ghost-Touch Protection: Suspended for zone; active elsewhere.");
    println!("  Admin Kill-Switch: Side-A retains control.");

    Ok(format!("Shared Surface zone added at ({}, {})", x, y))
}

#[pyfunction]
pub fn get_display_status() -> PyResult<String> {
    let state = get_display_state().lock().unwrap();
    let status = format!(
        "=== Display Manager Status ===\n\
         Rendering Mode: {:?}\n\
         Electrochromic State: {:?}\n\
         Brightness: {} nits\n\
         Side-A Active: {}\n\
         Side-B Active: {}\n\
         Shared Surface Zones: {}\n",
        state.rendering_mode,
        state.electrochromic_state,
        state.brightness_nits,
        state.side_a_active,
        state.side_b_active,
        state.shared_surface_zones.len()
    );
    println!("{}", status);
    Ok(status)
}
