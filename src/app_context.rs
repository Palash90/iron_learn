use std::sync::OnceLock;

#[derive(Debug)]
pub struct AppContext {
    pub app_name: &'static str,
    pub version: u32,
    pub gpu_enabled: bool,
    pub context: Option<cust::context::Context>,
}

// Declare a global mutable static
pub static GLOBAL_CONTEXT: OnceLock<AppContext> = OnceLock::new();

// Initialize the context
pub fn init_context(
    app_name: &'static str,
    version: u32,
    gpu_enabled: bool,
    context: Option<cust::context::Context>,
) {
    let ctx = AppContext {
        app_name,
        version,
        gpu_enabled,
        context,
    };
    GLOBAL_CONTEXT
        .set(ctx)
        .expect("Context can only be initialized once");
}
