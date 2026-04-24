// Rust GPU Rendering Engine (WGPU + Bevy)
// Modern, memory-safe high-performance visualization
// Supports DirectX 12, Vulkan, Metal, WebGPU

use bevy::prelude::*;
use bevy::render::{
    render_resource::{BufferBindingType, CachedRenderPipelineId, RenderPipeline},
    renderer::RenderContext,
};
use std::sync::Arc;
use std::sync::mpsc::Receiver;

// ============================================================================
// Market Data Structures
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct MarketTick {
    pub timestamp: u64,
    pub symbol: u32,
    pub price: f32,
    pub bid: f32,
    pub ask: f32,
    pub volume: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PricePoint {
    pub x: f32,
    pub y: f32,
    pub color: [f32; 3],
}

// ============================================================================
// GPU Buffers
// ============================================================================

#[derive(Resource)]
pub struct TikcBuffer {
    pub data: Vec<MarketTick>,
    pub buffer: Option<Arc<wgpu::Buffer>>,
    pub dirty: bool,
}

#[derive(Resource)]
pub struct PriceChartBuffer {
    pub points: Vec<PricePoint>,
    pub vertex_buffer: Option<Arc<wgpu::Buffer>>,
    pub index_buffer: Option<Arc<wgpu::Buffer>>,
    pub point_count: u32,
}

// ============================================================================
// Bevy Plugin
// ============================================================================

pub struct VeriTradeVisualizationPlugin;

impl Plugin for VeriTradeVisualizationPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_resource::<TikcBuffer>()
            .init_resource::<PriceChartBuffer>()
            .add_systems(Startup, setup_camera)
            .add_systems(Update, (
                update_price_buffers,
                render_price_charts,
                update_ui_metrics,
            ));
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

fn update_price_buffers(
    mut chart_buffer: ResMut<PriceChartBuffer>,
    tick_buffer: Res<TikcBuffer>,
) {
    if !tick_buffer.data.is_empty() {
        // Generate price points from last 1000 ticks
        let last_ticks = tick_buffer.data.iter().rev().take(1000).rev();
        
        let mut points = Vec::new();
        let mut idx = 0;
        
        for tick in last_ticks {
            points.push(PricePoint {
                x: idx as f32 * 0.01,
                y: (tick.price - 44000.0) / 1000.0, // Normalize
                color: [0.0, 1.0, 0.5], // Cyan for BTC
            });
            idx += 1;
        }
        
        chart_buffer.points = points;
        chart_buffer.dirty = true;
    }
}

fn render_price_charts(
    chart_buffer: Res<PriceChartBuffer>,
    mut gizmos: Gizmos,
) {
    // Draw price points as connected line
    for window in chart_buffer.points.windows(2) {
        if window.len() == 2 {
            let start = Vec3::new(window[0].x, window[0].y, 0.0);
            let end = Vec3::new(window[1].x, window[1].y, 0.0);
            let color = Color::rgb(window[0].color[0], window[0].color[1], window[0].color[2]);
            
            gizmos.line(start, end, color);
        }
    }
}

fn update_ui_metrics(
    tick_buffer: Res<TikcBuffer>,
    mut gizmos: Gizmos,
) {
    if let Some(last_tick) = tick_buffer.data.last() {
        // Display current price in top-left
        let spread_bps = (last_tick.ask - last_tick.bid) / last_tick.price * 10000.0;
        
        println!(
            "Symbol: {} | Price: ${:.2} | Bid/Ask: {:.2}/{:.2} | Spread: {:.1}bps | Volume: {}",
            last_tick.symbol,
            last_tick.price,
            last_tick.bid,
            last_tick.ask,
            spread_bps,
            last_tick.volume
        );
    }
}

// ============================================================================
// Market Data Streaming
// ============================================================================

pub struct DataStream {
    rx: Receiver<MarketTick>,
}

impl DataStream {
    pub fn new(rx: Receiver<MarketTick>) -> Self {
        Self { rx }
    }
    
    pub fn poll(&self, buffer: &mut TikcBuffer, max_items: usize) {
        let mut count = 0;
        while let Ok(tick) = self.rx.try_recv() {
            buffer.data.push(tick);
            buffer.dirty = true;
            count += 1;
            
            if count >= max_items {
                break;
            }
        }
        
        // Keep only last 1M ticks
        if buffer.data.len() > 1_000_000 {
            buffer.data.drain(0..buffer.data.len() - 1_000_000);
        }
    }
}
