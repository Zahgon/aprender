//! Pre-built simulation scenarios.
//!
//! Provides ready-to-use scenario templates for common simulation types:
//! - Rocket launch and stage separation
//! - Satellite orbital mechanics
//! - Pendulum systems (canonical physics example)
//! - Climate models (energy balance)
//! - Portfolio risk (Monte Carlo `VaR`)
//! - Epidemic models (SIR/SEIR)

pub mod arbiter;
pub mod bouncing_balls;
pub mod climate;
pub mod epidemic;
pub mod pendulum;
pub mod portfolio;
pub mod pr_flow;
pub mod rocket;
pub mod satellite;

pub use arbiter::{evaluate, parse_params, Params};
pub use bouncing_balls::{BouncingBallsConfig, BouncingBallsState};
pub use climate::{ClimateConfig, ClimateScenario};
pub use epidemic::{SEIRConfig, SEIRScenario, SIRConfig, SIRScenario};
pub use pendulum::{PendulumConfig, PendulumScenario};
pub use portfolio::{PortfolioConfig, PortfolioScenario, VaRResult};
pub use pr_flow::{
    binding_station, bounds, preset, reference_contract_ok, run_pr_flow, run_reference,
    PrFlowConfig, PrFlowResult, ReviewMode,
};
pub use rocket::{RocketConfig, RocketScenario, StageSeparation};
pub use satellite::{OrbitalElements, SatelliteScenario};
