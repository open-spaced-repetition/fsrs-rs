use snafu::Snafu;

#[derive(Snafu, Debug)] // Removed PartialEq as candle_core::Error may not implement it.
pub enum FSRSError {
    NotEnoughData,
    Interrupted,
    InvalidParameters,
    OptimalNotFound,
    InvalidInput,
    InvalidDeckSize,
    #[snafu(display("Candle error: {source}"))]
    CandleError { source: candle_core::Error },
    #[snafu(display("Internal error: {message}"))] // Added for general internal errors
    Internal { message: String },
}

// Allow conversion from candle_core::Error to FSRSError
impl From<candle_core::Error> for FSRSError {
    fn from(source: candle_core::Error) -> Self {
        FSRSError::CandleError { source }
    }
}

pub type Result<T, E = FSRSError> = std::result::Result<T, E>;
