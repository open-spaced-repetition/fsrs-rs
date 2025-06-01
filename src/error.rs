use snafu::Snafu;

#[derive(Snafu, Debug, PartialEq, Clone)]
pub enum FSRSError {
    NotEnoughData,
    Interrupted,
    InvalidParameters,
    OptimalNotFound,
    InvalidInput,
    InvalidDeckSize,
    #[snafu(display("Candle error: {message}"))]
    CandleError { message: String },
    #[snafu(display("Internal error: {message}"))] // Added for general internal errors
    Internal { message: String },
}

// Allow conversion from candle_core::Error to FSRSError
impl From<candle_core::Error> for FSRSError {
    fn from(source: candle_core::Error) -> Self {
        FSRSError::CandleError { message: source.to_string() }
    }
}

pub type Result<T, E = FSRSError> = std::result::Result<T, E>;
