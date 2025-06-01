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

// Manual Clone implementation since candle_core::Error doesn't implement Clone
impl Clone for FSRSError {
    fn clone(&self) -> Self {
        match self {
            FSRSError::NotEnoughData => FSRSError::NotEnoughData,
            FSRSError::Interrupted => FSRSError::Interrupted,
            FSRSError::InvalidParameters => FSRSError::InvalidParameters,
            FSRSError::OptimalNotFound => FSRSError::OptimalNotFound,
            FSRSError::InvalidInput => FSRSError::InvalidInput,
            FSRSError::InvalidDeckSize => FSRSError::InvalidDeckSize,
            FSRSError::CandleError { source } => {
                // Convert the error to a string and create an Internal error
                FSRSError::Internal { message: format!("Candle error: {}", source) }
            },
            FSRSError::Internal { message } => FSRSError::Internal { message: message.clone() },
        }
    }
}

// Allow conversion from candle_core::Error to FSRSError
impl From<candle_core::Error> for FSRSError {
    fn from(source: candle_core::Error) -> Self {
        FSRSError::CandleError { source }
    }
}

pub type Result<T, E = FSRSError> = std::result::Result<T, E>;
