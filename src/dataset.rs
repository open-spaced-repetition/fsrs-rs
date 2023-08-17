use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FSRSItem {
    t_history: Vec<i64>,
    r_history: Vec<i64>,
    delta_t: f64,
    label: f64,
}

#[test]
fn test() {
    const JSON_FILE: &str = "tests/data/revlog_history.json";
    use burn::data::dataset::InMemDataset;
    use burn::data::dataloader::Dataset;
    let dataset = InMemDataset::<FSRSItem>::from_json_rows(JSON_FILE).unwrap();
    dbg!(&dataset.get(704));
}