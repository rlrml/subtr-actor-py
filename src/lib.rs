use numpy::pyo3::IntoPy;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::{exceptions, wrap_pyfunction};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use subtr_actor::*;

#[pyfunction]
fn parse_replay<'p>(py: Python<'p>, data: &[u8]) -> PyResult<PyObject> {
    let replay = serde_json::to_value(replay_from_data(data)?).map_err(to_py_error)?;
    Ok(convert_to_py(py, &replay))
}

fn replay_from_data(data: &[u8]) -> PyResult<boxcars::Replay> {
    boxcars::ParserBuilder::new(data)
        .must_parse_network_data()
        .on_error_check_crc()
        .parse()
        .map_err(to_py_error)
}

#[pymodule]
fn subtr_actor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(parse_replay))?;
    m.add_wrapped(wrap_pyfunction!(get_ndarray_with_info_from_replay_filepath))?;
    m.add_wrapped(wrap_pyfunction!(get_replay_meta))?;
    m.add_wrapped(wrap_pyfunction!(get_column_headers))?;
    m.add_wrapped(wrap_pyfunction!(get_replay_frames_data))?;
    Ok(())
}

fn to_py_error<E: std::error::Error>(e: E) -> PyErr {
    PyErr::new::<exceptions::PyException, _>(format!("{}", e))
}

fn handle_frames_exception(e: subtr_actor::SubtrActorError) -> PyErr {
    PyErr::new::<exceptions::PyException, _>(format!("{:?} {}", e.variant, e.backtrace.to_string()))
}

fn convert_to_py(py: Python, value: &Value) -> PyObject {
    match value {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_py(py),
        Value::Number(n) => match n {
            n if n.is_u64() => n.as_u64().unwrap().into_py(py),
            n if n.is_i64() => n.as_i64().unwrap().into_py(py),
            n if n.is_f64() => n.as_f64().unwrap().into_py(py),
            _ => py.None(),
        },
        Value::String(s) => s.into_py(py),
        Value::Array(list) => {
            let list: Vec<PyObject> = list.iter().map(|e| convert_to_py(py, e)).collect();
            list.into_py(py)
        }
        Value::Object(m) => {
            let mut map = BTreeMap::new();
            m.iter().for_each(|(k, v)| {
                map.insert(k, convert_to_py(py, v));
            });
            map.into_py(py)
        }
    }
}

pub static DEFAULT_GLOBAL_FEATURE_ADDERS: [&str; 1] = ["BallRigidBody"];

pub static DEFAULT_PLAYER_FEATURE_ADDERS: [&str; 3] =
    ["PlayerRigidBody", "PlayerBoost", "PlayerAnyJump"];

#[pyfunction]
fn get_ndarray_with_info_from_replay_filepath<'p>(
    py: Python<'p>,
    filepath: PathBuf,
    global_feature_adders: Option<Vec<String>>,
    player_feature_adders: Option<Vec<String>>,
    fps: Option<f32>,
) -> PyResult<PyObject> {
    let data = std::fs::read(filepath.as_path()).map_err(to_py_error)?;
    let replay = replay_from_data(&data)?;

    let mut collector = build_ndarray_collector(global_feature_adders, player_feature_adders)
        .map_err(handle_frames_exception)?;

    FrameRateDecorator::new_from_fps(fps.unwrap_or(10.0), &mut collector)
        .process_replay(&replay)
        .map_err(handle_frames_exception)?;

    let (replay_meta_with_headers, rust_nd_array) = collector
        .get_meta_and_ndarray()
        .map_err(handle_frames_exception)?;

    let python_replay_meta = convert_to_py(
        py,
        &serde_json::to_value(&replay_meta_with_headers).map_err(to_py_error)?,
    );

    let python_nd_array = rust_nd_array.into_pyarray(py);
    Ok((python_replay_meta, python_nd_array).into_py(py))
}

fn build_ndarray_collector(
    global_feature_adders: Option<Vec<String>>,
    player_feature_adders: Option<Vec<String>>,
) -> subtr_actor::SubtrActorResult<subtr_actor::NDArrayCollector<f32>> {
    let global_feature_adders = global_feature_adders.unwrap_or_else(|| {
        DEFAULT_GLOBAL_FEATURE_ADDERS
            .iter()
            .map(|i| i.to_string())
            .collect()
    });
    let player_feature_adders = player_feature_adders.unwrap_or_else(|| {
        DEFAULT_PLAYER_FEATURE_ADDERS
            .iter()
            .map(|i| i.to_string())
            .collect()
    });
    let global_feature_adders: Vec<&str> = global_feature_adders.iter().map(|s| &s[..]).collect();
    let player_feature_adders: Vec<&str> = player_feature_adders.iter().map(|s| &s[..]).collect();
    subtr_actor::NDArrayCollector::<f32>::from_strings(
        &global_feature_adders,
        &player_feature_adders,
    )
}

#[pyfunction]
fn get_replay_meta<'p>(
    py: Python<'p>,
    filepath: PathBuf,
    global_feature_adders: Option<Vec<String>>,
    player_feature_adders: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let data = std::fs::read(filepath.as_path()).map_err(to_py_error)?;
    let replay = replay_from_data(&data)?;

    let mut collector = build_ndarray_collector(global_feature_adders, player_feature_adders)
        .map_err(handle_frames_exception)?;

    let replay_meta = collector
        .process_and_get_meta_and_headers(&replay)
        .map_err(handle_frames_exception)?;

    Ok(convert_to_py(
        py,
        &serde_json::to_value(&replay_meta).map_err(to_py_error)?,
    ))
}

#[pyfunction]
fn get_column_headers<'p>(
    py: Python<'p>,
    global_feature_adders: Option<Vec<String>>,
    player_feature_adders: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let header_info = build_ndarray_collector(global_feature_adders, player_feature_adders)
        .map_err(handle_frames_exception)?
        .get_column_headers();
    Ok(convert_to_py(
        py,
        &serde_json::to_value(&header_info).map_err(to_py_error)?,
    ))
}

#[pyfunction]
fn get_replay_frames_data<'p>(py: Python<'p>, filepath: PathBuf) -> PyResult<PyObject> {
    let data = std::fs::read(filepath.as_path()).map_err(to_py_error)?;
    let replay = replay_from_data(&data)?;

    let replay_data = subtr_actor::ReplayDataCollector::new()
        .get_replay_data(&replay)
        .map_err(handle_frames_exception)?;

    Ok(convert_to_py(
        py,
        &serde_json::to_value(replay_data).map_err(to_py_error)?,
    ))
}
