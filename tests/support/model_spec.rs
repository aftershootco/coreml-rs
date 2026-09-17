// Minimal protobuf fixtures from Apple's Core ML format:
// https://github.com/apple/coremltools/tree/main/mlmodel/format
// The fixture has one FLOAT32 input, one FLOAT32 output, and a ReLU layer.
fn varint(mut value: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    while value >= 128 {
        bytes.push((value as u8 & 127) | 128);
        value >>= 7;
    }
    bytes.push(value as u8);
    bytes
}

fn message(field: usize, bytes: &[u8]) -> Vec<u8> {
    [varint(field * 8 + 2), varint(bytes.len()), bytes.to_vec()].concat()
}

fn feature(name: &[u8]) -> Vec<u8> {
    let array = [vec![8, 1, 16], varint(65568)].concat();
    [message(1, name), message(3, &message(5, &array))].concat()
}

pub fn model_spec() -> Vec<u8> {
    let description = [
        message(1, &feature(b"input")),
        message(10, &feature(b"output")),
    ]
    .concat();
    let operation = message(130, &message(10, &[]));
    let layer = [
        message(1, b"test_layer"),
        message(2, b"input"),
        message(3, b"output"),
        operation,
    ]
    .concat();
    [
        vec![8, 2],
        message(2, &description),
        message(500, &message(1, &layer)),
    ]
    .concat()
}
