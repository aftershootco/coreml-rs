use half::f16;
use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr};

#[derive(Debug)]
pub enum MLArray {
    Float32Array(ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>),
    Float16Array(ArrayBase<OwnedRepr<f16>, Dim<IxDynImpl>>),
    Int32Array(ArrayBase<OwnedRepr<i32>, Dim<IxDynImpl>>),
    Int16Array(ArrayBase<OwnedRepr<i16>, Dim<IxDynImpl>>),
    Int8Array(ArrayBase<OwnedRepr<i8>, Dim<IxDynImpl>>),
    UInt32Array(ArrayBase<OwnedRepr<u32>, Dim<IxDynImpl>>),
    UInt16Array(ArrayBase<OwnedRepr<u16>, Dim<IxDynImpl>>),
    UInt8Array(ArrayBase<OwnedRepr<u8>, Dim<IxDynImpl>>),
}

impl MLArray {
    pub fn shape(&self) -> &[usize] {
        match self {
            MLArray::Float32Array(array_base) => array_base.shape(),
            MLArray::Float16Array(array_base) => array_base.shape(),
            MLArray::Int32Array(array_base) => array_base.shape(),
            MLArray::Int16Array(array_base) => array_base.shape(),
            MLArray::Int8Array(array_base) => array_base.shape(),
            MLArray::UInt32Array(array_base) => array_base.shape(),
            MLArray::UInt16Array(array_base) => array_base.shape(),
            MLArray::UInt8Array(array_base) => array_base.shape(),
        }
    }
}

pub fn mean_absolute_error_bytes<
    T: core::ops::Sub<Output = T>
        + PartialOrd
        + Copy
        + core::ops::Add<Output = T>
        + num::cast::AsPrimitive<f64>
        + bytemuck::Pod
        + core::fmt::Debug,
>(
    lhs: &[u8],
    rhs: &[u8],
) -> f64 {
    let lhs = bytemuck::cast_slice(lhs);
    let rhs = bytemuck::cast_slice(rhs);

    assert_eq!(lhs.len(), rhs.len(), "lhs and rhs have different lengths");
    mean_absolute_error::<T>(lhs, rhs)
}

pub fn mean_absolute_error<
    T: core::ops::Sub<Output = T>
        + PartialOrd
        + Copy
        + core::ops::Add<Output = T>
        + num::cast::AsPrimitive<f64>,
>(
    lhs: impl AsRef<[T]>,
    rhs: impl AsRef<[T]>,
) -> f64 {
    let (sum, count) = lhs
        .as_ref()
        .iter()
        .zip(rhs.as_ref())
        .map(|(&l, &r)| if l > r { l - r } else { r - l })
        .fold((0f64, 0usize), |(acc, count), x| (acc + x.as_(), count + 1));
    sum / count as f64
}

/// Scalar types supported by the Rust tensor container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum MLDataType {
    F32 = 0,
    F16 = 1,
    I32 = 2,
    U16 = 3,
    U8 = 4,
    I16 = 5,
    I8 = 6,
    U32 = 7,
}

impl std::fmt::Display for MLDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::I32 => "i32",
            Self::U16 => "u16",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::U32 => "u32",
        })
    }
}

/// A checked extraction requested a different scalar type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("CoreML tensor type mismatch: expected {expected}, found {actual}")]
pub struct MLArrayTypeError {
    pub expected: MLDataType,
    pub actual: MLDataType,
}

mod sealed {
    use super::{MLArray, MLArrayTypeError};
    use ndarray::ArrayD;

    pub trait Sealed: Sized {
        fn wrap(array: ArrayD<Self>) -> MLArray;
        fn extract(array: MLArray) -> Result<ArrayD<Self>, MLArrayTypeError>;
    }
}

/// A supported concrete scalar. Sealed so a caller cannot mislabel an allocation.
pub trait MLType: sealed::Sealed {
    const DATA_TYPE: MLDataType;
}

macro_rules! scalar {
    ($ty:ty, $variant:ident, $dtype:ident) => {
        impl MLType for $ty {
            const DATA_TYPE: MLDataType = MLDataType::$dtype;
        }

        impl sealed::Sealed for $ty {
            fn wrap(array: ndarray::ArrayD<Self>) -> MLArray {
                MLArray::$variant(array)
            }

            fn extract(array: MLArray) -> Result<ndarray::ArrayD<Self>, MLArrayTypeError> {
                match array {
                    MLArray::$variant(array) => Ok(array),
                    other => Err(MLArrayTypeError {
                        expected: MLDataType::$dtype,
                        actual: other.dtype(),
                    }),
                }
            }
        }
    };
}

scalar!(f32, Float32Array, F32);
scalar!(f16, Float16Array, F16);
scalar!(i32, Int32Array, I32);
scalar!(u16, UInt16Array, U16);
scalar!(u8, UInt8Array, U8);
scalar!(i16, Int16Array, I16);
scalar!(i8, Int8Array, I8);
scalar!(u32, UInt32Array, U32);

impl<T: MLType> From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>> for MLArray {
    fn from(value: ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>) -> Self {
        <T as sealed::Sealed>::wrap(value)
    }
}

impl MLArray {
    pub fn dtype(&self) -> MLDataType {
        match self {
            Self::Float32Array(_) => MLDataType::F32,
            Self::Float16Array(_) => MLDataType::F16,
            Self::Int32Array(_) => MLDataType::I32,
            Self::UInt16Array(_) => MLDataType::U16,
            Self::UInt8Array(_) => MLDataType::U8,
            Self::Int16Array(_) => MLDataType::I16,
            Self::Int8Array(_) => MLDataType::I8,
            Self::UInt32Array(_) => MLDataType::U32,
        }
    }

    /// Moves out the original typed array without copying or reinterpreting storage.
    pub fn extract_to_tensor<T: MLType>(
        self,
    ) -> Result<Array<T, Dim<IxDynImpl>>, MLArrayTypeError> {
        <T as sealed::Sealed>::extract(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, ArrayD, IxDyn};

    fn roundtrip<T: MLType + Copy + PartialEq + std::fmt::Debug>(values: Vec<T>) {
        let expected = values.clone();
        let array = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap();
        let pointer = array.as_ptr();
        let wrapped = MLArray::from(array);
        assert_eq!(wrapped.dtype(), T::DATA_TYPE);
        let restored = wrapped.extract_to_tensor::<T>().unwrap();
        assert_eq!(restored.as_slice().unwrap(), expected);
        assert_eq!(
            restored.as_ptr(),
            pointer,
            "typed moves must retain the allocation"
        );
    }

    #[test]
    fn concrete_scalars_preserve_values_and_allocations() {
        roundtrip(vec![1.5_f32, -2.0, 0.0]);
        roundtrip(vec![f16::from_f32(-0.5), f16::from_f32(3.0)]);
        roundtrip(vec![i32::MIN, 0, i32::MAX]);
        roundtrip(vec![0_u16, u16::MAX]);
        roundtrip(vec![0_u8, u8::MAX]);
        roundtrip(vec![i16::MIN, i16::MAX]);
        roundtrip(vec![i8::MIN, i8::MAX]);
        roundtrip(vec![0_u32, u32::MAX]);
    }

    #[test]
    fn extraction_rejects_different_scalar_width_and_representation() {
        let byte = ArrayD::from_shape_vec(IxDyn(&[1]), vec![255_u8]).unwrap();
        assert_eq!(
            MLArray::from(byte).extract_to_tensor::<f32>().unwrap_err(),
            MLArrayTypeError {
                expected: MLDataType::F32,
                actual: MLDataType::U8
            }
        );
        let integer = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1_i32]).unwrap();
        assert_eq!(
            MLArray::from(integer)
                .extract_to_tensor::<f32>()
                .unwrap_err(),
            MLArrayTypeError {
                expected: MLDataType::F32,
                actual: MLDataType::I32
            }
        );
    }

    #[test]
    fn typed_moves_preserve_offset_negative_strides_and_logical_values() {
        let array = ndarray::Array::from_shape_vec((2, 3), vec![1_f32, 2., 3., 4., 5., 6.])
            .unwrap()
            .slice_move(s![..;-1, ..;-1])
            .into_dyn();
        let pointer = array.as_ptr();
        let strides = array.strides().to_vec();
        let restored = MLArray::from(array).extract_to_tensor::<f32>().unwrap();
        assert_eq!(restored.shape(), &[2, 3]);
        assert_eq!(restored.strides(), strides);
        assert_eq!(restored.as_ptr(), pointer);
        assert_eq!(
            restored.iter().copied().collect::<Vec<_>>(),
            [6., 5., 4., 3., 2., 1.]
        );
    }
}
