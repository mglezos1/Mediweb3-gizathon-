mod chunk0;

use orion::operators::tensor::{I32Tensor, Tensor, TensorTrait};


fn get_node_new_shape() -> Tensor<i32> {
    let mut shape = array![2];

    let mut data = array![];
     chunk0::compute(ref data);

    TensorTrait::new(shape.span(), data.span())
}