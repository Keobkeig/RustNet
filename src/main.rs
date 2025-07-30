use rust_neural_network::{Matrix, Network};

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // 2 inputs, one hidden layer with 3 neurons, 1 output neuron.
    let mut network = Network::new(vec![2, 3, 1], 0.5);

    network.train(inputs, targets, 10_000);

    println!(
        "0 XOR 0: {:?}",
        network.feedforward(Matrix::from_vec(vec![0.0, 0.0])).data
    );
    println!(
        "0 XOR 1: {:?}",
        network.feedforward(Matrix::from_vec(vec![0.0, 1.0])).data
    );
    println!(
        "1 XOR 0: {:?}",
        network.feedforward(Matrix::from_vec(vec![1.0, 0.0])).data
    );
    println!(
        "1 XOR 1: {:?}",
        network.feedforward(Matrix::from_vec(vec![1.0, 1.0])).data
    );
}
