# RustNet 
RustNet is a simple yet efficient implementation of a feedforward neural network in Rust. 
It's specifically designed to demonstrate basic neural network operations such as forward propagation, 
backpropagation, and training, using matrix operations optimized with Rust's robust performance.

## About 
I wanted to figure out the math behind neural networks and what made them work. (no TensorFlow Timmy's). 
This was also my first real project using Rust besides Advent of Code, and it has really made me 
appreciate and consider using Rust in my projects. The compiler was of great help in helping me debug 
various issues in my implementation of gradient descent. I also loved traits and overriding operators to 
make the higher-level logic more terse. Overall, I would probably continue to write my machine learning
code in Rust if possible...

## Features
- Matrix Operations: Efficient implementations of matrix addition, subtraction, multiplication, and element-wise operations.
- Custom Neural Network: Flexible architecture with customizable layers, activation functions (sigmoid), and learning rate.
- Random Initialization: Utilizes random initialization for weights and biases.
- Training Loop: Provides a clear and structured training loop with epoch management.
- XOR Example: Includes a complete XOR gate training example.
- Performance Optimization: Reference-based operations that avoid unnecessary memory allocations, achieving up to 220% performance improvement in matrix operations.
- Benchmarking: Includes Criterion benchmarks to measure and compare performance improvements.

## Installation
Ensure you have Rust installed. If not, install it via rustup:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Clone the repository:
```
git clone https://github.com/yourusername/RustNet.git
cd RustNet
```
### To run the XOR example provided in main.rs, execute:
```
cargo run --release
```
The neural network will train for 10,000 epochs, after which it will display the XOR results. 
After successful training, the output will be similar to:
```
0 XOR 0: [0.0324]
0 XOR 1: [0.9653]
1 XOR 0: [0.9638]
1 XOR 1: [0.0456]
```

### Performance Benchmarking
Run benchmarks to see performance improvements from Rust optimization techniques:
```
cargo bench
```
This will show comparative performance between optimized (reference-based) and unoptimized (clone-heavy) implementations. The optimizations demonstrate advanced Rust concepts like borrowing and reference-based arithmetic operations, achieving significant performance gains:
- **Matrix Addition**: ~220% faster (3.2x performance improvement)
- **Neural Network Feedforward**: ~43% faster
- Reduced memory allocations and improved cache efficiency
### Customizing the Network:
Adjust the neural network's architecture and hyperparameters directly in main.rs:
`
let mut network = Network::new(vec![input_size, hidden_layer_size, output_size], learning_rate);
`

## License

This project is open-sourced under the [MIT License](https://opensource.org/license/mit).



