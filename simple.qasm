OPENQASM 3.0;

// Define a quantum register with 2 qubits
qubit[2] q;

// Define a classical register with 2 bits
bit[2] c;

// Apply a Hadamard gate to the first qubit
h q[0];

// Apply a controlled-NOT gate with q[0] as control and q[1] as target
cx q[0], q[1];

// Measure both qubits into classical bits
c[0] = measure q[0];
c[1] = measure q[1];
