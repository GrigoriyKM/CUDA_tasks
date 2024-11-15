all: create_matrix.out transpose_matrix.out matrix_multiplication.out matrix_mult_shared.out 

create_matrix.out:
	nvcc create_matrix.cu -o create_matrix.out 

matrix_multiplication.out:
	nvcc matrix_multiplication.cu -o matrix_multiplication.out 

transpose_matrix.out:
	nvcc transpose_matrix.cu -o transpose_matrix.out 

matrix_mult_shared.out:
	nvcc matrix_mult_shared.cu -o matrix_mult_shared.out 

add_matrices.out:
	nvcc add_matrices.cu -o add_matrices.out 