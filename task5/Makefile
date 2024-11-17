all: create_matrix.out transpose_matrix.out matrix_multiplication.out matrix_mult_shared.out matrix_mult_cublas.out matrix_mult_wmma.out

create_matrix.out:
	nvcc create_matrix.cu -o create_matrix.out 

matrix_multiplication.out:
	nvcc -g matrix_multiplication.cu -o matrix_multiplication.out 

transpose_matrix.out:
	nvcc transpose_matrix.cu -o transpose_matrix.out 

matrix_mult_shared.out:
	nvcc -g matrix_mult_shared.cu -o matrix_mult_shared.out 

add_matrices.out:
	nvcc add_matrices.cu -o add_matrices.out 

matrix_mul_cublas.out:
	nvcc -g matrix_mul_cublas.cu -o matrix_mul_cublas.out -lcublas

matrix_mul_wmma.out:
	nvcc -arch=sm_70 -g matrix_mul_wmma.cu -o matrix_mul_wmma.out 