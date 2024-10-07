filter_shared.out:
	nvcc filter_shared.cu -o filter_shared.out `pkg-config --cflags --libs opencv`

clean:
	rm ./filter_global.out ./filter_shared.out