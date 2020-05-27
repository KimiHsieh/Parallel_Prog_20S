all: student_submission reference_implementation

reference_implementation:
	g++ -std=c++17 -Wall -Wextra -o sequential_implementation -O3 -g sequential_implementation.cpp -lpthread -lm

student_submission:
	g++ -std=c++17 -Wall -Wextra -o student_submission -O3 -g student_submission.cpp -lpthread -lm

clean:
	rm -f sequential_implementation student_submission mandelbrot.ppm
