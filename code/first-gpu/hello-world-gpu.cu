#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <code/util.h>

int main(int argc, char *argv[]) {
    for (int i = 0; i < 10; ++i)
        printf("Hello world from iteration %d\n", i);

    return 0;
}
