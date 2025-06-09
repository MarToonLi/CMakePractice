#include "A_101_120_add.h"

// 类方法定义
IncrementalMean::IncrementalMean() : sum(0.0), count(0) {}

void IncrementalMean::addValue(double value) {
    sum += value;
    ++count;
}

double IncrementalMean::getMean() const {
    return count == 0 ? 0.0 : sum / count;
}

void IncrementalMean::reset() {
    sum = 0.0;
    count = 0;
}
