#ifndef INCREMENTAL_MEAN_H
#define INCREMENTAL_MEAN_H

class IncrementalMean {
private:
    double sum;
    unsigned long count;

public:
    IncrementalMean();  // 构造函数声明
    void addValue(double value);  // 添加新值
    double getMean() const;  // 获取当前均值
    void reset();  // 重置计算
};

#endif // INCREMENTAL_MEAN_H