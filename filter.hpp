#ifndef CONV_FILTER_H
#define CONV_FILTER_H

#include <cmath>
#include <memory>
#include <execution>
#include <string>

#include "NDGrid.hpp"

/**
 * v w(x);
 */
using v = std::vector<double>;

/**
 * matrix w(x, v(y));
 */
using matrix = std::vector<std::vector<double>>;

/**
 * tensor w(x, matrix(y,v(z)));
 */
using tensor = std::vector<std::vector<std::vector<double>>>;

template <typename F, typename T>
T for_matrix(F f, T m) {
    for_each(std::execution::par, m.begin(), m.end(), f);
    return m;
}

template <typename F, typename T>
T trans_matrix(F f, T &m) {
    transform(std::execution::par, m.begin(), m.end(),
            m.begin(), [f](auto v) -> auto {
                return f(v);
            });
    return m;
}

template <typename F, typename T>
T for_tensor(F f, T t) {
    for_each(std::execution::par, 
        t.begin(), t.end(),
        [f](auto m) {
            for_matrix(f, m);      
        });

    return t;
}

/**
 * Tensor reference value. 
 */
template <typename F, typename T>
T trans_tensor(F f, T &t) {
    transform(std::execution::par, 
        t.begin(), t.end(),
        t.begin(), [f](auto m) -> auto {
            return trans_matrix(f, m);
        });

    return t;
}

template <typename T>
auto sum(T t) {
    return reduce(std::execution::par, t.begin(), t.end());
}

// allocate memory for a tensor
tensor get_tensor(int x, int y, int z) {
    tensor w(x, matrix(y, v(z)));
    return w;
}

class filter {
public:
    tensor w;
    double b;     // krenel matrix, bias term
    int window, depth;

    static const size_t kDefaultSize = 3;

    filter() 
    : window(kDefaultSize), depth(kDefaultSize) {
        tensor w(window, matrix(window, v(depth)));
    }

    filter(int size) 
    : window(size), depth(size) {
        tensor w(window, matrix(window, v(depth)));
    }

    filter(int _window, int _depth) 
    : window(_window), depth(_depth) {
        tensor w(window, matrix(window, v(depth)));
    }

    filter(tensor _w, int _window, int _depth, int _b = 0) 
    : w(_w), window(_window), depth(_depth), b(_b) {
        
    }

    virtual ~filter() = default;

    // normalize the tensor
    void normalize() {
        double sum = 0;

        for_each(std::execution::par_unseq, w.begin(), w.end(), [&sum](auto m){
            for_each(std::execution::par_unseq, m.begin(), m.end(), [&sum](auto v){
                for_each(std::execution::par_unseq, v.begin(), v.end(), [&sum](auto v){
                    sum += std::abs(v);
                });
            });
        });

        for_each(std::execution::par_unseq, w.begin(), w.end(), [sum](auto m){
            for_each(std::execution::par_unseq, m.begin(), m.end(), [sum](auto v){
                for_each(std::execution::par_unseq, v.begin(), v.end(), [sum](auto v){
                    v /= sum;
                });
            });
        });
    }

};

#endif
